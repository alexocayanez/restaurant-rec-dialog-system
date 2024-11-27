from typing import Any, List, Dict

import mlflow
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from data import Data
from utils import preprocess_utterances, create_word_vocabulary

class ClassifierModel(mlflow.pyfunc.PythonModel):

    PREPROCESS_SETTINGS = {
        "min_word_length": 1,
        "stopwords": [],
        "stemming": True
    }
     
    def __init__(self, parameters: Dict[str, Any], pipeline: Pipeline):
        self.parameters = parameters
        self.pipeline = pipeline
        self.trained = False
        
    def fit(self, x_train: List[str], y_train: List[str]) -> None:
        """
        Performs preprocessing according to settings class dictionary. 
        Set the parameters in class pipeline and fit it with the provided train data.

        :param x_train: List of utterances in the training set.
        :param y_train: List of class tags in the training set.
        """
        x_train = preprocess_utterances(x_train,
                                        min_word_length=self.PREPROCESS_SETTINGS['min_word_length'],
                                        stopwords=self.PREPROCESS_SETTINGS['stopwords'],
                                        stemming=self.PREPROCESS_SETTINGS['stemming']
                                        )
        y_train = [Data.DIALOG_ACTS.index(tag) for tag in y_train]

        self.pipeline.set_params(vect__vocabulary=create_word_vocabulary(x_train))
        self.pipeline.set_params(**self.parameters)
        self.pipeline.fit(x_train, np.array(y_train, dtype=int))

        setattr(self, "trained", True)
 

    def predict(self, utterances: List[str]) -> List[str]:
        """ 
        Return the prediction for a list of utterances. 
        
        :param utterance: A list of raw text dialog system utterances.
        :return: A list of predicted classes' tags of the utterance by the model, or None if the model is not trained.
        """
        if self.trained:
            predicted_tags = self.pipeline.predict((preprocess_utterances(utterances,
                                                    min_word_length=self.PREPROCESS_SETTINGS['min_word_length'],
                                                    stopwords=self.PREPROCESS_SETTINGS['stopwords'],
                                                    stemming=self.PREPROCESS_SETTINGS['stemming'])))
            return [Data.DIALOG_ACTS[tag] for tag in predicted_tags]
        else:
            print('Please train the model first before making predictions.')
            return None
    
    def predict_tag(self, utterance: str) -> str:
        """ 
        Return the prediction for a single utterance. 
        
        :param utterance: The dialog system utterance as raw user input.
        :return: The tag of the predicted class of the utterance by the model, or None if the model is not trained.
        """
        if self.trained:
            utterance_list = []
            utterance_list.append(utterance)
            return self.predict(utterances=utterance_list)[0]
        else:
            print('Please train the model first before making predictions.')
            return None
        

class LogResModel(ClassifierModel):
    ALLOWED_PARAMETER_NAMES = ['C', 'penalty']
    ALLOWED_PARAMETER_NAMES_WITH_TAGS = ['lr__C', 'lr__penalty']
    SOLVERS = [ "saga", # Provides most options for penalty parameter
                "sag",
                "newton-cg",
                "lbfgs" # Default for sklearn
            ]

    def __init__(self, **parameters: dict[str, Any]) -> None:
        """
        :param C (float>=0, default=0): The inverse of the regularization strenght.
        :param penalty ({"l1", "l2", "elasticnet", None}, default="l2"): The penalty applied in regularization
        """
        self.name = "logistic_regression"
        self.registered_name = "log_res_model"
        self.title_name= "Logistic Regression"
        self.tag = "lr"
        self.solver='saga' #Default solver

        # Create a dictionary with the set parameters and raise TypeError if one of them is not in the allowed dictionary.
        parameters_dict = {"lr__C": 0, "lr__penalty": "l2"}
        for key, value in parameters.items():
            if key in self.ALLOWED_PARAMETER_NAMES:
                parameters_dict[self.tag+'__'+key] = value
            elif key in self.ALLOWED_PARAMETER_NAMES_WITH_TAGS:
                parameters_dict[key] = value
            elif key in ["solver", "lr__solver"]:
                if value in self.SOLVERS:
                    self.solver = value
                else:
                    raise TypeError(f"The solver {value} is not between available solvers for LogResModel.")
            else:
                raise TypeError(f"Parameter {key} is not allowed so model cannot be set.")
            
        pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)),
                ('pca', TruncatedSVD(n_components=100)),
                ('lr', LogisticRegression(solver=self.solver))
            ])
        ClassifierModel.__init__(self, parameters=parameters_dict, pipeline=pipeline)


class TreeModel(ClassifierModel):
    ALLOWED_PARAMETER_NAMES = ['max_depth', 'min_samples_split', 'max_leaf_nodes', 'ccp_alpha']
    ALLOWED_PARAMETER_NAMES_WITH_TAGS= ['tree__max_depth', 'tree__min_samples_split', 'tree__max_leaf_nodes', 'tree__ccp_alpha']
    def __init__(self, **parameters: dict[str, Any]) -> None:
        """
        :param max_tree_depth (int|None, default=50): The maximum allowed tree depth. If None there is no max_depth check for stopping the growing.
        :param min_samples_split (int>1, default=2): The minimum number of samples in a node to be able to be splitted, at least 2.
        :param max_leaf_nodes (int>1|None, default=None): The maximum number allowed of leaf nodes, at least 2. If None there is no maximum.
        :param ccp_alpha (float>=0, default=0): Complexity parameter used for Minimal Cost-Complexity Pruning. By default no pruning is performed.
        """
        self.name = "classification_tree"
        self.registered_name = "tree_model"
        self.title_name= "Classification Tree"
        self.tag = "tree"

        # Create a dictionary with the set parameters and raise TypeError if one of them is not in the allowed dictionary.
        parameters_dict  = {'tree__max_depth': 50, 'tree__min_samples_split': 2, 'tree__max_leaf_nodes': None, 'tree__ccp_alpha': 0}
        for key, value in parameters.items():
            if key in self.ALLOWED_PARAMETER_NAMES:
                parameters_dict[self.tag+'__'+key] = value
            elif key in self.ALLOWED_PARAMETER_NAMES_WITH_TAGS:
                parameters_dict[key] = value
            else:
                raise TypeError(f"Parameter {key} is not allowed so model cannot be set.")
        
        pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer(norm='l2', use_idf=True, sublinear_tf=True)),
                ('tree', DecisionTreeClassifier())
            ])
        
        ClassifierModel.__init__(self, parameters=parameters_dict, pipeline=pipeline)
        