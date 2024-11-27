from datetime import datetime
import numbers
from pathlib import PurePath
import shutil

import mlflow
import mlflow.pyfunc
import mlflow.models
from mlflow.types import ColSpec, Schema

from classifiers import LogResModel, TreeModel, ClassifierModel
from data import Data
from evaluation import EvaluationLogger
from utils import create_and_set_mlflow_experiment, wrong_parameter, is_natural_number, is_non_negative_float


def train_model(model: ClassifierModel,
                x_train: list[str],
                y_train: list[str],
                deduplicated: bool,
                override_saved_model: bool = False,
                evaluator: EvaluationLogger | None = None
                ) -> None:
    """
    Function that fits a given model to training data according to specified hyperparameters,
    logs the model and given hyperparameters to mlflow experiment
    and logs the evaluation and/or creates a new saved model if requested.

    :param model: One of the models in classifiers.py. It also provides model.parameters to log.
    :param x_train: List of utterances in the training set.
    :param y_train: List of class tags in the training set.
    :param override_saved_model (bool): If true override saved model in ../Models folder
    :param evaluator: An instance of EvaluationLogger with the test data, if we want evaluation to be logged.
    """

    if model.trained:
        raise Exception("Model already trained. Please provide an Classifier object before applying .fit method.")
    else:
        # Retrieve experiment id according to model name and start a run in it.
        deduplicated_tag = "deduplicated_" if deduplicated else ""
        experiment_id = create_and_set_mlflow_experiment(experiment_name=deduplicated_tag+model.name + "_training")
        TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        with mlflow.start_run(run_name=TIMESTAMP + "_" + deduplicated_tag + "_training", experiment_id=experiment_id) as run:
            mlflow.log_param("data_dedupliction", deduplicated)

            # We log the parameters given in the dictionary.
            for key, value in model.parameters.items():
                mlflow.log_param(key, value)

            # Model is fitted to training data according to given parameters.
            model.fit(x_train=x_train, y_train=y_train)
           
            schema = Schema([ColSpec(type="string")])           
            mlflow.pyfunc.log_model(artifact_path="python_model",
                                    python_model=model, 
                                    registered_model_name= deduplicated_tag + model.registered_name,
                                    code_path=["./train.py"], 
                                    input_example= ["i want some italian food", "north part of the town"],
                                    signature=False,
                                    pip_requirements= "../requirements.txt"
                                    )

            # Override saved model is requested
            if override_saved_model:
                model_path = PurePath("../Models", deduplicated_tag + model.registered_name)
                shutil.rmtree(model_path)
                mlflow.pyfunc.save_model(python_model= model,
                                         path=model_path,
                                         mlflow_model=mlflow.models.Model(),
                                         signature=False,
                                         input_example= ["i want some italian food", "north part of the town"],
                                         pip_requirements="../requirements.txt"
                                         )

            # Logging of evaluation done if requested.
            if evaluator is not None:
                evaluator.log_evaluation(model=model)


def get_model_by_keyboard_input() -> ClassifierModel:
    """
    Function that get by keyboard user input specifying the name of a model and its parameters.

    :return model: A ClassifierModel object with its DEFAULT_PARAMETERS dictionary according to given parameters.
                   It still does not have the parameters set in the model pipeline. 
    """
    selected_model = ''
    while wrong_parameter(selected_model, ['log_res', 'tree']):
        selected_model = input("Select classifier model between 'log_res' (Logistic Regression) or 'tree' (Decision Tree): ").lower()

    if selected_model == 'log_res':
        selected_C = ''
        while not is_non_negative_float(selected_C):
            selected_C = input("Enter C, the inverse of regularization strenght (positive float): ")
        selected_penalty = ""
        while wrong_parameter(selected_penalty, ["l1", "l2", "elasticnet", "none"]):
            selected_penalty = input("PSelect regularization penalty between 'l1' , 'l2', 'elasticnet' or None : ").lower()
        model = LogResModel(C = float(selected_C), 
                            penalty = None if selected_penalty=="none" else selected_penalty)
    
    if selected_model == 'tree':
        selected_max_depth = ''
        while( not is_natural_number(selected_max_depth)) and (selected_max_depth!="none"):
            selected_max_depth = input("Choose the maximum depth of the tree (int or none): ").lower()
        selected_min_samples_split = ''
        while not is_natural_number(selected_min_samples_split, greater_than=1):
            selected_min_samples_split = input("Choose the minimum samples in a node to be able to split it (int >= 2): ")
        selected_max_leaf_nodes = ''
        while (not is_natural_number(selected_max_leaf_nodes, greater_than=1)) and (selected_max_leaf_nodes!="none"):
            selected_max_leaf_nodes = input("Choose the maximum leaf nodes allowed (int or none): ").lower()
        selected_ccp_alpha = ''
        while not is_non_negative_float(selected_ccp_alpha):
            selected_ccp_alpha = input("Enter C, the inverse of regularization strenght (non negative float): ")
        model = TreeModel(max_depth = None if selected_max_depth=="none" else int(selected_max_depth),
                          min_samples_split = int(selected_min_samples_split),
                          max_leaf_nodes = None if selected_max_leaf_nodes=="none" else int(selected_max_leaf_nodes),
                          ccp_alpha = float(selected_ccp_alpha)
                          )
    return model

def main(keyboard_input: bool=True):
    selected_data = ''
    while wrong_parameter(selected_data, ['1', '2']):
            selected_data = input("What data do you want to use? Introduce one of the following (numbers):\n 1) Non deduplicated data\n 2) Deduplicated data\n").lower()
    deduplicated_bool = True if int(selected_data) == 2 else False

    PATH_DIALOG_ACTS = """../Data/dialog_acts.dat"""
    if deduplicated_bool:
        data = Data(PATH_DIALOG_ACTS, deduplicated=True)
    else:
        data = Data(PATH_DIALOG_ACTS, deduplicated=False)
    x_train, x_test, y_train, y_test = data.split()
    logger = EvaluationLogger(x_test, y_test, deduplicated=deduplicated_bool)

    if keyboard_input:
        model = get_model_by_keyboard_input()

    else:  #IF NO KEYBOARD INPUT HERE IS THE PLACE TO CHANGE THE PARAMETERS
        lr = LogResModel(C=200, penalty="l2")
        TREE_PARAMETERS = {'max_depth': 200, 'min_samples_split': 2, 'max_leaf_nodes': None, 'ccp_alpha': 0}
        tree = TreeModel(**TREE_PARAMETERS)
        model = tree 
    
    selected_override = ''
    while wrong_parameter(selected_override, ['y', 'n']):
        selected_override = input("Do you want to override best model? (Y/n) ").lower()
    selected_override_saved_model = True if selected_override == 'y' else False

    selected_evaluation = ''
    while wrong_parameter(selected_evaluation, ['y', 'n']):
        selected_evaluation = input("Do you want to log the evaluation of best model? (Y/n) ").lower()
    selected_evaluator = logger if selected_evaluation == "y" else None

    train_model(model=model, x_train=x_train, y_train=y_train, 
                deduplicated=deduplicated_bool, 
                override_saved_model=selected_override_saved_model, 
                evaluator=selected_evaluator)

if __name__ == "__main__":
    main()
