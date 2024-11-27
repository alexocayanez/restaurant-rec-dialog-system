from datetime import datetime
from pathlib import PurePath
import shutil
from typing import Any

import mlflow
from mlflow.models import Model
import mlflow.pyfunc
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold   

from data import Data
from classifiers import LogResModel, TreeModel
from evaluation import EvaluationLogger
from utils import create_and_set_mlflow_experiment, wrong_parameter, is_natural_number

def log_grid_search_nested_run(grid: GridSearchCV,
                               run_index: int, 
                               parameters: dict[str, Any],
                               number_splits: int
                               ) -> None:
    """
    Function that starts a mlflow run in which will be logged the parameters and metrics
    corresponding to one parameter combination within a GridSearchCV.

    :param grid: parent GridSearchCV
    :param run_index (int): The index of the combination in the search.
    :param parameters: Dictionary of the selected combination of model hyper-parameters.
    :param number_splits (int): Number of cross-validation splits, usually denoted as k.
    """
    results = grid.cv_results_
    with mlflow.start_run(run_name=str(run_index), nested=True):
        
        # Log the parameters selected in the run
        for key, value in parameters.items():
            mlflow.log_param(key, value)
        
        # Log the cross-validated test scores (for train scores: GridSearchCV(..., return_train_score=True))
        for i in range(number_splits):
            split = f"split{i}_test_score"
            mlflow.log_metric(split, grid.cv_results_[split][run_index])
        mlflow.log_metric("mean_test_score", results["mean_test_score"][run_index])
        mlflow.log_metric("std_test_score", results["std_test_score"][run_index])
                 
def grid_search_experiment(model: mlflow.pyfunc.PythonModel, 
                           x_train: list[str], 
                           y_train: list[str],
                           parameters_grid:  dict[str, Any]|None,
                           deduplicated: bool,
                           number_cv_splits: int=3,
                           override_saved_model: bool=False,
                           log_nested_runs: bool=False,
                           evaluator: EvaluationLogger|None=None
                           ) -> None:
    """
    Performs a grid_search_cv mlflow experiment for the given model and training data. 
    It logs the paremeters and cross-validation metrics for every combination allowed in input parameters grid if requested.
    It log the best found parameters and it can save the best model obtained and log its evaluation if required.

    :param model: One of the models in classifiers.py
    :param x_train: List of utterances in the training set.
    :param y_train: List of class tags in the training set. 
    :param parameters_grid: Dictionary specifying the search grid of model hyper-parameters.
    :param deduplicated (bool): Indicates if we are using deduplicated data (True) or not (False)
    :param number_splits (int): Number of splits (k) in (k-fold) cross validation.
    :param override_saved_model (bool): If true override saved model in ../Models folder
    :param log_nested_runs (bool): If true starts a new run for every combination of parameters, and log their information.
    :param evaluator: An instance of EvaluationLogger with the test data, if we want evaluation to be logged for highest score model.
    """

    if model.trained:
        raise Exception("Model already trained. Please provide an Classifier object before applying .fit method.")
    else:
        deduplicated_tag = "deduplicated_" if deduplicated else ""
        experiment_id = create_and_set_mlflow_experiment(experiment_name=deduplicated_tag+model.name+"_grid_search_cv")
        TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        with mlflow.start_run(run_name=TIMESTAMP +"_"+ deduplicated_tag +"_grid_search_cv", experiment_id=experiment_id) as run:
            mlflow.log_param("data_dedupliction", deduplicated)
            grid_search = GridSearchCV(estimator=model.pipeline,
                                       param_grid=parameters_grid,
                                       n_jobs=2,
                                       cv= StratifiedKFold(n_splits=number_cv_splits, shuffle=True)
                                       )
            grid_search.fit(x_train, y_train)

            # Log the parameters and cv-scores of nested runs
            if log_nested_runs:
                for i, dic in enumerate(grid_search.cv_results_['params']):
                    log_grid_search_nested_run(grid=grid_search, 
                                            run_index=i,
                                            parameters=dic,
                                            number_splits=number_cv_splits
                                            )

            # Log the best parameters and log a python model trained with them
            mlflow.log_params(grid_search.best_params_)
            if isinstance(model, LogResModel):
                best_model = LogResModel(**grid_search.best_params_)
            elif isinstance(model, TreeModel):
                best_model = TreeModel(**grid_search.best_params_)
            best_model.fit(x_train, y_train)
            mlflow.pyfunc.log_model(artifact_path="python_model",
                                    python_model=best_model, 
                                    registered_model_name=deduplicated_tag+model.registered_name+"_tuned",
                                    code_path=["./grid_tuning.py"], 
                                    input_example= ["i want some italian food", "north part of the town"],
                                    signature=False,
                                    pip_requirements= "../requirements.txt"
                                    )
            
            mlflow.sklearn.log_model(grid_search.best_estimator_,
                                     artifact_path="sk_model",
                                     registered_model_name=deduplicated_tag+"sk_"+model.registered_name+"_tuned",
                                     signature= False,
                                     input_example= ["i want some italian food", "north part of the town"],
                                     )
            
            if override_saved_model:
                model_path = PurePath("../Models", deduplicated_tag + best_model.registered_name)
                shutil.rmtree(model_path)
                mlflow.pyfunc.save_model(python_model= best_model,
                                         path=model_path,
                                         mlflow_model=Model(),
                                         signature=False,
                                         input_example= ["i want some italian food", "north part of the town"],
                                         pip_requirements="../requirements.txt"
                                         )
                
            if evaluator is not None:
                evaluator.log_evaluation(model=best_model)

LR_PARAM_GRID_LIST = [
                            [ "Compare solvers for no regularization",
                                {   
                                    "lr__penalty": [None] ,
                                    "lr__C": [float('inf')],
                                    "lr__solver": ["lbfgs", "newton-cg", "sag", "saga"]
                                }
                            ],
                            [ "Only l1 penalty",
                                {
                                    "lr__C": np.logspace(-5, 2, num=30, base=10),  
                                    "lr__penalty": ["l1"],
                                    "lr__solver": ["saga"]  # The unique solver for "l1" penalty in multi-class
                                },
                            ],
                            [ "Compare solvers for l2 penalty",
                                {   
                                    "lr__C": np.logspace(-5, 2, num=30, base=10),  
                                    "lr__penalty": ["l2"],
                                    "lr__solver": ["lbfgs", "newton-cg", "sag", "saga"] 
                                },
                            ],
                            [ "Full grid with regularization",
                                {   
                                    "lr__C": np.logspace(-5, 2, num=30, base=10),  
                                    "lr__penalty": ["l1", "l2"],
                                    "lr__solver": ["saga"] 
                                },
                            ],
                        ]

TREE_PARAM_GRID_LIST= [   
                        [   "Full grid", 
                            {  
                                "tree__max_depth": [None, 70, 80, 100],
                                "tree__min_samples_split": [2, 5, 9],
                                "tree__max_leaf_nodes": [None, 90, 100],
                                "tree__ccp_alpha": np.logspace(-5, 0, num=10, base=10),
                            },
                        ],
                        [   "Default min samples and no max leaf nodes", 
                            {    
                                "tree__max_depth": [None, 70, 80, 100],
                                "tree__min_samples_split": [2],
                                'tree__max_leaf_nodes': [None],
                                "tree__ccp_alpha": np.logspace(-5, 0, num=10, base=10)
                            },
                        ],
                        [  "Extensive search over cost complexity pruning alpha", 
                            {   
                                "tree__max_depth": [None],
                                "tree__min_samples_split": [2],
                                'tree__max_leaf_nodes': [None],
                                "tree__ccp_alpha": np.logspace(-5, 0, num=100, base=10)
                            },
                        ],
                        [   "Search over minimum samples per split and cost complexity pruning alpha", 
                            {  
                                "tree__max_depth": [None],
                                "tree__min_samples_split": range(2, 10),
                                "tree__max_leaf_nodes": [None],
                                "tree__ccp_alpha": np.logspace(-5, 0, num=10, base=10),
                            },
                        ],
                     ]

def perform_grid_search_by_keyboard_input(x_train: list[str], y_train: list[str], deduplicated: bool, evaluator: EvaluationLogger|None=None) -> None:
    """
    Function that get by keyboard user input specifying the parameters of a grid search and performs it.

    :param x_train: List of utterances in the training set.
    :param y_train: List of class tags in the training set.
    :param deduplicated: Boolean representing if we are using deduplicated data (True) or not (False)
    :para evaluator: An instance of EvaluationLogger with the test data.
    """

    selected_model = ''
    while wrong_parameter(selected_model, ['log_res', 'tree']):
        selected_model = input("Select classifier model between 'log_res' (Logistic Regression) or 'tree' (Decision Tree): ").lower()
    model = LogResModel() if selected_model == "log_res" else TreeModel()

    selected_grid_list = LR_PARAM_GRID_LIST if selected_model == "log_res" else TREE_PARAM_GRID_LIST
    selected_grid_index = ''
    posible_indexes = [str(i+1) for i in range(len(selected_grid_list))]
    while wrong_parameter(selected_grid_index, posible_indexes):
        print("Select the grid between the following: ")
        for i, l in enumerate(selected_grid_list):
            print(f"{i+1}) {l[0]}" )
        selected_grid_index = input()
    selected_grid_index = int(selected_grid_index)-1

    selected_override = ''
    while wrong_parameter(selected_override, ['y', 'n']):
        selected_override = input("Do you want to override best model? (Y/n) ").lower()
    selected_override_saved_model = True if selected_override == 'y' else False

    selected_evaluation = ''
    while wrong_parameter(selected_evaluation, ['y', 'n']):
        selected_evaluation = input("Do you want to log the evaluation of best model? (Y/n) ").lower()
    selected_evaluator = evaluator if selected_evaluation == "y" else None

    grid_search_experiment(model=model, x_train=x_train, y_train=y_train, 
                           parameters_grid=selected_grid_list[selected_grid_index][1], 
                           deduplicated=deduplicated,
                           evaluator=selected_evaluator, 
                           override_saved_model=selected_override_saved_model, 
                           log_nested_runs=False)

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
        perform_grid_search_by_keyboard_input(x_train=x_train, y_train=y_train, deduplicated=deduplicated_bool, evaluator = logger)

    else: #MANUAL INPUT OF PARAMETERS
        grid_search_experiment(model=TreeModel() , x_train=x_train, y_train=y_train, 
                            parameters_grid=TREE_PARAM_GRID_LIST[1][1], evaluator=logger, 
                            override_saved_model=False, log_nested_runs=False)

if __name__=="__main__":
    main()