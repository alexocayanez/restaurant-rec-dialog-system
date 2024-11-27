from datetime import datetime
import math
import os
from pathlib import PurePath
import shutil
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import Model
import mlflow.pyfunc
import numpy as np
import pandas as pd
import seaborn
import scipy.stats 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

from data import Data
from classifiers import LogResModel, TreeModel
from evaluation import EvaluationLogger
from utils import create_and_set_mlflow_experiment, wrong_parameter, is_natural_number, is_ratio


def random_search_experiment(model: mlflow.pyfunc.PythonModel,
                             x_train: list[str], 
                             y_train: list[str],
                             parameters_distributions:  dict[str, Any]|None,
                             number_searches: int=40,
                             ratio_to_plot: float=0.8,
                             compare: str|None=None,
                             significance: float=0.01,
                             number_cv_splits: int=3,
                             evaluator: EvaluationLogger|None=None
                             ) -> None:
    if model.trained:
        raise Exception("Model already trained. Please provide an Classifier object before applying .fit method.")
    elif compare not in {"lr__regularization", "tree__pruning", None}:
        raise TypeError("Comparison not admitted.")
    else:
        experiment_id = create_and_set_mlflow_experiment(experiment_name=model.name+"_random_search_cv")
        TIMESTAMP = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        with mlflow.start_run(run_name=TIMESTAMP + "_random_search_cv", experiment_id=experiment_id) as run:
            cv_splitter = StratifiedKFold(n_splits=number_cv_splits, shuffle=True)
            temp_dir = tempfile.TemporaryDirectory().name
            os.mkdir(temp_dir) 

            
            if compare is not None:
                comparison_file_path = PurePath(temp_dir, "comparison_results.txt")
                comparison_file = open(comparison_file_path, "a")
                mlflow.log_param("statistical_test_significance", significance)
                comparison_file.write(f"Statistical significance: {significance} \n")
                comparison_file.write(f"We will test statistical difference between mean accuracy for: \n")
                line_group_1 = f"1) Group of models with "+compare+" == False \n"
                line_group_2 = f"1) Group of models with "+compare+" == True \n"
                comparison_file.writelines([line_group_1, line_group_2])
                # If we compare we make sure that: 1) We have more than one parameter grid, 2) We have even number of searches
                assert isinstance(parameters_distributions, list) 
                assert isinstance(compare, str)
                half_number_searches = math.ceil(number_searches/2)
                number_searches = 2*half_number_searches 
                
                # Create two different cv searches with half of the runs in each to compare
                if compare == "lr__regularization":
                    df_1 = pd.DataFrame()
                    for n in range(half_number_searches):
                        n_search = GridSearchCV(estimator=model.pipeline, param_grid=parameters_distributions[0] , n_jobs=1, cv=cv_splitter)
                        n_search.fit(x_train, y_train)
                        df_1 = pd.concat([df_1, pd.DataFrame(n_search.cv_results_)])
                else:
                    search_1 = RandomizedSearchCV(estimator=model.pipeline, param_distributions=parameters_distributions[0] , 
                                                    n_jobs=1, n_iter=half_number_searches, cv=cv_splitter)
                    search_1.fit(x_train, y_train)
                    df_1 = pd.DataFrame(search_1.cv_results_)
                df_1["param_"+compare] = False
                mean_test_scores_1 = df_1["mean_test_score"].to_numpy()

                search_2 = RandomizedSearchCV(estimator=model.pipeline, param_distributions=parameters_distributions[1], 
                                                n_jobs=1, n_iter=half_number_searches, cv=cv_splitter)
                search_2.fit(x_train, y_train)
                df_2 = pd.DataFrame(search_2.cv_results_)
                df_2["param_"+compare] = True
                mean_test_scores_2 = df_2["mean_test_score"].to_numpy()

                df = pd.concat([df_1, df_2])

            else:
                random_search = RandomizedSearchCV(estimator=model.pipeline,param_distributions=parameters_distributions,
                                                n_jobs=1, n_iter=number_searches, cv=cv_splitter)
                random_search.fit(x_train, y_train)
                df = pd.DataFrame(random_search.cv_results_)

            # Log the best parameters and log a python model trained with them
            df = df.sort_values(by='rank_test_score')
            best_params = df['params'].iloc[0]
            mlflow.log_params(best_params)
            if isinstance(model, LogResModel):
                best_model = LogResModel(**best_params)
            elif isinstance(model, TreeModel):
                best_model = TreeModel(**best_params)
            best_model.fit(x_train, y_train)
            mlflow.pyfunc.log_model(artifact_path="python_model",
                                    python_model=best_model, 
                                    registered_model_name=model.registered_name+"_best_random_tuned_model",
                                    code_path=["./random_tuning.py"], 
                                    input_example= ["i want some italian food", "north part of the town"],
                                    signature=False,
                                    pip_requirements= "../requirements.txt"
                                    )

            # Save the entire search as .csv 
            csv_file_name=f"random_search_{model.registered_name}_all_searches.csv"
            csv_file_path = PurePath(temp_dir, csv_file_name)
            df.to_csv(csv_file_path)
            mlflow.log_artifact(local_path=csv_file_path, artifact_path="data")

            # Log search parameters
            number_plotted_searches = math.ceil(number_searches * ratio_to_plot)
            mlflow.log_param("number_searches", number_searches)
            mlflow.log_param("number_plotted_searches", number_plotted_searches)
            
            # Create and log the general plot of the search
            plot_file_name=f"random_search_{model.registered_name}_cv_accuracy_by_parameter.png"
            plot_file_path = PurePath(temp_dir, plot_file_name)
            plot_df = df.head(number_plotted_searches)
            n_plots = len(model.ALLOWED_PARAMETER_NAMES)
            n_columns = 2  
            n_rows = math.ceil(n_plots/n_columns)
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, figsize=(14, n_rows*6))
            fig.suptitle(f"{model.title_name} best {len(plot_df)} parameter settings in {number_searches} random searches")

            if isinstance(model, LogResModel):
                if not compare:
                    assert isinstance(parameters_distributions, dict)
                    if None in parameters_distributions['lr__penalty']:
                        plt.close(fig)
                    else:
                        seaborn.boxplot(ax=axes[0], data=plot_df, x="param_lr__penalty", y="mean_test_score")
                        seaborn.scatterplot(ax=axes[1], data=plot_df, x="param_lr__C", y="mean_test_score", hue="param_lr__penalty", palette="rainbow")
                        plt.savefig(plot_file_path)
                        mlflow.log_artifact(local_path=plot_file_path, artifact_path="images")

            if isinstance(model, TreeModel):
                tree_hue = "param_tree__pruning" if compare == "tree__pruning" else None
                seaborn.scatterplot(ax=axes[0,0], data=plot_df, x="param_tree__max_depth", y="mean_test_score", hue=tree_hue)
                seaborn.scatterplot(ax=axes[0,1], data=plot_df, x="param_tree__min_samples_split", y="mean_test_score", hue=tree_hue)
                seaborn.scatterplot(ax=axes[1,0], data=plot_df, x="param_tree__max_leaf_nodes", y="mean_test_score", hue=tree_hue)
                if compare == "tree__pruning":
                    seaborn.boxplot(ax=axes[1,1], data=plot_df, x="param_tree__pruning", y="mean_test_score")
                else:
                    seaborn.scatterplot(ax=axes[1,1], data=plot_df, x="param_tree__ccp_alpha", y="mean_test_score")
                
                plt.savefig(plot_file_path)
                mlflow.log_artifact(local_path=plot_file_path, artifact_path="images")


            if compare is not None:
                # Comparison plot: Log accuracy boxplots for both classes (compare = True, compare = False)
                compare_plot_file_name=f"{compare}_comparison_by_random_search.png"
                compare_plot_file_path = PurePath(temp_dir, compare_plot_file_name)
                fig = plt.figure(figsize=(7,6))
                fig.suptitle(f"{compare} accuracy comparison with {number_searches} random searches")
                seaborn.boxplot(data=df, x="param_"+compare, y="mean_test_score")
                plt.savefig(compare_plot_file_path)
                mlflow.log_artifact(local_path=compare_plot_file_path, artifact_path="images")

                #Performing statistical tests to compare both   
                mlflow.log_param("comparison_parameter", compare)
                ks_test_1 = scipy.stats.kstest(mean_test_scores_1, "norm")
                mlflow.log_metrics({
                        "statistic_ks_normality_test_without_comparison_parameter_applied": ks_test_1[0],
                        "pvalue_ks_normality_test_without_comparison_parameter_applied": ks_test_1[1]
                })
                if ks_test_1[1] < significance:
                    comparison_file.write("We have significant evidence that mean acuracy for Group 1 does not follow a normal distribution. (Kolmogorov-Smirnov test)\n")
                else:
                    comparison_file.write("We can accept the null hypothesis mean acuracy for that Group 1 follow a normal distribution. (Kolmogorov-Smirnov test)\n")

                ks_test_2 = scipy.stats.kstest(mean_test_scores_2, "norm")
                mlflow.log_metrics({
                        "statistic_ks_normality_test_with_comparison_parameter_applied": ks_test_2[0],
                        "pvalue_ks_normality_test_with_comparison_parameter_applied": ks_test_2[1]
                })
                if ks_test_2[1] < significance:
                    comparison_file.write("We have significant evidence that mean acuracy for Group 2 does not follow a normal distribution. (Kolmogorov-Smirnov test)\n")
                else:
                    comparison_file.write("We can accept the null hypothesis that mean acuracy for Group 2 follow a normal distribution. (Kolmogorov-Smirnov test)\n")

                ks_test_between = scipy.stats.ks_2samp(mean_test_scores_1, mean_test_scores_2)
                mlflow.log_metrics({
                        "statistic_ks_distribution_test_between_settings": ks_test_between[0],
                        "pvalue_ks_distribution_test_between_settings": ks_test_between[1]
                })
                if ks_test_between[1] < significance:
                    comparison_file.write("We have significant evidence that Groups 1 and 2 do not follow the same distribution. (Kolmogorov-Smirnov test)\n\n")
                else:
                    comparison_file.write("We can accept the null hypothesis that mean acuracy for Groups 1 and 2 follow the same distribution. (Kolmogorov-Smirnov test)\n\n")

                comparison_file.write("Let's test now the statistical differences between mean accuracies of groups.\n")
                if np.mean(mean_test_scores_1) < np.mean(mean_test_scores_2):
                    alternative = "less"
                    comparison_file.write("Our alteranative hypothesis will be that the mean accuracy for group 1 is lower than mean accuracy for group 2.\n")
                elif np.mean(mean_test_scores_1) > np.mean(mean_test_scores_2):
                    alternative = "greater"
                    comparison_file.write("Our alteranative hypothesis will be that the mean accuracy for group 1 is greater than mean accuracy for group 2.\n")
                else:
                    alternative = "two-sided"
                    comparison_file.write("Our alteranative hypothesis will be that the mean accuracy for group 1 is different than mean accuracy for group 2.\n")
                alternative_tag = "different" if alternative == "two-sided" else alternative   

                if ks_test_1[1] > significance and ks_test_2[1] > significance and ks_test_between[1] > significance:
                    comparison_file.write("We will perform a T Test as the samples seem to be normal.\n")
                    mlflow.log_param("alternative_comparison_test", alternative)
                    t_test = scipy.stats.ttest_rel(mean_test_scores_1, mean_test_scores_2, alternative=alternative)
                    mlflow.log_metrics({
                            "statistic_t_test_between_mean_score_settings": t_test[0],
                            "pvalue_t_test_between_mean_score_settings": t_test[1]
                    })
                    comparison_test = t_test
                else:
                    comparison_file.write("We will perform a Wilcoxon Test as samples don't follow normal distribution.\n")
                    wilcoxon_test = scipy.stats.wilcoxon(mean_test_scores_1, mean_test_scores_2, alternative=alternative)
                    mlflow.log_metrics({
                            "statistic_wilcoxon_between_mean_score_settings": wilcoxon_test[0],
                            "pvalue_wilcoxon_between_mean_score_settings": wilcoxon_test[1]
                    })
                    comparison_test = wilcoxon_test
                
                if comparison_test[1] < significance:
                        comparison_file.write(f"We have significant evidence that mean accuracy for Group 1 is {alternative_tag} than mean accuracy for Group 2.\n")
                else:
                    comparison_file.write("We can accept the null hypothesis that mean accuracies for both groups are equal.\n")
                comparison_file.close()
                mlflow.log_artifact(local_path=comparison_file_path, artifact_path="data")


LR_PARAM_GRID_SELECTION = [
                            {   # No regularization
                                "lr__penalty": [None] ,
                                "lr__C": [float('inf')],
                                "lr__solver": ["sag"]
                            },
                            {   #Full grid with regularization
                                "lr__C": scipy.stats.uniform(1e-5, 200),  
                                "lr__penalty": ["l1", "l2"],
                                "lr__solver": ["saga"]
                            },
                            [   # Comparison between both                
                                {
                                    "lr__penalty": [None],
                                    "lr__C": [float('inf')],
                                    "lr__solver": ["sag"]
                                },
                                {
                                    "lr__C": scipy.stats.uniform(1e-5, 200),
                                    "lr__penalty": ["l2"],
                                    "lr__solver": ["sag"]
                                },
                            ]
                        ]

TREE_PARAM_GRID_SELECTION = [
                                {   
                                    "tree__max_depth": range(5, 101),
                                    "tree__min_samples_split": range(2, 11),
                                    'tree__max_leaf_nodes': range(2, 101),
                                    "tree__ccp_alpha": [0]
                                },
                                {   #Full distributions grid 
                                    "tree__max_depth": range(5, 101),
                                    "tree__min_samples_split": range(2, 11),
                                    "tree__max_leaf_nodes": range(2, 101),
                                    "tree__ccp_alpha": scipy.stats.loguniform(1e-6, 1e1)
                                },
                                [
                                    {
                                        "tree__max_depth": range(5, 101),
                                        "tree__min_samples_split": range(2, 11),
                                        'tree__max_leaf_nodes': range(2, 101),
                                        "tree__ccp_alpha": scipy.stats.loguniform(1e-6, 1e1)
                                    },
                                    {
                                        "tree__max_depth": range(5, 101),
                                        "tree__min_samples_split": range(2, 11),
                                        'tree__max_leaf_nodes': range(2, 101),
                                        "tree__ccp_alpha": [0]
                                    }
                                ]
                            ]


def perform_random_search_by_keyboard_input(x_train: list[str], y_train: list[str]) -> None:
    """
    Function that get by keyboard user input specifying the parameters of a random random search and performs it.

    :param x_train:
    :param y_train:
    """
    selected_model = ''
    while wrong_parameter(selected_model, ['log_res', 'tree']):
        selected_model = input("Select classifier model between 'log_res' (Logistic Regression) or 'tree' (Decision Tree): ").lower()
    
    selected_number_searches = ''
    while not is_natural_number(selected_number_searches):
        selected_number_searches = input("Introduce the total number of random searches to perform: ")
    
    selected_ratio_to_plot = ''
    while not is_ratio(selected_ratio_to_plot):
        selected_ratio_to_plot = input("Choose the ratio (between 0 and 1) of best searches to be plotted: ")


    if selected_model == "log_res":
        selected_comparison = ''
        while wrong_parameter(selected_comparison, ['1', '2', '3']):
            selected_comparison = input("What do you want to do with Logistic Regression regularization? Introduce one of the following (numbers):\n 1) No regularization\n 2) Apply regularization with random parameters.\n 3) Compare between applying regularization or not\n").lower()
        selected_grid_index = int(selected_comparison)-1
        selected_compare = "lr__regularization" if selected_grid_index == 2 else None

        if selected_grid_index == 0:
            solver = "sag"
        elif selected_grid_index == 1:
            solver = "saga"
        else:
            solver = "lbfgs"
            selected_significance=''
            while not is_ratio(selected_significance):
                selected_significance = input("Choose the significance for the comparison statistical tests: ")

        random_search_experiment(model=LogResModel(solver=solver), 
                                 x_train=x_train,
                                 y_train=y_train, 
                                 number_searches=int(selected_number_searches), 
                                 ratio_to_plot=float(selected_ratio_to_plot), 
                                 compare=selected_compare,
                                 significance=float(selected_significance),
                                 parameters_distributions=LR_PARAM_GRID_SELECTION[selected_grid_index])

    elif selected_model == "tree":
        selected_comparison = ''
        while wrong_parameter(selected_comparison, ['1', '2', '3']):
            selected_comparison = input("What do you want to do with Decision Tree cost complexity pruning? Introduce one of the following (numbers):\n 1) No pruning.\n 2) Apply ccp with random parameters.\n 3) Compare between applying ccp or not.\n").lower()
        selected_grid_index = int(selected_comparison)-1
        selected_compare = "tree__pruning" if selected_grid_index == 2 else None
        if selected_compare:
            selected_significance=''
            while not is_ratio(selected_significance):
                selected_significance = input("Choose the significance for the comparison statistical tests: ")

        random_search_experiment(model=TreeModel(), 
                                 x_train=x_train,
                                 y_train=y_train, 
                                 number_searches=int(selected_number_searches), 
                                 ratio_to_plot=float(selected_ratio_to_plot), 
                                 compare=selected_compare,
                                 significance=float(selected_significance),
                                 parameters_distributions=TREE_PARAM_GRID_SELECTION[selected_grid_index])
    
def main(keyboard_input: bool=True):
    PATH_DIALOG_ACTS = """../Data/dialog_acts.dat"""
    data = Data(PATH_DIALOG_ACTS)
    x_train, x_test, y_train, y_test = data.split()
    lr = LogResModel()
    tree = TreeModel()
    logger = EvaluationLogger(x_test, y_test)

    if keyboard_input:
        perform_random_search_by_keyboard_input(x_train=x_train, y_train=y_train)

    else: # MANUAL INPUT OF PARAMETERS
        random_search_experiment(model=tree, x_train=x_train, y_train=y_train, 
                                number_searches=40, ratio_to_plot=0.2, compare="tree__pruning",
                                parameters_distributions=TREE_PARAM_GRID_SELECTION[1])

if __name__ == "__main__":
    main()