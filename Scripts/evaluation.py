import os
from pathlib import PurePath
import tempfile
from typing import Tuple

from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn

from data import Data


class EvaluationLogger:
    def __init__(self, x_test: list[str], y_test: list[str], deduplicated: bool=False) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.deduplicated = deduplicated

    def print_evaluation(self, model: mlflow.pyfunc.PythonModel) -> None:
        y_pred = model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred))

    def get_test_number_of_true_and_false_predictions(self, model: mlflow.pyfunc.PythonModel) -> Tuple[int, int]:
        y_pred = model.predict(self.x_test)
        n_true_predictions = 0
        n_false_predictions = 0

        cm = confusion_matrix(self.y_test, y_pred)
        assert cm.shape[0] == cm.shape[1]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i == j:
                    n_true_predictions += cm[i, j]
                else:
                    n_false_predictions += cm[i, j]

        return n_true_predictions, n_false_predictions

    def get_incorrect_predictions(self, model: mlflow.pyfunc.PythonModel) -> list[Tuple[str, str, str]]:
        """
        :param model: The trained model
        :return: A list of wrongly classified utterances with their true tag and predicted tag, respectively.
        """
        y_pred = model.predict(self.x_test)
        incorrect_predictions = []

        for utterance, true_value, predicted_value in zip(self.x_test, self.y_test, y_pred):
            if true_value != predicted_value:
                incorrect_predictions.append((utterance, true_value, predicted_value))

        return incorrect_predictions

    def log_evaluation(self, model: mlflow.pyfunc.PythonModel) -> None:
        """
        Logs the evaluation of a trained model in a (already started) mlflow run.

        :param model: The trained model to be evaluated in the test set
        """
        y_pred = model.predict(self.x_test)
        mlflow.log_metric("accuracy", accuracy_score(self.y_test, y_pred))
        mlflow.log_metric("precision", precision_score(self.y_test, y_pred, average="macro", zero_division=0))
        mlflow.log_metric("recall", recall_score(self.y_test, y_pred, average="macro", zero_division=0))

        cm = confusion_matrix(self.y_test, y_pred) 
        self.log_confusion_matrix_as_png(confusion_matrix=cm, prefix_filename=model.name)

    def log_confusion_matrix_as_png(self, confusion_matrix: np.ndarray, prefix_filename: str = "") -> None:
        """
        Log the confusion matrix in a already started mlflow run as a .png artifact.

        :param confusion_matrix: np.ndarray representing the confusion matrix
        :param prefix_filename(str): Prefix to add to the name of the file, usually name model.
        """
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        seaborn.heatmap(confusion_matrix, annot=True, ax=ax, fmt='g')

        ax.set_xlabel('Predicted act', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        #plt.xticks(rotation=90)
        #ax.xaxis.set_ticklabels(ticklabels=Data.DIALOG_ACTS, fontsize=10)
        #ax.xaxis.tick_bottom()

        ax.set_ylabel('True act', fontsize=20)
        #ax.yaxis.set_ticklabels(ticklabels=Data.DIALOG_ACTS, fontsize=10)
        #plt.yticks(rotation=0)
        plt.title('Multiclass Confusion Matrix', fontsize=20)

        # Create a temporary directory, save the confusion matix there as .png and log the image in mlflow.
        temp_dir = tempfile.TemporaryDirectory().name
        os.mkdir(temp_dir)
        filename = prefix_filename + "_confusion_matrix.png"
        file_path = PurePath(temp_dir, filename)
        plt.savefig(file_path)
        mlflow.log_artifact(local_path=file_path, artifact_path="images")
