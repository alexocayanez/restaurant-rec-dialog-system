import mlflow.pyfunc

from evaluation import EvaluationLogger
from data import Data


def main():
    PATH_DIALOG_ACTS = """../Data/dialog_acts.dat"""
    data = Data(PATH_DIALOG_ACTS)
    x_train, x_test, y_train, y_test = data.split()
    logger = EvaluationLogger(x_test, y_test)

    log_res = mlflow.pyfunc.load_model(model_uri="../Models/log_res_model").unwrap_python_model()
    tree = mlflow.pyfunc.load_model(model_uri="../Models/tree_model").unwrap_python_model()

    # print("LogRes predictions: " + str(log_res.predict(x_test)[:5]))
    # print("Tree predictions: " + str(tree.predict(x_test)[:5]))

    total_predictions = len(y_test)

    for model in [log_res, tree]:
        print(f"{model.name.upper()} MODEL:")

        # Overall analysis
        true_count, false_count = logger.get_test_number_of_true_and_false_predictions(model)
        true_perc, false_perc = round(true_count / total_predictions * 100, 1), \
            round(false_count / total_predictions * 100, 1)
        print(f"Correct predictions: {true_count}/{total_predictions} ({true_perc}%). "
              f"Incorrect predictions: {false_count}/{total_predictions} ({false_perc}%).")

        # Utterance analysis
        print('Wrongly classified utterances are printed below:')
        incorrect_predictions = logger.get_incorrect_predictions(model)
        for utterance, true_tag, predicted_tag in incorrect_predictions:
            print(f'  - "{utterance}" was classified as \'{predicted_tag}\', but it was \'{true_tag}\'.')
        print("\n")


if __name__ == "__main__":
    main()
