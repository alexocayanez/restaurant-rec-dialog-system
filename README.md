# MAIR Restaurant Recommendation Dialog System
This project was done within the Methods for Artificial Intelligence Research course at Utrecht University. This project aims to build a restaurant recommender system that interacts with users conversationally. The system should understand user preferences and then use this information to recommend a suitable restaurant from its database

Through this project, we explore key AI and NLP concepts such as dialog act classification, dialog management, and preference extraction. By combining rule-based methods with machine-learning techniques, the system delivers an engaging and intelligent user experience. The report explaining the implementation of the system can be found [here](https://github.com/alexocayanez/restaurant-rec-dialog-system/blob/main/Reports/Part1_MAIR_project_report.pdf).

## Installation
1. Clone the repository with `git clone https://github.com/alexocayanez/restaurant-rec-dialog-system.git`.
2. It is recommended to use python 3.11.0 to replicate experiments
3. Optionally but encouraged, create a virtual environment. Install the necessary Python packages using `pip install -r requirements.txt`.
4. Change directory to Scripts in order to execute the dialog system or the experiments: `cd Scripts`.

Note: If you encounter an error with running the text to speech package (pyttsx3) on a macbook, this could solve it: https://github.com/nateshmbhat/pyttsx3/issues/290
If the program still does not work as expected (for example crashing after the welcome text), just turn the text to speech option off in the config file.

## Usage
The project implementation is divided into three main parts. Each part contributes to the functionality of the overall restaurant recommender system.

### Part 1A: Text Classification

#### Data statistics
For running the functions that cover the data analysis, run `data.py`. This will generate some plots and prints data statistics in the command line.
The data analysis covers the following:
- Label distribution
- Utterance length statistics
- Dialog act frequency by length
- Dialog act length statistics
- Function to calculate the number of out of vocabulary words.
  - This can be called on subsets of the data, like an x_test or x_train.

#### Machine Learning
The machine learning classifier models can be found in the file classifiers.py. In the implementation of the classifiers, we use a combination of scikit learn pipelines of estimators and mlflow models, which will help us to keep track of all the information of our estimators.

#### ML Experiments
We have three of machine learning experiments to be able to reproduce the training and hyper-parameter tuning of the models with precision, using mlflow Tracking to log all the important information of the experiments. To see all the logged information with mlflow, open a new terminal in 'Scripts' folder and run the command `mlflow ui`, which will start a localhost server hosting an ui where it is possible to see the information from all the experiments. These experiments are:

1. **Training of the model**: In this experiment, we select a ml model from logistic regression or classifier tree and also its hyperparameters, and train it according to the training data. The program will ask if we want to override saved mode (in this case, it will save the model trained to use it in the dialog system), and if we want to log its evaluation on test data. To perform a run of this experiment just run `python train.py`. 

2. **Grid search**: We can perform grid search of hyper-parameters on both models by running `python grid_tuning.py` and selecting one model and grid parameter from those printed on screen. Saving and logging of best found model are also requirements of the experiment.

3. **Random search**: We can perform random search of hyper-parameters on both models by running `python random_tuning.py`. We need to input the total number of runs, the portion of the best searches to appear in the logged plot, and if we want to perform a comparison between some hyper-parameter states.

By default, the program will run in terminal and ask for keyboard input, but this can be changed at the end of each python file. Also, at the start of all the three experiments we can choose if we want to work with deduplicated or non deduplicated data.
It is possible that when you start runing new experiments the program fails to retrieve experiments ids or paths. This is because it is not set the mlflow Project workflow and we don't save our run searches in a database. To solve this, just delete `Scripts/mlruns` folder and start exprimenting! However if you don't perform any experiment you can open mlflow UI and see our logged runs.

### Part 1B: Dialog management
This part handles the flow of the conversation and ensures the system responds appropriately to user input.

The files dialog_system.py and recommendation.py contain the implementation for Part 1b.
The code related with the restaurant database and lookup function is modelled by Recommendation class, which also provides some functions for the system utterances building.
The dialog_system.py contains the state_transition function, the keyword matching and pattern functions, and the main function that runs the system. It is possible to turn on/off debug prints with bool debug argument of main function.
To start the dialog system just run `python dialog_system.py` from the `Scripts` directory.

### Part 1C
This part enhances the system's intelligence and customizability.

The files `dialog_system.py`, `recommendation.py` and `Config/config.json` contain the implementation for Part 1c.
The configurations implemented are:
- OUTPUT IN ALL CAPS OR NOT (`all_caps`)
- Use text-to-speech for system utterances (`text_to_speech`)
- Levenshtein edit distance for preference extraction (`levenshtein_distance`)
- Allow dialog restarts or not (`allow_restarts`)

it is also optional to choose the model that is used in the config file. The options are:
1. `logisticRegression`
2. `decisionTree`

First, set the configuration in the `config.json` file. Afterwards, the program can be run.
Running the dialog system with the reasoning implemented can be done by **navigating to the `Scripts` folder**, training the model by running `train.py` and running the following command: `python dialog_system.py`

The functions that implement the reasoning part can be found in `recommendation.py`:
- `add_additional_properties()`
- `current_k_reccomendation_to_str()`
