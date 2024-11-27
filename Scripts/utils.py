import os
from pathlib import PurePath
import tempfile
from typing import Any

import pandas as pd
import numpy as np
import nltk
import mlflow

def add_restaurant_information(database_path: str) -> None:
    """
    Takes the restaurant info as input and generates a csv with new properties (food quality, crowdedness, length of stay)

    Args:
        database_path (str): path to the current restaurant info csv
    """
    new_database_path = "../Data/restaurant_info_new.csv"
    df = pd.read_csv(database_path)
    food_quality_list = ['excellent', 'good', 'average', 'poor']
    crowdedness_list = ['empty', 'quiet', 'moderate', 'busy']
    length_of_stay_list = ['short', 'average', 'long']
    df['food quality'] = np.random.choice(food_quality_list, size=len(df))
    df['crowdedness'] = np.random.choice(crowdedness_list, size=len(df))
    df['length of stay'] = np.random.choice(length_of_stay_list, size = len(df))
    df.to_csv(new_database_path)

def preprocess_utterance(utterance: str, min_word_length: int, stopwords: list, stemming: bool = True) -> list[str]:
    """
    Removes inputted stopwords and really short unuseful words (specified by min_word_length argument).
    If activated, the stemming simplifies word forms to one form. For example 'go' 'going' 'goes' result in the same stemmed
    word. It uses nltk.stem.StemmerI model.
    It returns a list of the splitted selected words from the utterance.
    """
    utterance = utterance.lower()
    words = nltk.tokenize.word_tokenize(utterance)
    words = [w for w in words if len(w) > min_word_length]
    words = [w for w in words if w not in stopwords]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        words = [stemmer.stem(w) for w in words]
    return words

def preprocess_utterances(utterances: list[str], min_word_length: int, stopwords: list, stemming: bool = True) -> list[
    str]:
    """
    Serialization of prepare_utterance function with word merging.
    It takes a list of utterances with specified preprocess arguments, and return the list of preprocessed utterances.
    """
    return [' '.join(preprocess_utterance(utterance=line,
                                          min_word_length=min_word_length,
                                          stopwords=stopwords,
                                          stemming=stemming
                                          ))
            for line in utterances]

def create_word_vocabulary(data: list) -> list:
    """
    Create a list of all used words. Useful for vectorization.
    """
    all_words = set()
    for sentence in data:
        all_words = all_words.union(set(sentence.split()))
    return list(all_words)


def create_and_set_mlflow_experiment(experiment_name: str, artifact_location: str|None=None, tags:dict[str,Any]|None=None) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location.

    :param experiment_name The name of the experiment to create.
    :param artifact_location: The artifact location of the experiment to create.  
    :param tags: The tags of the experiment to create.
    :return experiment_id: The id of the created experiment.
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id

 
 #The following functions are used to validate user input by keyboard for the different experiments.
 
def wrong_parameter(parameter:str, possible_parameter: list[str]) -> bool:
    if parameter in possible_parameter:
        return False
    return True

def is_non_negative_float(number: str) -> bool:
    try:
        if float(number)>=0:
            return True
        else: 
            return False
    except ValueError:
        return False

def is_natural_number(number: str, greater_than: int=0) -> bool:
    try:
        if int(number) > greater_than:
            return True
    except ValueError:
        return False
    return False

def is_ratio(number: str) -> bool:
    try:
        if float(number)>0 and float(number)<=1:
            return True
        else: 
            return False
    except ValueError:
        return False
    