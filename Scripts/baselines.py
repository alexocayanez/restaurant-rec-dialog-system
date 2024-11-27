from collections import Counter
from data import Data
import collections
from sklearn.metrics import accuracy_score, precision_score, recall_score


class MajoritySystem:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.train()

    def train(self):
        counter = collections.Counter(self.y_train)
        self.prediction = counter.most_common(1)[0][0]
        return

    def predict(self, utterance: str):
        return self.prediction

    def predict_tag(self, utterance: str):
        return self.predict(utterance)

    def test_accuracy(self):
        self.y_test_predicted = [self.predict(x) for x in self.x_test]
        return accuracy_score(self.y_test, self.y_test_predicted)

    def test_precision(self):
        self.y_test_predicted = [self.predict(x) for x in self.x_test]
        return precision_score(self.y_test, self.y_test_predicted, average="macro", zero_division=0)

    def test_recall(self):
        self.y_test_predicted = [self.predict(x) for x in self.x_test]
        return recall_score(self.y_test, self.y_test_predicted, average="macro", zero_division=0)


class RuleBasedSystem:
    ACTS_RELATED_WORDS = Data.DIALOG_ACTS_SET

    ## THE WORDS THAT APPEAR HIGHER IN THE RANKING SHOULD BE CHECKED ABOUT APPEARING IN OTHER CLASSES
    KEYWORDS_RANKING = [
        # First we classify the "outliers" of reqmore (sencences with only 1 word)
        {"more": "reqmore"},
        # Word "no" has over 100% appearence in class "negate"
        {"no": "negate"},
        # Now we will separate between "thanyou" and "bye". If the utterance is 'thank you good bye' it is classified as "thankyou".
        # But, if something appears before ('alright thank you good bye') it will be "bye"
        {"thank": "thankyou", "thanks": "thankyou"},
        {"bye": "bye"},
        # Afirm and restart
        {"yes": "affirm", "right": "affirm", "start": "restart", "reset": "restart"},
        # repeat and hello
        {"hi": "hello", "hello": "hello", "looking": "hello", "repeat": "repeat", "back": "repeat"},
        # request and reqalts
        {"phone": "request", "address": "request", "code": "request", "cost": "request",
         "about": "reqalts", "else": "reqalts", "how": "reqalts"},
        # deny ack confirm
        {"wrong": "deny", "not": "deny", "dont": "deny",
         "okay": "ack", "good": "ack", "um": "ack",
         "it": "confirm", "is": "confirm", "serve": "confirm"},
        # Inform
        {"food": "inform", "restaurant": "inform"},
        # Null
        {"noise": "null", "sil": "null", "unintelligible": "null", "um": "null"}
    ]

    def __init__(self, x_train: list[str]=[], x_test: list[str]=[], y_train: list[str]=[], y_test: list[str]=[]):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def predict(self, utterance: str):
        words = utterance.split()
        for dictionary in self.KEYWORDS_RANKING:
            for word in words:
                if word in dictionary:
                    return dictionary[word]
        return "inform"

    def predict_tag(self, utterance: str):
        return self.predict(utterance)

    def test_accuracy(self):
        self.y_test_predicted = [self.predict(x) for x in self.x_test]
        return accuracy_score(self.y_test, self.y_test_predicted)

    def test_precision(self):
        self.y_test_predicted = [self.predict(x) for x in self.x_test]
        return precision_score(self.y_test, self.y_test_predicted, average="macro", zero_division=0)

    def test_recall(self):
        self.y_test_predicted = [self.predict(x) for x in self.x_test]
        return recall_score(self.y_test, self.y_test_predicted, average="macro", zero_division=0)
