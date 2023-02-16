#!/usr/bin/env python
from typing import List
from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame


def train_tree(data: DataFrame) -> DecisionTreeClassifier:
    """
    Trains and returns a DecisionTreeClassifier learned on a given pandas Dataframe.
    """
    sample_bug_count = len(data[(data["oracle"].astype(str) == "BUG")])
    sample_count = len(data)
    data = data.fillna(0)

    clf = DecisionTreeClassifier(
        min_samples_leaf=1,
        min_samples_split=2,  # minimal value
        max_features=None,
        max_depth=5,  # max depth of the decision tree
        class_weight={
            str("BUG"): (1.0 / sample_bug_count),
            str("NO_BUG"): (1.0 / (sample_count - sample_bug_count)),
        },
    )
    clf = clf.fit(data.drop("oracle", axis=1), data["oracle"].astype(str))
    # self.__tree = treetools.remove_infeasible(clf, features) # MARTIN: This is optional, but is a nice extension
    # that results in nicer decision trees
    return clf


class Learner(ABC):
    def __init__(self):
        self.learner = None

    @abstractmethod
    def learn(self, data: List, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_input_specifications(self):
        raise NotImplementedError


class DecisionTreeLearner(Learner):
    def __init__(
        self,
        min_sample_leaf: int = 1,
        min_samples_split: int = 2,
        max_features=None,
        max_depth: int = 5,
    ):
        super().__init__()
        self.min_sample_leaf = min_sample_leaf
        self.min_sample_split = min_samples_split
        self.max_features = max_features
        self.max_depth = max_depth

    def learn(self, data: List, **kwargs):
        raise NotImplementedError

    def get_input_specifications(self):
        raise NotImplementedError


class RandomForestLearner(Learner):
    def __init__(self):
        super().__init__()

    def learn(self, **kwargs):
        raise NotImplementedError

    def get_input_specifications(self):
        raise NotImplementedError
