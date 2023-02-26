#!/usr/bin/env python
from typing import List
from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from fuzzingbook.Parser import EarleyParser

from alhazen.oracle import OracleResult
from alhazen.features import Feature
from alhazen.input_specifications import (
    SPECIFICATION_GRAMMAR,
    InputSpecification,
    extracting_prediction_paths,
    create_new_input_specification,
)


class Learner(ABC):
    def __init__(self):
        self.learner = None

    @abstractmethod
    def train(self, data: DataFrame, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_input_specifications(
        self,
        model,
        all_features: List[Feature],
        feature_names: List[str],
        data: DataFrame,
        **kwargs
    ) -> List[InputSpecification]:
        raise NotImplementedError

    @abstractmethod
    def visualize(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
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

    def train(self, data: DataFrame, **kwargs):
        """
        Trains and returns a DecisionTreeClassifier learned on a given pandas Dataframe.
        """
        assert all(map(lambda x: isinstance(x, OracleResult), data["oracle"]))

        sample_bug_count = len(data[(data["oracle"].astype(str) == "BUG")])
        sample_count = len(data)

        data = data.fillna(0)
        x_train = data.drop(["oracle"], axis=1)
        y_train = data["oracle"].astype(str)

        clf = DecisionTreeClassifier(
            min_samples_leaf=self.min_sample_leaf,
            min_samples_split=self.min_sample_split,  # minimal value
            max_features=self.max_features,
            max_depth=self.max_depth,  # max depth of the decision tree
            class_weight={
                str("BUG"): (1.0 / sample_bug_count),
                str("NO_BUG"): (1.0 / (sample_count - sample_bug_count)),
            },
        )
        # self.__tree = treetools.remove_infeasible(clf, features) # MARTIN: This is optional, but is a nice extension
        # that results in nicer decision trees
        return clf.fit(x_train, y_train)

    def get_input_specifications(
        self,
        decision_tree,
        all_features: List[Feature],
        feature_names: List[str],
        data: DataFrame,
        **kwargs
    ) -> List[InputSpecification]:
        assert isinstance(decision_tree, DecisionTreeClassifier)

        prediction_paths = extracting_prediction_paths(
            decision_tree, feature_names, data
        )
        input_specifications = []

        for r in prediction_paths:
            parser = EarleyParser(SPECIFICATION_GRAMMAR)
            try:
                for tree in parser.parse(r):
                    input_specifications.append(
                        create_new_input_specification(tree, all_features)
                    )
            except SyntaxError:
                # Catch Parsing Syntax Errors: num(<term>) in [-900, 0] will fail; Might fix later
                # For now, inputs following that form will be ignored
                pass

        return input_specifications

    def predict(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError


class RandomForestLearner(Learner):
    def __init__(self):
        super().__init__()

    def train(self, data: DataFrame, **kwargs):
        assert all(map(lambda x: isinstance(x, OracleResult), data["oracle"]))

        sample_bug_count = len(data[(data["oracle"].astype(str) == "BUG")])
        sample_count = len(data)
        if 0 == (sample_bug_count - sample_count):
            raise AssertionError("There are no samples for the no bug case.")
        if 0 == sample_bug_count:
            raise AssertionError("There are no samples for the bug case.")

        data = data.fillna(0)
        x_train = data.drop(["oracle"], axis=1)
        y_train = data["oracle"].astype(str)

        clf = RandomForestClassifier(
            n_estimators=10,
            max_features=None,
            max_depth=5,
            min_samples_split=2,
            class_weight={
                str(OracleResult.BUG): (1.0 / sample_bug_count),
                str(OracleResult.NO_BUG): (1.0 / (sample_count - sample_bug_count)),
            },
        )
        return clf.fit(x_train, y_train)

    def get_input_specifications(
        self,
        random_forest: RandomForestClassifier,
        all_features: List[Feature],
        feature_names: List[str],
        data: DataFrame,
        **kwargs
    ) -> List[InputSpecification]:
        assert isinstance(random_forest, RandomForestClassifier)

        prediction_paths = set()
        for tree in random_forest.estimators_:
            prediction_paths.update(
                extracting_prediction_paths(
                    tree, feature_names, data, random_forest.classes_
                )
            )

        input_specifications = []

        for r in prediction_paths:
            parser = EarleyParser(SPECIFICATION_GRAMMAR)
            try:
                for tree in parser.parse(r):
                    input_specifications.append(
                        create_new_input_specification(tree, all_features)
                    )
            except SyntaxError:
                # Catch Parsing Syntax Errors: num(<term>) in [-900, 0] will fail; Might fix later
                # For now, inputs following that form will be ignored
                pass

        return input_specifications

    def predict(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError


class XGBTLearner(Learner):
    def train(self, data: DataFrame, **kwargs):
        pass

    def get_input_specifications(
        self,
        model,
        all_features: List[Feature],
        feature_names: List[str],
        data: DataFrame,
        **kwargs
    ) -> List[InputSpecification]:
        pass

    def visualize(self):
        pass

    def predict(self):
        pass
