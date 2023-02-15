#!/usr/bin/env python

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
