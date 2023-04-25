import graphviz
from sklearn import tree

from alhazen.requirementExtractionDT.treetools import remove_unequal_decisions


def show_tree(clf, feature_names):
    dot_data = tree.export_graphviz(
        remove_unequal_decisions(clf),
        out_file=None,
        feature_names=feature_names,
        class_names=["BUG", "NO_BUG"],
        filled=True,
        rounded=True,
    )
    return graphviz.Source(dot_data)


def get_dot_data(clf, feature_names):
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=["BUG", "NO_BUG"],
        filled=True,
        rounded=True,
    )
    return dot_data
