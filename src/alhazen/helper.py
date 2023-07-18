import graphviz
from sklearn import tree

from alhazen.requirementExtractionDT.treetools import remove_unequal_decisions


NEGATIVE_SHIFT: int = 16  # where should we define this constant, maybe this can also be a class variable


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


def get_clf_text(clf, feature_names):
    clf_text = tree.export_text(
        decision_tree=clf,
        feature_names=feature_names)
    return clf_text


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


# problems will arise when using modulo
# addition, subtraction, multiplication and division should work fine

def shift_from_negative(number: int) -> int:
    return number + (1 << NEGATIVE_SHIFT)


def revert_shift_to_negative(number: int) -> int:
    return number - (1 << NEGATIVE_SHIFT)


def is_negative(number: int) -> int:
    return True if number > NEGATIVE_SHIFT else False


def is_zero(number: int) -> int:
    return True if number == NEGATIVE_SHIFT else False

