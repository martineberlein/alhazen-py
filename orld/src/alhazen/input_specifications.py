import string
import pandas
from typing import List

from fuzzingbook.Parser import tree_to_string
from fuzzingbook.Grammars import Grammar

from alhazen.features import Feature
from alhazen.requirementExtractionDT.requirements import tree_to_paths


def extracting_prediction_paths(clf, feature_names, data, classes=None):
    # determine the bounds
    bounds = (
        pandas.DataFrame(
            [
                {"feature": c, "min": data[c].min(), "max": data[c].max()}
                for c in feature_names
            ],
            columns=["feature", "min", "max"],
        )
        .set_index(["feature"])
        .transpose()
    )

    # go through tree leaf by leaf
    all_reqs = set()
    for path in tree_to_paths(clf, feature_names, classes):
        if len(path) < 1:
            continue
        # generate conditions
        for i in range(0, len(path) + 1):
            reqs_list = []
            bins = format(i, "#0{}b".format(len(path) + 2))[2:]
            for p, b in zip(range(0, len(bins)), bins):
                r = path.get(p)
                if "1" == b:
                    reqs_list.append(r.get_neg_ext(bounds))
                else:
                    reqs_list.append([r.get_str_ext()])
            for reqs in all_combinations(reqs_list):
                all_reqs.add(", ".join(sorted(reqs)))
    return all_reqs


def all_combinations(reqs_lists):
    result = [[]]
    for reqs in reqs_lists:
        t = []
        for r in reqs:
            for i in result:
                t.append(i + [r])
        result = t
    return result


SPECIFICATION_GRAMMAR: Grammar = {
    "<start>": ["<req_list>"],
    "<req_list>": ["<req>", "<req>" ", " "<req_list>"],
    "<req>": ["<feature>" " " "<quant>" " " "<num>"],
    "<feature>": [
        "exists(<string>)",
        "num(<string>)",
        # currently not used
        "char(<string>)",
        "len(<string>)",
    ],
    "<quant>": ["<", ">", "<=", ">="],
    "<num>": ["-<value>", "<value>"],
    "<value>": ["<integer>.<integer>", "<integer>"],
    "<integer>": ["<digit><integer>", "<digit>"],
    "<digit>": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<string>": ["<letters>"],
    "<letters>": ["<letter><letters>", "<letter>"],
    "<letter>": list(string.ascii_letters + string.digits + string.punctuation),
}


class Requirement:
    """
    This class represents a requirement for a new input sample that should be generated.
    This class contains the feature that should be fullfiled (Feature), a quantifier
    ("<", ">", "<=", ">=") and a value. For instance exist(feature) >= 0.5 states that
    the syntactical existence feature should be used to produce a new input.

    feature  : Is the associated feature class
    quant    : The quantifier
    value    : The value of the requirement. Note that for existence features this value
                is allways between 0 and 1.
    """

    def __init__(self, feature: Feature, quantificator, value):
        self.feature: Feature = feature
        self.quant = quantificator
        self.value = value

    def __str__(self):
        return f"Requirement({self.feature.name} {self.quant} {self.value})"


class InputSpecification:
    """
    This class represents a complete input specification of a new input. A input specification
    consists of one or more requirements.

    requirements  : Is a list of all requirements that must be used.
    """

    def __init__(self, requirements: List[Requirement]):
        self.requirements: List[Requirement] = requirements

    def __str__(self):
        # Handle first element
        s = f"{str(self.requirements[0])}"
        for count in range(1, len(self.requirements)):
            s += ", " + str(self.requirements[count])

        return f"NewInputSpecification({s})"


def get_all_subtrees(derivation_tree, non_terminal):
    """
    Iteratively returns a list of subtrees that start with a given non_terminal.
    """

    subtrees = []
    (node, children) = derivation_tree

    if node == non_terminal:
        subtrees.append(derivation_tree)

    for child in children:
        subtrees = subtrees + get_all_subtrees(child, non_terminal)

    return subtrees


def create_new_input_specification(derivation_tree, all_features) -> InputSpecification:
    """
    This function creates a new input specification for a parsed decision tree path.
    The input derivation_tree corresponds to an already negated path in the decision tree.
    """

    requirement_list = []

    for req in get_all_subtrees(derivation_tree, "<req>"):
        feature_name = tree_to_string(get_all_subtrees(req, "<feature>")[0])
        quant = tree_to_string(get_all_subtrees(req, "<quant>")[0])
        value = tree_to_string(get_all_subtrees(req, "<num>")[0])

        feature_class = None
        for f in all_features:
            if f.name == feature_name:
                feature_class = f

        requirement_list.append(Requirement(feature_class, quant, value))

    return InputSpecification(requirement_list)
