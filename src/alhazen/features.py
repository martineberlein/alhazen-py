from typing import List, Dict
from abc import ABC, abstractmethod
from numpy import nanmax, isnan

from collections import defaultdict
import re

from fuzzingbook.Grammars import Grammar
from fuzzingbook.GrammarFuzzer import expansion_to_children
from fuzzingbook.GrammarFuzzer import tree_to_string
from fuzzingbook.Grammars import reachable_nonterminals
from fuzzingbook.Parser import EarleyParser

from isla.derivation_tree import DerivationTree

from alhazen.input import Input


class Feature(ABC):
    """
    The abstract base class for grammar features.

    Args:
        name : A unique identifier name for this feature. Should not contain Whitespaces.
               e.g., 'type(<feature>@1)'
        rule : The production rule (e.g., '<function>' or '<value>').
        key  : The feature key (e.g., the chosen alternative or rule itself).
    """

    def __init__(self, name: str, rule: str, key: str) -> None:
        self.name = name
        self.rule = rule
        self.key = key
        super().__init__()

    def __repr__(self) -> str:
        """Returns a printable string representation of the feature."""
        return self.name_rep()

    @abstractmethod
    def name_rep(self) -> str:
        pass

    @abstractmethod
    def get_feature_value(self, derivation_tree) -> float:
        """Returns the feature value for a given derivation tree of an input."""
        pass


class ExistenceFeature(Feature):
    """
    This class represents existence features of a grammar. Existence features indicate
    whether a particular production rule was used in the derivation sequence of an input.
    For a given production rule P -> A | B, a production existence feature for P and
    alternative existence features for each alternative (i.e., A and B) are defined.

    name : A unique identifier name for this feature. Should not contain Whitespaces.
           e.g., 'exist(<digit>@1)'
    rule : The production rule.
    key  : The feature key, equal to the rule attribute for production features,
           or equal to the corresponding alternative for alternative features.
    """

    def __init__(self, name: str, rule: str, key: str) -> None:
        super().__init__(name, rule, key)

    def name_rep(self) -> str:
        if self.rule == self.key:
            return f"exists({self.rule})"
        else:
            return f"exists({self.rule} == {self.key})"

    def get_feature_value(self, derivation_tree: DerivationTree) -> float:
        """Counts the number of times this feature was matched in the derivation tree."""
        (node, children) = derivation_tree

        # The local match count (1 if the feature is matched for the current node, 0 if not)
        count = 0

        # First check if the current node can be matched with the rule
        if node == self.rule:
            # Production existence feature
            if self.rule == self.key:
                count = 1

            # Production alternative existence feature
            # We compare the children of the expansion with the actual children
            else:
                expansion_children = list(
                    map(lambda x: x[0], expansion_to_children(self.key))
                )
                node_children = list(map(lambda x: x[0], children))
                if expansion_children == node_children:
                    count = 1

        # Recursively compute the counts for all children and return the sum for the whole tree
        for child in children:
            count = max(count, self.get_feature_value(child))

        return count


class NumericInterpretation(Feature):
    """
    This class represents numeric interpretation features of a grammar. These features
    are defined for productions that only derive words composed of the characters
    [0-9], '.', and '-'. The returned feature value corresponds to the maximum
    floating-point number interpretation of the derived words of a production.

    name : A unique identifier name for this feature. Should not contain Whitespaces.
           e.g., 'num(<integer>)'
    rule : The production rule.
    """

    def __init__(self, name: str, rule: str) -> None:
        super().__init__(name, rule, rule)

    def name_rep(self) -> str:
        return f"num({self.key})"

    def get_feature_value(self, derivation_tree: DerivationTree) -> float:
        """Determines the maximum float of this feature in the derivation tree."""
        (node, children) = derivation_tree

        value = float("nan")
        if node == self.rule:
            try:
                # print(self.name, float(tree_to_string(derivation_tree)))
                value = float(
                    tree_to_string(derivation_tree)
                )  # TODO here fuzzingbook Derivation Tree
            except ValueError:
                # print(self.name, float(tree_to_string(derivation_tree)), "err")
                pass

        # Return maximum value encountered in tree, ignoring all NaNs
        tree_values = [value] + [self.get_feature_value(c) for c in children]
        if all(isnan(tree_values)):
            return value
        else:
            return nanmax(tree_values)


def extract_existence(grammar: Grammar) -> List[ExistenceFeature]:
    """
    Extracts all existence features from the grammar and returns them as a list.
    grammar : The input grammar.
    """

    features = []

    for rule in grammar:
        # add the rule
        features.append(ExistenceFeature(f"exists({rule})", rule, rule))
        # add all alternatives
        for count, expansion in enumerate(grammar[rule]):
            features.append(
                ExistenceFeature(f"exists({rule}@{count})", rule, expansion)
            )

    return features


RE_NONTERMINAL = re.compile(r"(<[^<> ]*>)")


def extract_numeric(grammar: Grammar) -> List[NumericInterpretation]:
    """
    Extracts all numeric interpretation features from the grammar and returns them as a list.

    grammar : The input grammar.
    """

    features = []

    # Mapping from non-terminals to derivable terminal chars
    derivable_chars = defaultdict(set)

    for rule in grammar:
        for expansion in grammar[rule]:
            # Remove non-terminal symbols and whitespace from expansion
            terminals = re.sub(RE_NONTERMINAL, "", expansion).replace(" ", "")

            # Add each terminal char to the set of derivable chars
            for c in terminals:
                derivable_chars[rule].add(c)

    # Repeatedly update the mapping until convergence
    while True:
        updated = False
        for rule in grammar:
            for r in reachable_nonterminals(grammar, rule):
                before = len(derivable_chars[rule])
                derivable_chars[rule].update(derivable_chars[r])
                after = len(derivable_chars[rule])

                # Set of derivable chars was updated
                if after > before:
                    updated = True

        if not updated:
            break

    numeric_chars = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"}

    for key in derivable_chars:
        # Check if derivable chars contain only numeric chars
        if len(derivable_chars[key] - numeric_chars) == 0:
            features.append(NumericInterpretation(f"num({key})", key))

    return features


def get_all_features(grammar: Grammar) -> List[Feature]:
    return extract_existence(grammar) + extract_numeric(grammar)


def test_features(features: List[Feature]) -> None:
    existence_features = 0
    numeric_features = 0

    for feature in features:
        if isinstance(feature, ExistenceFeature):
            existence_features += 1
        elif isinstance(feature, NumericInterpretation):
            numeric_features += 1

    assert existence_features == 27
    assert numeric_features == 4

    expected_feature_names = [
        "exists(<start>)",
        "exists(<start> == <function>(<term>))",
        "exists(<function>)",
        "exists(<function> == sqrt)",
        "exists(<function> == tan)",
        "exists(<function> == cos)",
        "exists(<function> == sin)",
        "exists(<term>)",
        "exists(<term> == -<value>)",
        "exists(<term> == <value>)",
        "exists(<value>)",
        "exists(<value> == <integer>.<integer>)",
        "exists(<value> == <integer>)",
        "exists(<integer>)",
        "exists(<integer> == <digit><integer>)",
        "exists(<integer> == <digit>)",
        "exists(<digit>)",
        "exists(<digit> == 0)",
        "exists(<digit> == 1)",
        "exists(<digit> == 2)",
        "exists(<digit> == 3)",
        "exists(<digit> == 4)",
        "exists(<digit> == 5)",
        "exists(<digit> == 6)",
        "exists(<digit> == 7)",
        "exists(<digit> == 8)",
        "exists(<digit> == 9)",
        "num(<term>)",
        "num(<value>)",
        "num(<digit>)",
        "num(<integer>)",
    ]

    actual_feature_names = list(map(lambda f: f.name_rep(), features))

    for feature_name in expected_feature_names:
        assert (
            feature_name in actual_feature_names
        ), f"Missing feature with name: {feature_name}"

    print("All checks passed!")


def collect_features(test_input: Input, all_features: List[Feature]) -> Dict:
    parsed_features = {}  # {"sample": str(test_input.tree)}
    for feature in all_features:
        parsed_features[feature.name] = 0
        parsed_features[feature.name] = feature.get_feature_value(
            test_input.tree
        )  # TODO can be done prettier!

    return parsed_features


def get_feature_vector(sample, grammar, features):
    """Returns the feature vector of the sample as a dictionary of feature values"""

    feature_dict = defaultdict(int)

    earley = EarleyParser(grammar)
    for tree in earley.parse(sample):
        for feature in features:
            feature_dict[feature.name] = feature.get_feature_value(tree)

    return feature_dict


def compute_feature_values(sample: str, grammar: Grammar) -> Dict[str, float]:
    """
    Extracts all feature values from an input.

    sample   : The input.
    grammar  : The input grammar.
    features : The list of input features extracted from the grammar.

    """
    parser = EarleyParser(grammar)

    features = {}
    for tree in parser.parse(sample):
        for feature in get_all_features(grammar):
            features[feature.name_rep()] = feature.get_feature_value(tree)
    return features
