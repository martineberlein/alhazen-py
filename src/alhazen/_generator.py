import numpy as np
from itertools import product
from fuzzingbook.GrammarFuzzer import GrammarFuzzer

from dbg.explanation.candidate import ExplanationSet
from dbg.generator.generator import Generator

from alhazen import Grammar
from alhazen._data import AlhazenInput
from alhazen.features.features import Feature
from alhazen.features.collector import GrammarFeatureCollector


class AlhazenGenerator(Generator):

    def __init__(self, grammar: Grammar, **kwargs):
        super().__init__(grammar, **kwargs)
        self.fuzzer = GrammarFuzzer(grammar)
        self.collector = GrammarFeatureCollector(grammar)

    def generate(self, explanation, *args, **kwargs):
        val_inputs = []
        for _ in range(10):
            for _ in range(10):
                val_inputs.append(AlhazenInput(self.fuzzer.fuzz_tree()))

            for inp in val_inputs:
                inp.features = self.collector.collect_features(inp)

            for inp in val_inputs:
                if all(prop.evaluate(inp) for prop in explanation):
                    return inp


class Property:

    def __init__(self, feature, operator, value):
        assert isinstance(feature, Feature)

        self.feature: Feature = feature
        self.operator = operator
        self.value = value

    def evaluate(self, test_input: AlhazenInput):
        input_value = test_input.features.get_feature_value(self.feature)
        if self.operator == "<=":
            return input_value <= self.value
        elif self.operator == ">":
            return input_value > self.value
        else:
            raise ValueError(f"Invalid operator: {self.operator}")

    def __str__(self):
        return f"{self.feature} {self.operator} {self.value}"

    def __neg__(self):
        operator = "<=" if self.operator == ">" else ">"
        return Property(self.feature, operator, self.value)


class HypothesisProducer:

    def produce(
        self, explanations: ExplanationSet, all_features: list[Feature]
    ) -> list[list[Property]]:
        positive = []
        for explanation in explanations:
            positive_hypotheses = get_positive_paths(
                explanation.explanation, all_features
            )
            positive.extend(positive_hypotheses)

        negated = []
        for hypothesis in positive:
            negated_hypotheses: list[list[Property]] = self.negate(hypothesis)
            negated.extend(negated_hypotheses)

        hypotheses = positive + negated
        return hypotheses

    @staticmethod
    def negate(hypothesis: list[Property]) -> list[list[Property]]:
        negated_hypotheses = [
            [-prop if bit else prop for prop, bit in zip(hypothesis, combination)]
            for combination in product([0, 1], repeat=len(hypothesis))
            if any(combination)
        ]
        return negated_hypotheses


def get_positive_paths(
    tree,
    all_features: list[Feature],
    class_label=0,
    remove_redundant_split: bool = True,
):
    """Extracts paths leading to a positive prediction (class_label).

    Args:
        tree: Fitted DecisionTreeClassifier model.
        all_features: List of feature names.
        class_label: The target class for which to extract paths.
        remove_redundant_split: Whether to remove redundant splits.

    Returns:
        A list of paths as readable conditions.
    """
    tree_ = tree.tree_
    feature_names = [str(feature) for feature in all_features]

    def traverse(node, path):
        if tree_.feature[node] == -2:  # Leaf node
            predicted_class = np.argmax(tree_.value[node])  # Get predicted class
            if predicted_class == class_label:
                paths.append(path)
            return

        # Get child nodes
        left, right = tree_.children_left[node], tree_.children_right[node]

        # Get predicted classes for left and right nodes
        left_prediction = int(np.argmax(tree_.value[left]))
        right_prediction = int(np.argmax(tree_.value[right]))

        # Stop if both children predict the same class (redundant split)
        if remove_redundant_split and left_prediction == right_prediction:
            if left_prediction == class_label:
                paths.append(path)
            return

        # Otherwise, continue traversing
        feature = all_features[tree_.feature[node]]
        threshold = tree_.threshold[node]

        traverse(left, path + [Property(feature, "<=", threshold)])
        traverse(right, path + [Property(feature, ">", threshold)])

    paths = []
    traverse(0, [])
    return paths
