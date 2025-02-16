from abc import ABC
from typing import Iterable, Optional, List
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree
from pandas import DataFrame, concat
from fuzzingbook.Parser import EarleyParser

from dbg.explanation.candidate import ExplanationSet
from dbg.learner.learner import Learner

from alhazen.data.features import Feature
from alhazen.data.input import AlhazenInput, OracleResult
from alhazen.learner.input_specifications import (
    extracting_prediction_paths,
    InputSpecification,
    create_new_input_specification,
    SPECIFICATION_GRAMMAR,
)


class AlhazenLearner(Learner):
    """Abstract learner class for Alhazen."""

    def learn_explanation(self, test_inputs: Iterable[AlhazenInput], **kwargs) -> Optional[ExplanationSet]:
        """Learns an explanation based on given test inputs."""
        pass

    def get_explanations(self) -> Optional[ExplanationSet]:
        """Returns the set of explanations."""
        pass

    def get_best_candidates(self) -> Optional[ExplanationSet]:
        """Returns the best explanation candidates."""
        pass


class SKLearnLearner(ABC):
    """Abstract base class for machine learning-based learners."""

    def __init__(self):
        self.data = DataFrame()

    def train(self, test_inputs: Iterable[AlhazenInput], **kwargs):
        """Trains a model based on test inputs."""
        pass

    def predict(self, test_input: AlhazenInput, **kwargs):
        """Predicts outcomes for a given test input."""
        pass

    def _update_data(self, test_inputs: Iterable[AlhazenInput]) -> DataFrame:
        """Updates and returns the training data DataFrame."""
        data_records = [
            {**inp.features, "oracle": inp.oracle}
            for inp in test_inputs
            if inp.oracle != OracleResult.UNDEFINED
        ]

        if data_records:
            new_data = DataFrame.from_records(data_records)
            self.data = concat([self.data, new_data], sort=False) if not self.data.empty else new_data

        return self.data


class DecisionTreeLearner(SKLearnLearner):
    """Decision tree learner using scikit-learn's DecisionTreeClassifier."""

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

        self.clf: Optional[DecisionTreeClassifier] = None

    @staticmethod
    def _compute_class_weights(data: DataFrame, test_inputs: Iterable[AlhazenInput]) -> dict:
        """Computes class weights based on the distribution of failing and passing samples."""
        sample_bug_count = sum(1 for x in test_inputs if x.oracle == OracleResult.FAILING)
        sample_count = len(data)

        if sample_bug_count == 0 or sample_count - sample_bug_count == 0:
            return None  # Avoid division by zero if data is imbalanced

        return {
            str(OracleResult.FAILING): 1.0 / sample_bug_count,
            str(OracleResult.PASSING): 1.0 / (sample_count - sample_bug_count),
        }

    def train(self, test_inputs: Iterable[AlhazenInput], **kwargs) -> DecisionTreeClassifier:
        """
        Trains and returns a DecisionTreeClassifier on the provided test inputs.
        """
        data = self._update_data(test_inputs)
        if data.empty:
            raise ValueError("No valid data available for training.")

        data.fillna(0, inplace=True)
        x_train, y_train = data.drop(columns=["oracle"]), data["oracle"].astype(str)

        class_weights = self._compute_class_weights(data, test_inputs)

        self.clf = DecisionTreeClassifier(
            min_samples_leaf=self.min_sample_leaf,
            min_samples_split=self.min_sample_split,
            max_features=self.max_features,
            max_depth=self.max_depth,
            class_weight=class_weights,
        )

        self.clf.fit(x_train, y_train)
        return self.clf

    def predict(self, test_input: AlhazenInput | Iterable[AlhazenInput], ** kwargs) -> list[str] | str:
        """
        Predicts the class label for a given test input (or multiple test inputs) using the trained model.

        Args:
            test_input (AlhazenInput | Iterable[AlhazenInput]): Single input or list of inputs for prediction.

        Returns:
            list[str] | str: Predicted class labels ('FAILING' or 'PASSING').
        """
        if self.clf is None:
            raise ValueError("The model has not been trained yet. Call `train()` first.")

        # Normalize input to handle both single and multiple inputs
        if isinstance(test_input, AlhazenInput):
            test_inputs = [test_input]
            single_input = True
        else:
            test_inputs = list(test_input)
            single_input = False

        # Convert inputs to DataFrame format
        feature_data = DataFrame([inp.features for inp in test_inputs]).fillna(0)

        # Perform predictions
        predictions = self.clf.predict(feature_data).tolist()

        return predictions[0] if single_input else predictions

    def get_input_specifications(
        self,
        decision_tree: DecisionTreeClassifier,
        all_features: List[Feature],
        feature_names: List[str],
        data: DataFrame,
        **kwargs
    ) -> List[InputSpecification]:
        """
        Extracts input specifications based on decision tree prediction paths.
        """
        assert isinstance(decision_tree, DecisionTreeClassifier)

        prediction_paths = extracting_prediction_paths(decision_tree, feature_names, data)
        input_specifications = []

        for rule in prediction_paths:
            parser = EarleyParser(SPECIFICATION_GRAMMAR)
            try:
                for parse_tree in parser.parse(rule):
                    input_specifications.append(create_new_input_specification(parse_tree, all_features))
            except SyntaxError:
                # Parsing may fail for certain expressions; these cases are skipped
                continue

        return input_specifications


def show_tree(clf: DecisionTreeClassifier, feature_names: List[str]) -> graphviz.Source:
    """
    Generates and returns a visual representation of a trained decision tree.
    """
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=["FAILING", "PASSING"],
        filled=True,
        rounded=True,
    )
    return graphviz.Source(dot_data)
