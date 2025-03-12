from typing import Iterable, Optional
from abc import ABC

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from dbg.explanation.candidate import ExplanationSet, Explanation
from dbg.learner.learner import Learner
from dbg.data.oracle import OracleResult

from alhazen._data import AlhazenInput
from alhazen import tree_to_if_then_statement


class AlhazenExplanation(Explanation):
    """
    AlhazenExplanation is a concrete implementation of an explanation
    based on a DecisionTreeClassifier.
    """

    def __init__(self, explanation: DecisionTreeClassifier, feature_names: list[str]):
        super().__init__(explanation)
        self.feature_names = feature_names

    def evaluate(self, inputs: set[AlhazenInput]):
        """
        Evaluates the explanation on a set of inputs.
        """
        for inp in inputs:
            if inp in self.cache.keys():
                continue
            eval_result = self.explanation.predict(
                pd.DataFrame.from_records([{**inp.features.features}])
            )[0]
            eval_result = True if eval_result == str(OracleResult.FAILING) else False
            if inp.oracle == OracleResult.FAILING:
                self.failing_inputs_eval_results.append(eval_result)
            else:
                self.passing_inputs_eval_results.append(eval_result)
            self.cache[inp] = eval_result

    def __neg__(self):
        return self

    def __str__(self):
        return self.tree_to_explanation()

    def tree_to_explanation(self):
        return tree_to_if_then_statement(self.explanation, self.feature_names)


class AlhazenLearner(Learner):

    def learn_explanation(
        self, test_inputs: set[AlhazenInput], **kwargs
    ) -> Optional[ExplanationSet]:
        sk_learner = DecisionTreeLearner()
        diagnosis = sk_learner.train(test_inputs)
        explanation = AlhazenExplanation(diagnosis, sk_learner.data.columns)
        self.explanations = ExplanationSet([explanation])
        return self.explanations


class SKLearnLearner(ABC):
    """Abstract base class for machine learning-based learners."""

    def __init__(self):
        self.data = pd.DataFrame()

    def train(self, test_inputs: Iterable[AlhazenInput], **kwargs):
        """Trains a model based on test inputs."""
        pass

    def _update_data(self, test_inputs: Iterable[AlhazenInput]) -> pd.DataFrame:
        """Updates and returns the training data DataFrame."""
        data_records = [
            {**inp.features.features, "oracle": inp.oracle}
            for inp in test_inputs
            if inp.oracle != OracleResult.UNDEFINED
        ]

        if data_records:
            new_data = pd.DataFrame.from_records(data_records)
            self.data = (
                pd.concat([self.data, new_data], sort=False)
                if not self.data.empty
                else new_data
            )

        return self.data


class DecisionTreeLearner(SKLearnLearner):
    """Decision tree learner using scikit-learns DecisionTreeClassifier."""

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
    def _compute_class_weights(
        data: pd.DataFrame, test_inputs: set[AlhazenInput]
    ) -> dict:
        """Computes class weights based on the distribution of failing and passing samples."""
        sample_bug_count = sum(
            1 for x in test_inputs if x.oracle == OracleResult.FAILING
        )
        sample_count = len(data)

        return {
            str(OracleResult.FAILING): 1.0 / sample_bug_count,
            str(OracleResult.PASSING): 1.0 / (sample_count - sample_bug_count),
        }

    def train(self, test_inputs: set[AlhazenInput], **kwargs) -> DecisionTreeClassifier:
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
            random_state=1,
        )

        self.clf.fit(x_train, y_train)
        return self.clf
