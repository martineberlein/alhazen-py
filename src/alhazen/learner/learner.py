from abc import ABC
from typing import Iterable, Optional
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree
from pandas import DataFrame, concat
from fuzzingbook.Parser import EarleyParser

from dbg.data.input import Input
from dbg.explanation.candidate import ExplanationSet
from dbg.learner.learner import Learner

from alhazen.data.features import Feature
from alhazen.data.input import AlhazenInput, OracleResult
from alhazen.learner.input_specifications import extracting_prediction_paths, InputSpecification, create_new_input_specification, SPECIFICATION_GRAMMAR


class AlhazenLearner(Learner):

    def learn_explanation(self, test_inputs: Iterable[Input], **kwargs) -> Optional[ExplanationSet]:
        pass

    def get_explanations(self) -> Optional[ExplanationSet]:
        pass

    def get_best_candidates(self) -> Optional[ExplanationSet]:
        pass


class SKLearnLearner(ABC):

    def __init__(self):
        self.data = DataFrame()

    def train(self, test_inputs: Iterable[AlhazenInput], **kwargs):
        pass

    def predict(self, test_input: AlhazenInput, **kwargs):
        pass

    def _update_data(self, test_inputs: Iterable[AlhazenInput], **kwargs):
        data = []
        for inp in test_inputs:
            if inp.oracle != OracleResult.UNDEFINED:
                learning_data = inp.features
                learning_data["oracle"] = inp.oracle
                data.append(learning_data)

        new_data = DataFrame.from_records(data)

        if 0 != len(new_data):
            if self.data is None:
                self.data = new_data
            else:
                self.data = concat([self.data, new_data], sort=False)

        return self.data

class DecisionTreeLearner(SKLearnLearner):
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

    def train(self, test_inputs: Iterable[AlhazenInput], **kwargs):
        """
        Trains and returns a DecisionTreeClassifier learned on a given pandas Dataframe.
        """
        assert all(map(lambda x: isinstance(x, AlhazenInput), test_inputs))
        data = self._update_data(test_inputs)


        sample_bug_count = len([x for x in test_inputs if x.oracle == OracleResult.FAILING])
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
                str("FAILING"): (1.0 / sample_bug_count),
                str("PASSING"): (1.0 / (sample_count - sample_bug_count)),
            },
        )
        # self.__tree = treetools.remove_infeasible(clf, features) # MARTIN: This is optional, but is a nice extension
        # that results in nicer decision trees
        return clf.fit(x_train, y_train)

    def predict(self, test_input: AlhazenInput, **kwargs):
        pass

    def get_input_specifications(
        self,
        decision_tree,
        all_features: list[Feature],
        feature_names: list[str],
        data: DataFrame,
        **kwargs
    ) -> list[InputSpecification]:
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


def show_tree(clf, feature_names):
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=["FAILING", "PASSING"],
        filled=True,
        rounded=True,
    )
    return graphviz.Source(dot_data)