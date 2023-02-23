import unittest

import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

from alhazen.oracle import OracleResult
from alhazen.learner import Learner, DecisionTreeLearner


class TestLearner(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [
            {
                "function-sqrt": 1,
                "function-cos": 0,
                "function-sin": 0,
                "number": -900,
            },  # sqrt(-900)
            {
                "function-sqrt": 0,
                "function-cos": 1,
                "function-sin": 0,
                "number": 300,
            },  # cos(300)
            {
                "function-sqrt": 1,
                "function-cos": 0,
                "function-sin": 0,
                "number": -1,
            },  # sqrt(-1)
            {
                "function-sqrt": 0,
                "function-cos": 1,
                "function-sin": 0,
                "number": -10,
            },  # cos(-10)
            {
                "function-sqrt": 0,
                "function-cos": 0,
                "function-sin": 1,
                "number": 36,
            },  # sin(36)
            {
                "function-sqrt": 0,
                "function-cos": 0,
                "function-sin": 1,
                "number": -58,
            },  # sin(-58)
            {
                "function-sqrt": 1,
                "function-cos": 0,
                "function-sin": 0,
                "number": 27,
            },  # sqrt(27)
        ]

        self.oracle = [
            OracleResult.BUG,
            OracleResult.NO_BUG,
            OracleResult.BUG,
            OracleResult.NO_BUG,
            OracleResult.NO_BUG,
            OracleResult.NO_BUG,
            OracleResult.NO_BUG,
        ]

    def test_decision_tee(self):
        x = DictVectorizer().fit_transform(self.features).toarray()

        oracle_clean = [str(c) for c in self.oracle]

        clf = DecisionTreeClassifier(random_state=10)
        clf.fit(x, oracle_clean)

        self.assertTrue(isinstance(clf, DecisionTreeClassifier))

    def test_train_tree(self):
        raw_data = [
            {
                # "sample": 'sqrt(-900)',
                "function-sqrt": 1,
                "function-cos": 0,
                "function-sin": 0,
                "number": -900,
                "oracle": OracleResult.BUG,
            },  # sqrt(-900)
            {
                # "sample": 'cos(300)',
                "function-sqrt": 0,
                "function-cos": 1,
                "function-sin": 0,
                "number": 300,
                "oracle": OracleResult.NO_BUG,
            },  # cos(300)
            {
                # "sample": 'UNDEF(x)',
                "function-sqrt": 0,
                "function-cos": 0,
                "function-sin": 0,
                "number": 0,
                "oracle": OracleResult.UNDEF,
            },  # UNDEF
        ]

        data = pandas.DataFrame.from_records(data=raw_data)
        clf = DecisionTreeLearner().train(data)

        self.assertTrue(isinstance(clf, DecisionTreeClassifier))


if __name__ == "__main__":
    unittest.main()
