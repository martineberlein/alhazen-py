import unittest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from alhazen import Alhazen
from alhazen.learner import RandomForestLearner
from alhazen_formalizations.calculator import (
    initial_inputs,
    prop,
    grammar_alhazen as grammar,
)


class TestAlhazen(unittest.TestCase):
    def test_initialization(self):
        alhazen = Alhazen(
            initial_inputs=initial_inputs,
            grammar=grammar,
            evaluation_function=prop,
        )
        result = alhazen.run()

        self.assertEqual(len(result), 10)
        self.assertTrue(
            all([isinstance(tree, DecisionTreeClassifier) for tree in result])
        )

    def test_random_forest(self):
        alhazen = Alhazen(
            initial_inputs=initial_inputs,
            grammar=grammar,
            evaluation_function=prop,
            learner=RandomForestLearner(),
        )
        result = alhazen.run()

        self.assertEqual(len(result), 10)
        self.assertTrue(
            all([isinstance(tree, RandomForestClassifier) for tree in result])
        )


if __name__ == "__main__":
    unittest.main()
