import unittest

from sklearn.tree import DecisionTreeClassifier
from dbg.explanation.candidate import ExplanationSet

from alhazen.core import Alhazen
from alhazen._learner import AlhazenExplanation
from resources.calculator import initial_inputs, grammar, oracle


class TestAlhazen(unittest.TestCase):

    def test_initialization(self):
        alhazen = Alhazen(
            initial_inputs=initial_inputs,
            grammar=grammar,
            oracle=oracle
        )
        explanations = alhazen.explain()
        self.assertIsInstance(explanations, ExplanationSet)
        self.assertEqual(len(explanations), 1)

        for explanation in explanations:
            self.assertIsInstance(explanation, AlhazenExplanation)
            self.assertIsInstance(explanation.explanation, DecisionTreeClassifier)


if __name__ == "__main__":
    unittest.main()
