import unittest

from alhazen import Alhazen

from alhazen.isla_helper import decision_tree_to_isla_constraint

from alhazen_formalizations.calculator import initial_inputs, grammar_alhazen as grammar, prop
from alhazen.feature_collector import Collector


class TestConstraints(unittest.TestCase):
    def setUp(self) -> None:
        self.MAX_ITERATION = 30
        self.alhazen = Alhazen(
            initial_inputs=initial_inputs,
            grammar=grammar,
            evaluation_function=prop,
            max_iter=self.MAX_ITERATION,
        )
        self.collector = Collector(grammar)

    def test_constraints(self):
        trees = self.alhazen.run()
        clf = trees[self.MAX_ITERATION - 1]

        all_features = self.collector.get_all_features()

        constraints = decision_tree_to_isla_constraint(clf=clf, all_features=all_features)

        print(constraints)


if __name__ == "__main__":
    unittest.main()
