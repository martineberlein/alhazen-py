import unittest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from alhazen import Alhazen
from alhazen.generator import AdvancedGenerator
from alhazen.features import NUMERIC_INTERPRETATION_FEATURE, EXISTENCE_FEATURE
from alhazen.oracle import OracleResult
from alhazen_formalizations.calculator import (
    initial_inputs,
    prop,
    grammar_alhazen as grammar,
)


class TestAlhazenGenerate(unittest.TestCase):

    def setUp(self) -> None:
        self.alhazen = Alhazen(
            initial_inputs=initial_inputs,
            grammar=grammar,
            evaluation_function=prop,
            features={EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE}
        )
        _ = self.alhazen.run()

    @unittest.skip
    def test_alhazen_generate(self):
        from alhazen.requirementExtractionDT.treetools import grouped_rules

        print(grouped_rules(self.alhazen._get_last_model()))

        generator = AdvancedGenerator(grammar=grammar)

        test_inputs = []
        for _ in range(10):
            test_inputs.append(self.alhazen.generate(bug_triggering=True, generator=generator))

        for inp in test_inputs:
            self.assertEqual(prop(inp.tree), OracleResult.BUG)


if __name__ == '__main__':
    unittest.main()
