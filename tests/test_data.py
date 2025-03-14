import unittest
from typing import Tuple

from isla.derivation_tree import DerivationTree
from fuzzingbook.Parser import EarleyParser, Grammar, is_valid_grammar

from dbg.data.oracle import OracleResult

from alhazen.features.collector import GrammarFeatureCollector
from alhazen._data import AlhazenInput
from alhazen.features.features import FeatureVector
from resources.calculator import grammar, initial_inputs, oracle


class TestInputs(unittest.TestCase):
    def setUp(self) -> None:
        inputs = {"sqrt(-900)", "cos(10)"}

        self.test_inputs = set()
        for inp in inputs:
            self.test_inputs.add(
                AlhazenInput.from_str(grammar, inp, oracle(inp))
            )

        self.collector = GrammarFeatureCollector(grammar=grammar)

    def test_test_inputs(self):
        inputs = {"sqrt(-900)", "cos(10)"}
        oracles = [OracleResult.FAILING, OracleResult.PASSING]

        test_inputs = set()
        for inp, oracle_ in zip(inputs, oracles):
            test_input = AlhazenInput.from_str(grammar, inp)
            test_input.oracle = oracle_
            test_inputs.add(test_input)

        self.assertEqual(inputs, set(map(lambda f: str(f), test_inputs)))

        for inp, orc in zip(inputs, oracles):
            self.assertIn(
                (inp, orc), set(map(lambda x: (str(x), x.oracle), test_inputs))
            )

        self.assertFalse(
            set(map(lambda f: str(f.tree), test_inputs)).__contains__("cos(X)")
        )

    def test_input_execution(self):
        for inp in self.test_inputs:
            inp.oracle = oracle(inp)
            self.assertIsInstance(inp.oracle, OracleResult)

    def test_feature_extraction(self):
        for inp in self.test_inputs:
            inp.features = self.collector.collect_features(inp)
            self.assertIsNotNone(inp.features)
            self.assertIsInstance(inp.features, FeatureVector)

    def test_hash(self):
        grammar_: Grammar = {
            "<start>": ["<number>"],
            "<number>": ["<maybe_minus><one_nine>"],
            "<maybe_minus>": ["-", ""],
            "<one_nine>": [str(i) for i in range(1, 10)],
        }
        assert is_valid_grammar(grammar_)

        initial_test_inputs = ["-8", "-8"]

        test_inputs = set()
        for inp in initial_test_inputs:
            test_inputs.add(
                AlhazenInput.from_str(grammar_, inp)
            )

        self.assertEqual(1, len(test_inputs))


if __name__ == "__main__":
    unittest.main()
