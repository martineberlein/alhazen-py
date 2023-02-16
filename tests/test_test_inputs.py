import unittest
from typing import Tuple

from isla.derivation_tree import DerivationTree
from fuzzingbook.Parser import EarleyParser

from alhazen_formalizations.calculator import grammar_alhazen as grammar
from alhazen.helper import OracleResult
from alhazen.input import TestInput


class TestInputs(unittest.TestCase):

    def test_test_inputs(self):
        inputs = {'sqrt(-900)', 'cos(10)'}
        oracles = [OracleResult.BUG, OracleResult.NO_BUG]
        test_inputs = set()
        for inp, oracle in zip(inputs, oracles):
            test_input = TestInput(DerivationTree.from_parse_tree(
                next(EarleyParser(grammar).parse(inp))
            ))
            test_input.oracle = oracle
            test_inputs.add(test_input)

        self.assertEqual(inputs, set(map(lambda f: str(f.tree), test_inputs)))

        for inp, orc in zip(inputs, oracles):
            self.assertIn((inp, orc),
                          set(map(lambda x: (str(x.tree), x.oracle), test_inputs)))

        self.assertFalse(set(map(lambda f: str(f.tree), test_inputs)).__contains__('cos(X)'))


if __name__ == '__main__':
    unittest.main()
