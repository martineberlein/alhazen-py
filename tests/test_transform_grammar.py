import unittest

from fuzzingbook.Grammars import EXPR_GRAMMAR
from alhazen.grammar_transformation import transform_grammar


class TestGrammarTransformation(unittest.TestCase):
    def test_grammar_transformation(self):
        transformed_grammar = transform_grammar("1 + 2", EXPR_GRAMMAR)

        self.assertTrue(
            all(map(lambda x: x in transformed_grammar["<expr>"], ["1 + 2", "2"]))
        )
        self.assertTrue(
            all(map(lambda x: x in transformed_grammar["<term>"], ["1", "2"]))
        )
        self.assertTrue(
            all(map(lambda x: x in transformed_grammar["<factor>"], ["1", "2"]))
        )
        self.assertTrue(
            all(map(lambda x: x in transformed_grammar["<integer>"], ["1", "2"]))
        )
        self.assertTrue(
            all(map(lambda x: x in transformed_grammar["<digit>"], ["1", "2"]))
        )


if __name__ == "__main__":
    unittest.main()
