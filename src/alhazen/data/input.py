from typing import Optional
from fuzzingbook.GrammarFuzzer import tree_to_string, DerivationTree

from dbg.data.input import Input
from dbg.data.oracle import OracleResult


class AlhazenInput(Input):

    def __init__(self, tree: DerivationTree, oracle: Optional[OracleResult] = None):
        super().__init__(tree, oracle)
        self.hash = hash(tree_to_string(tree))

    def traverse(self):
        pass

    def __hash__(self) -> int:
        return self.hash

    @classmethod
    def from_str(cls, grammar, input_string, oracle: Optional[OracleResult] = None):
        pass

    def __repr__(self):
        return f"AlhazenInput('{tree_to_string(self.tree)}')"

    def __str__(self):
        return tree_to_string(self.tree)