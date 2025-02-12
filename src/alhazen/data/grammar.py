
from fuzzingbook.Grammars import Grammar as FuzzingBookGrammar
from fuzzingbook.Parser import EarleyParser
from fuzzingbook.GrammarFuzzer import GrammarFuzzer

from dbg.data.input import Input
from dbg.data.grammar import AbstractGrammar


class Grammar(AbstractGrammar):

    def __init__(self, grammar: FuzzingBookGrammar, **kwargs):
        super().__init__(grammar, **kwargs)
        self.parser = EarleyParser(self.grammar)
        self.fuzzer = GrammarFuzzer(self.grammar)

    def parse(self, input_string: str) -> Input | None:
        tree = self.parser.parse(input_string)
        if tree:
            return Input(tree)
        return None

    def fuzz(self) -> Input:
        tree = self.fuzzer.fuzz_tree()
        return Input(tree)

    def __str__(self):
        pass

    def get_nonterminals(self):
        pass

    def get_rules(self):
        pass
