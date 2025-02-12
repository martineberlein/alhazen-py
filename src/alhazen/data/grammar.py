
from fuzzingbook.Grammars import Grammar as FuzzingBookGrammar
from fuzzingbook.Parser import EarleyParser
from fuzzingbook.GrammarFuzzer import GrammarFuzzer

from dbg.data.grammar import AbstractGrammar

from alhazen.data.input import AlhazenInput


class Grammar(AbstractGrammar):

    def __init__(self, grammar: FuzzingBookGrammar, **kwargs):
        super().__init__(grammar, **kwargs)
        self.parser = EarleyParser(self.grammar)
        self.fuzzer = GrammarFuzzer(self.grammar)

    def parse(self, input_string: str) -> AlhazenInput | None:
        for tree in self.parser.parse(input_string):
            return AlhazenInput(tree=tree)
        return None

    def fuzz(self) -> AlhazenInput:
        tree = self.fuzzer.fuzz_tree()
        return AlhazenInput(tree)

    def __str__(self):
        pass

    def get_nonterminals(self):
        pass

    def get_rules(self):
        pass

    def __iter__(self):
        return iter(self.grammar)

    def __getitem__(self, item):
        return self.grammar[item]
