import copy

from fuzzingbook.Grammars import Grammar, is_nonterminal, is_valid_grammar
from fuzzingbook.Parser import EarleyParser
from fuzzingbook.GrammarFuzzer import tree_to_string


def extend_grammar(derivation_tree, grammar):
    (node, children) = derivation_tree

    if is_nonterminal(node):
        assert node in grammar
        word = tree_to_string(derivation_tree)

        # Only add to grammar if not already existent
        if word not in grammar[node]:
            grammar[node].append(word)

    for child in children:
        extend_grammar(child, grammar)


def transform_grammar(sample: str, grammar: Grammar) -> Grammar:
    # copy of the grammar
    transformed_grammar = copy.deepcopy(grammar)

    # parse sample
    parser = EarleyParser(grammar)
    for derivation_tree in parser.parse(sample):
        extend_grammar(derivation_tree, transformed_grammar)

    assert is_valid_grammar(transformed_grammar)
    return transformed_grammar
