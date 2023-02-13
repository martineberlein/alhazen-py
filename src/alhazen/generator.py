import time
import copy
import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas

from fuzzingbook.Parser import EarleyParser
from fuzzingbook.GrammarFuzzer import (
    tree_to_string,
)
from fuzzingbook.Grammars import Grammar, is_valid_grammar

from alhazen.Activity1_1_FeatureExtraction import (
    ExistenceFeature,
    NumericInterpretation,
)
from alhazen.helper import OracleResult, CALC_GRAMMAR
from alhazen.Activity3_RequirementExtraction import InputSpecification, Requirement
from alhazen.Activity1_1_FeatureExtraction import get_all_features, collect_features

from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.derivation_tree import DerivationTree


def best_trees(forest, spec):
    samples = [tree_to_string(tree) for tree in forest]
    fulfilled_fractions = []
    for sample in samples:
        gen_features = collect_features([sample], CALC_GRAMMAR)

        # calculate percentage of fulfilled requirements (used to rank the sample)
        fulfilled_count = 0
        total_count = len(spec.requirements)
        for req in spec.requirements:
            # for now, interpret requirement(exists(<...>) <= number) as false and requirement(exists(<...>) > number) as true
            if isinstance(req.feature, ExistenceFeature):
                expected = 1.0 if req.quant == ">" or req.quant == ">=" else 0.0
                actual = gen_features[req.feature.name][0]
                if actual == expected:
                    fulfilled_count += 1
                else:
                    pass
                    # print(f'{req.feature} expected: {expected}, actual:{actual}')
            elif isinstance(req.feature, NumericInterpretation):
                expected_value = float(req.value)
                actual_value = gen_features[req.feature.name][0]
                fulfilled = False
                if req.quant == "<":
                    fulfilled = actual_value < expected_value
                elif req.quant == "<=":
                    fulfilled = actual_value <= expected_value
                elif req.quant == ">":
                    fulfilled = actual_value > expected_value
                elif req.quant == ">=":
                    fulfilled = actual_value >= expected_value

                if fulfilled:
                    fulfilled_count += 1
                else:
                    pass
                    # print(f'{req.feature} expected: {expected_value}, actual:{actual_value}')
        fulfilled_fractions.append(fulfilled_count / total_count)
        # print(f'Fraction of fulfilled requirements: {fulfilled_count / total_count}')
    max_frac = max(fulfilled_fractions)
    best_chosen = []
    if max_frac == 1.0:
        return True, forest[fulfilled_fractions.index(1.0)]

    for i, t in enumerate(forest):
        if fulfilled_fractions[i] == max_frac:
            best_chosen.append(t)
    return False, best_chosen


# well, not perfect and probably not very robust. but it works :)
def generate_samples_advanced(
    grammar: Grammar, new_input_specifications: List[InputSpecification], timeout: int
) -> List[str]:
    # if there are no input specifications: generate some random samples
    if len(new_input_specifications) == 0:
        fuzzer = GrammarFuzzer(grammar)
        samples = [fuzzer.fuzz() for _ in range(100)]
        return samples

    final_samples = []
    each_spec_timeout = timeout / len(new_input_specifications)

    rhs_nonterminals = (
        grammar.keys()
    )  # list(chain(*[non-terminals(expansion) for expansion in grammar[rule]]))

    fuzzer = GrammarFuzzer(grammar)

    for spec in new_input_specifications:
        done = False
        starttime = time.time()
        best_chosen = [fuzzer.fuzz_tree() for _ in range(100)]
        done, best_chosen = best_trees(best_chosen, spec)
        if done:
            final_samples.append(tree_to_string(best_chosen))
        while not done and time.time() - starttime < each_spec_timeout:
            # split in prefix, postfix and try to reach targets
            for tree in best_chosen:
                prefix_len = random.randint(1, 3)
                curr = tree
                valid = True
                for i in range(prefix_len):
                    nt, children = curr
                    poss_desc_idxs = []
                    for c_idx, c in enumerate(children):
                        s, _ = c
                        possible_descend = s in rhs_nonterminals
                        if possible_descend:
                            poss_desc_idxs.append(c_idx)
                    if len(poss_desc_idxs) < 1:
                        valid = False
                        break
                    desc = random.randint(0, len(poss_desc_idxs) - 1)
                    curr = children[poss_desc_idxs[desc]]
                if valid:
                    nt, _ = curr
                    for req in spec.requirements:
                        if (
                            isinstance(req.feature, NumericInterpretation)
                            and nt == req.feature.key
                        ):
                            # hacky: generate a derivation tree for this numeric interpretation
                            hacky_grammar = copy.deepcopy(grammar)
                            hacky_grammar["<start>"] = [nt]
                            parser = EarleyParser(hacky_grammar)
                            try:
                                test = parser.parse(req.value)
                                x = list(test)[0]
                                _, s = x
                                # print(str(s[0]))
                                # replace curr in tree with this new tree
                                curr = s[0]
                            except SyntaxError:
                                pass
            done, best_chosen = best_trees(best_chosen, spec)
            if done:
                final_samples.append(tree_to_string(best_chosen))
        if not done:
            final_samples.extend([tree_to_string(t) for t in best_chosen])

    return final_samples


def generate_samples_random(grammar, num):
    f = GrammarFuzzer(grammar, max_nonterminals=50, log=False)
    data = []
    for _ in range(num):
        new_input = DerivationTree.from_parse_tree(f.fuzz_tree())
        assert isinstance(new_input, DerivationTree)
        data.append(new_input)

    return data


class Generator(ABC):
    def __init__(self, grammar, timeout: int = 10):
        assert is_valid_grammar(grammar)
        self.grammar: Grammar = grammar
        self.timeout: int = timeout

    @abstractmethod
    def generate(self, **kwargs) -> DerivationTree:
        raise NotImplementedError


class SimpleGenerator(Generator):
    """
    Simple Generator to produce random inputs as Derivation trees.
    """

    def __init__(self, grammar: Grammar):
        super().__init__(grammar)

    def generate(self, **kwargs) -> DerivationTree:
        f = GrammarFuzzer(self.grammar, max_nonterminals=50, log=False)
        new_input = DerivationTree.from_parse_tree(f.fuzz_tree())
        assert isinstance(new_input, DerivationTree)

        return new_input


class AdvancedGenerator(Generator):
    """
    Generator to produce new inputs according to a given set of input specifications.
    """

    def __init__(self, grammar: Grammar, timeout: int = 10):
        super().__init__(grammar, timeout=timeout)

    @staticmethod
    def validate_requirement(
        sample_features: pandas.DataFrame, requirement: Requirement
    ) -> bool:
        if isinstance(requirement.feature, ExistenceFeature):
            expected = (
                1.0 if requirement.quant == ">" or requirement.quant == ">=" else 0.0
            )
            actual = sample_features[requirement.feature.name][0]
            if actual == expected:
                return True
            return False

        elif isinstance(requirement.feature, NumericInterpretation):
            expected = float(requirement.value)
            actual_value = sample_features[requirement.feature.name][0]

            match requirement.quant:
                case "<":
                    return actual_value < expected
                case "<":
                    return actual_value <= expected
                case ">":
                    return actual_value > expected
                case ">=":
                    return actual_value >= expected

    def validate(
        self, sample: DerivationTree, specification: InputSpecification
    ) -> Tuple[bool, int]:
        """
        Checks whether an input fulfills a given input specification. The result is the number of unfulfilled
        requirements. Thus, 0 corresponds to a perfectly fulfilled input.
        Args:
            sample: input sample to be checked
            specification: an input specification with requirements for a given input

        Returns:
            Tuple[bool, str]
        """
        assert isinstance(sample, DerivationTree)
        features = collect_features({sample}, self.grammar)

        count = len(specification.requirements)
        for requirement in specification.requirements:
            if self.validate_requirement(features, requirement=requirement):
                count -= 1

        return True if count == 0 else False, count

    def generate(self, input_specification: InputSpecification) -> DerivationTree:
        generator = SimpleGenerator(self.grammar)
        if not input_specification:
            return generator.generate()

        samples = [generator.generate() for _ in range(100)]
        queue = []
        for sample in samples:
            validated, unfulfilled = self.validate(sample, input_specification)
            if validated:
                return sample
            queue.append((sample, unfulfilled))


class ISLAGenerator(Generator):
    def generate(self, **kwargs) -> DerivationTree:
        raise NotImplementedError
