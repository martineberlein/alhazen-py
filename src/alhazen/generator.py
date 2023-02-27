import time
import copy
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from fuzzingbook.Parser import EarleyParser
from fuzzingbook.GrammarFuzzer import (
    tree_to_string,
)
from fuzzingbook.Grammars import Grammar, is_valid_grammar

from alhazen.features import (
    Feature,
    ExistenceFeature,
    NumericInterpretation,
    LengthFeature,
    extract_numeric,
    extract_existence,
)

from alhazen.input_specifications import InputSpecification, Requirement
from alhazen.feature_collector import Collector
from alhazen.input import Input

from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.derivation_tree import DerivationTree


def best_trees(forest, spec, grammar):
    samples = [tree for tree in forest]
    fulfilled_fractions = []

    collector = Collector(grammar)
    for sample in samples:
        gen_features = collector.collect_features(
            Input(tree=DerivationTree.from_parse_tree(sample))
        )

        # calculate percentage of fulfilled requirements (used to rank the sample)
        fulfilled_count = 0
        total_count = len(spec.requirements)
        for req in spec.requirements:
            # for now, interpret requirement(exists(<...>) <= number) as false and requirement(exists(<...>) > number) as true
            if isinstance(req.feature, ExistenceFeature):
                expected = 1.0 if req.quant == ">" or req.quant == ">=" else 0.0
                actual = gen_features[req.feature.name]
                if actual == expected:
                    fulfilled_count += 1
                else:
                    pass
                    # print(f'{req.feature} expected: {expected}, actual:{actual}')
            elif isinstance(req.feature, NumericInterpretation):
                expected_value = float(req.value)
                actual_value = gen_features[req.feature.name]
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
        done, best_chosen = best_trees(best_chosen, spec, grammar)
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
            done, best_chosen = best_trees(best_chosen, spec, grammar)
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
    def generate(self, **kwargs) -> Input:
        raise NotImplementedError


class SimpleGenerator(Generator):
    """
    Simple Generator to produce random inputs as Derivation trees.
    """

    def __init__(self, grammar: Grammar):
        super().__init__(grammar)

    def generate(self, **kwargs) -> Input:
        f = GrammarFuzzer(self.grammar, max_nonterminals=50, log=False)
        new_tree = DerivationTree.from_parse_tree(f.fuzz_tree())
        assert isinstance(new_tree, DerivationTree)

        return Input(tree=new_tree)


class AdvancedGenerator(Generator):
    """
    Generator to produce new inputs according to a given set of input specifications.
    """

    def __init__(
        self,
        grammar: Grammar,
        grammar_features: List[Feature] = None,
        timeout: int = 10,
    ):
        super().__init__(grammar, timeout=timeout)

        self._collector = Collector(grammar=grammar)

        if grammar_features is None:
            self._grammar_features = self._collector.get_all_features()
        self._grammar_features = grammar_features  # This can become problematic!

    @staticmethod
    def validate_requirement(input_features: Dict, requirement: Requirement) -> bool:
        if isinstance(requirement.feature, ExistenceFeature):
            expected = (
                1.0 if requirement.quant == ">" or requirement.quant == ">=" else 0.0
            )
            actual = input_features[requirement.feature.name]
            if actual == expected:
                return True
            return False

        elif isinstance(requirement.feature, NumericInterpretation):
            expected = float(requirement.value)
            actual_value = input_features[requirement.feature.name]

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
        self, test_input: Input, specification: InputSpecification
    ) -> Tuple[bool, int]:
        """
        Checks whether an input fulfills a given input specification. The result is the number of unfulfilled
        requirements. Thus, 0 corresponds to a perfectly fulfilled input.
        Args:
            test_input: input sample to be checked
            specification: an input specification with requirements for a given input

        Returns:
            Tuple[bool, str]
        """
        assert isinstance(test_input, Input)
        features = self._collector.collect_features(test_input)

        count = len(specification.requirements)
        for requirement in specification.requirements:
            if self.validate_requirement(features, requirement=requirement):
                count -= 1

        return True if count == 0 else False, count

    def generate(self, input_specification: InputSpecification) -> Input:
        generator = SimpleGenerator(self.grammar)
        if not input_specification:
            return generator.generate()

        sample_inputs: List[Input] = [generator.generate() for _ in range(100)]
        queue = []
        for inp in sample_inputs:
            validated, unfulfilled = self.validate(inp, input_specification)
            if validated:
                return inp
            queue.append((inp, unfulfilled))


class ISLAGenerator(Generator):

    def __init__(self, grammar: Grammar):
        super().__init__(grammar)

    @staticmethod
    def transform_constraints(input_specification: InputSpecification):
        constraints = []
        for idx, requirement in enumerate(input_specification.requirements):
            """
            We use the extended syntax of ISLA
            - 1D: 
                    - exists(<digit>)                       ???
                    - num(<number>) </>/<=/>= xyz           str.to.int(<number>) < 12
                    - len(<function>) </>/<=/>= xyz         str.len(<function>) > 3.5
            - 2D: 
                1. f.key is terminal:
                    - exists(<function> == sqrt)            <function> = "sqrt"
                    - exists(<maybe_minus>) == )            <maybe_minus> = ""
                2. f.key is nonterminal:
                    - exists(<function> == .<digit>)        ???
            """
            feature = requirement.feature
            constraint = ""
            if feature.rule == feature.key:
                # 1D Case
                if isinstance(feature, NumericInterpretation):
                    constraint = f"str.to.int({feature.rule}) {requirement.quant} {requirement.value}"
                if isinstance(feature, LengthFeature):
                    constraint = f"str.len({feature.rule}) {requirement.quant} {requirement.value}"
            else:
                if isinstance(feature, ExistenceFeature):
                    constraint = f'''{feature.rule} = "{feature.key}"'''

            constraints.append(constraint)

        p = " and ".join(constraints)

        return p

    def generate(self, input_specification: InputSpecification, **kwargs) -> Input:
        raise NotImplementedError
