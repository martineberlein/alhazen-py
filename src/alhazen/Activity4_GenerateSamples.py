#!/usr/bin/env python
# coding: utf-8

# # Activity 4: Generate New Samples from the Decision Tree and the learned Features

# <div class="alert alert-success alertsuccess">
# [Task] Implement the function <i>generate_samples(grammar, new_input_specifications, timeout)</i>, that generates a set of new inputs to refine or refute the hypothesis of the decision tree.
# </div>

# Please implement a _Grammar-Based Input Generator_ that generates new input samples from a List of `Input Specifications`. The Input Specifications are extracted from the decision tree boundaries in the previous Activity 3: _RequirementExtraction_. A Input Specification consists of **1 to n** many predicates or requirements (e.g. feature '>=' value, or 'num(term) <= 13'). Your task is to generate a new input for each InputSpecification. The new input must fulfill all the given requirements of an IputSpecification.

# <div class="alert alert-info">
# [Info]: For furter details, please refer to <b>Section 4.4 and 4.5</b> of the <a href="https://publications.cispa.saarland/3107/7/fse2020-alhazen.pdf">paper</a> and the Chapter <b><a href="https://www.fuzzingbook.org/html/GrammarFuzzer.html">Efficient Grammar Fuzzing</a></b> in the fuzzingbook.
# </div>

# ```python
#
# def generate_samples(grammar: Grammar,
#                      new_input_specifications: List[InputSpecification],
#                      timeout: int) -> List[str]
#
# ```

# **INPUT**:
# the function requires the following input parameter:
# - grammar: the grammar this is used to produce new inputs (e.g. the CALCULATOR-Grammar)
# - new_input_specification: a List of new inputs specifications (List\[InputSpecification\])
# - timeout: a max time budget. Return the generated inputs when the timebudget is exeeded.

# **OUTPUT**: the function should return a list of new inputs that are specified by the given input specifications.

# <div class="alert alert-info">
# [Hint]: You can implement the functionality described in the paper or develop your own method of generating new inputs that fulfill the given input specification. (For instance, you can generate inputs with the grammar and check whether one of the inputs meets one of the input specifications. However, this may not be very efficient and may require a lot of time.)
# </div>

# <div class="alert alert-danger" role="alert">
# The classes Inputspecifications and Requirements require the functionallity of the FeatureExtration. You might want to finish the feature-extraction task first.
# </div>

# In[ ]:


from typing import List

from IPython.core.display_functions import display
from fuzzingbook.Grammars import Grammar

from alhazen.helper import OracleResult, CALC_GRAMMAR, START_SYMBOL
from alhazen.Activity3_RequirementExtraction import InputSpecification, Requirement


# In[ ]:


def generate_samples(
    grammar: Grammar, new_input_specifications: List[InputSpecification], timeout: int
) -> List[str]:
    # write your code here
    raise NotImplementedError("Func. generate samples: Function not implemented.")


# Possible solution for the function `generate_samples`:

# In[4]:


import time
import copy
from copy import deepcopy
import random
from typing import List
from itertools import chain

from fuzzingbook.Parser import EarleyParser
from fuzzingbook.GrammarFuzzer import (
    all_terminals,
    Grammar,
    tree_to_string,
)
from fuzzingbook.Grammars import Grammar, nonterminals, opts, is_valid_grammar
from fuzzingbook.Grammars import reachable_nonterminals, unreachable_nonterminals

from alhazen.Activity3_RequirementExtraction import (
    InputSpecification,
    Requirement,
    get_all_input_specifications,
)
from alhazen.Activity1_1_FeatureExtraction import (
    Feature,
    ExistenceFeature,
    NumericInterpretation,
)
from alhazen.helper import OracleResult, CALC_GRAMMAR, START_SYMBOL
from alhazen.Activity3_RequirementExtraction import InputSpecification, Requirement
from alhazen.Activity1_1_FeatureExtraction import get_all_features, collect_features


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
    )  # list(chain(*[nonterminals(expansion) for expansion in grammar[rule]]))

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


# ### Other interesting generator functions:

# In[2]:


from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.derivation_tree import DerivationTree


def generate_samples_random(grammar, new_input_specifications, num):
    f = GrammarFuzzer(grammar, max_nonterminals=50, log=False)
    data = []
    for _ in range(num):
        #new_input = f.fuzz()

        new_input = DerivationTree.from_parse_tree(f.fuzz_tree())
        assert isinstance(new_input, DerivationTree)
        data.append(new_input)

    return data


# In[8]:


from alhazen.helper import OracleResult, CALC_GRAMMAR, START_SYMBOL

# some tests for debugging
exsqrt = ExistenceFeature("exists(<function>@0)", "<function>", "sqrt")
exdigit = ExistenceFeature("exists(<digit>)", "<digit>", "<digit>")

reqDigit = Requirement(exdigit, ">", "0.5")
fbdDigit = Requirement(exdigit, "<=", "0.5")

req0 = Requirement(exsqrt, ">", "-6.0")
testspec0 = InputSpecification([req0, reqDigit])
req1 = Requirement(exsqrt, "<=", "-6.0")
testspec1 = InputSpecification([req1, fbdDigit])

numterm = NumericInterpretation("num(<term>)", "<term>")
req2 = Requirement(numterm, "<", "-31.0")
testspec2 = InputSpecification([req2, req0, reqDigit])

if __name__ == "__main__":
    print("--generating samples--")
    # samples = generate_samples(CALC_GRAMMAR, [testspec0, testspec1], 10)
    samples = generate_samples_advanced(CALC_GRAMMAR, [testspec2], 10)
    display(samples)


# In[ ]:
