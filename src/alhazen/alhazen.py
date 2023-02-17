#!/usr/bin/env python
import logging
import sys
from typing import List, Callable, Set, Union
from pandas import DataFrame, concat

from fuzzingbook.Grammars import Grammar, is_valid_grammar
from fuzzingbook.Parser import EarleyParser
from sklearn.tree import DecisionTreeClassifier
from isla.derivation_tree import DerivationTree

from alhazen.features import (
    extract_existence,
    extract_numeric,
    collect_features,
    Feature,
)

from alhazen.input import Input
from alhazen.learner import train_tree, Learner, DecisionTreeLearner
from alhazen.generator import SimpleGenerator, Generator
from alhazen.input_specifications import get_all_input_specifications
from alhazen.oracle import OracleResult

GENERATOR_TIMEOUT = 10  # timeout in seconds
MAX_ITERATION = 20


class Alhazen:
    def __init__(
        self,
        initial_inputs: List[str],
        grammar: Grammar,
        evaluation_function: Callable,
        max_iter: int = 10,
        generator_timeout: int = 10,
        generator: Union[Generator | None] = None,
        learner: Union[Learner | None] = None,
    ):
        self._initial_inputs: List[str] = initial_inputs
        self._grammar: grammar = grammar
        self._prop: Callable[[DerivationTree], bool] = evaluation_function
        self._max_iter: int = max_iter
        self._previous_samples: Set[Input] = set()
        self._data = None
        self._trees: List[DecisionTreeClassifier] = []
        self._generator_timeout: int = generator_timeout

        assert is_valid_grammar(self._grammar)

        if generator is None:
            self._generator: Generator = SimpleGenerator(self._grammar)
        else:
            assert isinstance(generator, Generator)
            self._generator: Generator = generator

        if learner is None:
            self._learner: Learner = DecisionTreeLearner()
        else:
            assert isinstance(learner, Learner)
            self._learner: Learner = learner

        self._setup()

    def _setup(self):
        for inp in self._initial_inputs:
            try:
                self._previous_samples.add(
                    Input(
                        DerivationTree.from_parse_tree(
                            next(EarleyParser(self._grammar).parse(inp))
                        )
                    )
                )
            except SyntaxError:
                logging.error(
                    "Alhazen-py: Could not parse initial inputs with given grammar!"
                )
                sys.exit(-1)

        self._all_features: List[Feature] = extract_existence(
            self._grammar
        ) + extract_numeric(self._grammar)
        self._feature_names = [f.name for f in self._all_features]

    def _add_new_data(self, test_inputs: Set[Input]):
        data = []
        for inp in test_inputs:
            if inp.oracle != OracleResult.UNDEF:
                learning_data = inp.features  # .drop(["sample"], axis=1)
                learning_data["oracle"] = inp.oracle
                data.append(learning_data)

        new_data = DataFrame.from_records(data)

        if 0 != len(new_data):
            if self._data is None:
                self._data = new_data
            else:
                self._data = concat([self._data, new_data], sort=False)

    def _finalize(self):
        return self._trees

    def run(self) -> List:
        for iteration in range(self._max_iter):
            logging.info(f"Starting Iteration: " + str(iteration))
            self._loop(self._previous_samples)

        return self._finalize()

    def _loop(self, test_inputs: Set[Input]):
        # obtain labels, execute samples (Initial Step, Activity 5)
        for inp in test_inputs:
            inp.oracle = self._prop(inp)

        # collect features from the new samples (Activity 1)
        for inp in test_inputs:
            inp.features = collect_features(inp, self._all_features)

        # combine the new data with the already existing data
        self._add_new_data(test_inputs)

        # train a tree (Activity 2)
        dec_tree = train_tree(self._data)
        self._trees.append(dec_tree)

        # extract new requirements from the tree (Activity 3)
        new_input_specifications = get_all_input_specifications(
            dec_tree,
            self._all_features,
            self._feature_names,
            self._data.drop(["oracle"], axis=1),
        )

        # generate new inputs according to the new input specifications
        # (Activity 4)
        new_samples = self._generate_inputs(new_input_specifications)

        self._previous_samples = new_samples

    def _generate_inputs(self, input_specifications) -> Set[Input]:
        inputs = set()
        for specification in input_specifications:
            inputs.add(self._generator.generate(input_specification=specification))

        return inputs

    def _execute_input_files(self, inputs: Set[Input]) -> DataFrame:
        logging.info("Executing input files")

        exec_oracle = []
        for inp in inputs:
            try:
                result = self._prop(inp.tree)  # TODO What about UNDEF?
                exec_oracle.append(
                    {
                        # "sample_id": id.hex,
                        # "sample": sample,
                        # "subject": SUBJECT,
                        "oracle": result
                    }
                )
            except SyntaxError:
                logging.info(
                    f"Input {str(inp)} is not a valid input of the program. You might want to rewrite your grammar!"
                )

        return DataFrame.from_records(exec_oracle)
