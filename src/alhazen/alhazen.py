import logging
import sys
from typing import List, Callable, Set, Union
from pandas import DataFrame, concat

from fuzzingbook.Grammars import Grammar, is_valid_grammar
from fuzzingbook.Parser import EarleyParser
from isla.derivation_tree import DerivationTree

from alhazen.input import Input
from alhazen.learner import Learner, DecisionTreeLearner
from alhazen.generator import SimpleGenerator, Generator
from alhazen.oracle import OracleResult
from alhazen.features import FeatureWrapper, STANDARD_FEATURES
from alhazen.feature_collector import Collector
from alhazen.helper import show_tree

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
        features: Set[FeatureWrapper] = STANDARD_FEATURES,
    ):
        self._initial_inputs: List[str] = initial_inputs
        self._grammar: grammar = grammar
        self._prop: Callable[[Input], OracleResult] = evaluation_function
        self._max_iter: int = max_iter
        self._previous_samples: Set[Input] = set()
        self._data = None
        self._models: List = []
        self._generator_timeout: int = generator_timeout
        self._syntactic_features: Set[FeatureWrapper] = features

        assert is_valid_grammar(self._grammar)

        # Syntactic Feature Collection
        self._collector: Collector = Collector(self._grammar, self._syntactic_features)
        self._all_features = self._collector.get_all_features()
        self._feature_names = [f.name for f in self._all_features]

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

        return self._data

    def _finalize(self):
        return self._models

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
            inp.features = self._collector.collect_features(inp)

        # combine the new data with the already existing data
        learning_data = self._add_new_data(test_inputs)

        # train a tree (Activity 2)
        model = self._learner.train(learning_data)
        self._models.append(model)

        new_input_specifications = self._learner.get_input_specifications(
            model,
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

    def show_model(self):
        return show_tree(self._models[-1], self._all_features)
