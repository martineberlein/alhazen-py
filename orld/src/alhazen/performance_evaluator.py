import time
from typing import Callable, List, Dict, Set
from pandas import DataFrame

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from fuzzingbook.Grammars import Grammar
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.derivation_tree import DerivationTree

from alhazen import Alhazen
from alhazen.oracle import OracleResult
from alhazen.input import Input
from alhazen.feature_collector import Collector
from alhazen.features import FeatureWrapper, STANDARD_FEATURES
from alhazen.requirementExtractionDT.treetools import grouped_rules


class EvaluationDataSet:
    def __init__(self, test_inputs: Set[Input]):
        self.test_data: Set[Input] = test_inputs

    def __len__(self):
        return len(self.test_data)

    @property
    def bug_samples(self):
        bug_samples = set(
            [inp for inp in self.test_data if inp.oracle == OracleResult.BUG]
        )
        return bug_samples


class Evaluator:
    """
    The evaluation class that runs and executes all experiments. Main method: run().
    """

    def __init__(
        self,
        name: str,
        tools: List[Callable[[int, int], Alhazen]],
        job_names: List[str],
        grammar: Grammar,
        features: Set[FeatureWrapper] = STANDARD_FEATURES,
        prop: Callable = None,
        repetitions: int = 30,
        timeout: int = 60 * 60,
        evaluation_data_set: Set[Input] = None,
    ):
        self.name = name
        self.tools = tools
        self.job_names = job_names
        self.repetitions = repetitions
        self.timeout = timeout
        self.grammar = grammar
        self.prop = prop
        self.features = features

        if evaluation_data_set is None:
            self.evaluation_data_set = self.generate_evaluation_data_set()

    def run(self):
        experiment_results = self.execute_experiments(self.tools, self.job_names)
        evaluation_results = self.evaluate_experiments(
            experiment_results, self.evaluation_data_set
        )
        return evaluation_results

    def evaluate_experiments(
        self, experiment_results: List[Dict], evaluation_data_set: DataFrame
    ):
        p = evaluation_data_set
        evaluation_data_set = evaluation_data_set.fillna(0).drop(["input"], axis=1)
        eval_iteration = self.repetitions - 1
        for experiment in experiment_results:
            model = experiment["models"][eval_iteration]

            predictions = model.predict(evaluation_data_set.drop(["oracle"], axis=1))

            accuracy = accuracy_score(
                evaluation_data_set["oracle"].astype(str), predictions, normalize=True
            )
            accuracy = round(accuracy * 100, 3)

            precision = precision_score(
                evaluation_data_set["oracle"].astype(str),
                predictions,
                pos_label="BUG",
                average="binary",
            )
            precision = round(precision * 100, 3)

            recall = recall_score(
                evaluation_data_set["oracle"].astype(str),
                predictions,
                pos_label="BUG",
                average="binary",
            )
            recall = round(recall * 100, 3)

            f1 = f1_score(
                evaluation_data_set["oracle"].astype(str),
                predictions,
                pos_label="BUG",
                average="binary",
            )

            experiment["accuracy"] = accuracy
            experiment["precision"] = precision
            experiment["recall"] = recall
            experiment["f1-score"] = f1

            collector: Collector = Collector(self.grammar, self.features)
            all_features = collector.get_all_features()
            feature_names = [f.name for f in all_features]

            # print(grouped_rules(model, feature_names=feature_names))

            for prediction, oracle, inp in zip(
                predictions, evaluation_data_set["oracle"].astype(str), p["input"]
            ):
                if prediction != oracle:
                    print(prediction, oracle, inp)

        return experiment_results

    def execute_experiments(
        self, tools: List[Callable[[int, int], Alhazen]], job_names: List[str]
    ) -> List[Dict]:
        experiment_results: List = []
        for job_name, tool in zip(job_names, tools):
            tool = tool(self.timeout, self.repetitions)
            experiment_results.append(self.learn_model(job_name, tool))

        return experiment_results

    @staticmethod
    def learn_model(
        job_name: str,
        tool: Alhazen,
        start_time=time.time(),
    ):
        if isinstance(tool, Alhazen):
            models = tool.run()
            data = {
                "name": job_name,
                "models": models,
                "time": time.time() - start_time,
            }
            return data
        else:
            raise AssertionError

    def generate_evaluation_data_set(self) -> DataFrame:
        test_inputs = set()
        g = GrammarFuzzer(self.grammar)
        collector = Collector(self.grammar, self.features)
        for _ in range(1000):
            inp = Input(DerivationTree.from_parse_tree(g.fuzz_tree()))
            test_inputs.add(inp)

        evaluation_data = []
        for test_input in test_inputs:
            test_input.oracle = self.prop(test_input)
            if test_input.oracle != OracleResult.UNDEF:
                test_input.features = collector.collect_features(test_input)
                learning_data = test_input.features
                learning_data["input"] = str(test_input)
                learning_data["oracle"] = test_input.oracle
                evaluation_data.append(learning_data)

        return DataFrame.from_records(evaluation_data)
