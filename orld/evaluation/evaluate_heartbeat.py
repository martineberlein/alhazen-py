import logging

from pandas import DataFrame

from alhazen import Alhazen
from alhazen_formalizations.heartbeat import (
    prop,
    grammar,
    initial_inputs,
)
from alhazen.performance_evaluator import Evaluator
from alhazen.learner import RandomForestLearner


def alhazen_decision_tree(timeout, max_iterations):
    return Alhazen(
        grammar=grammar,
        initial_inputs=initial_inputs,
        evaluation_function=prop,
        generator_timeout=timeout,
        max_iter=max_iterations,
    )


def alhazen_random_forest(timeout, max_iterations):
    return Alhazen(
        grammar=grammar,
        initial_inputs=initial_inputs,
        evaluation_function=prop,
        generator_timeout=timeout,
        max_iter=max_iterations,
        learner=RandomForestLearner()
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")

    evaluator = Evaluator(
        "Heartbleed",
        timeout=60 * 60,
        repetitions=20,
        tools=[alhazen_decision_tree, alhazen_random_forest],
        job_names=["Alhazen", "Alhazen RandomForest"],
        grammar=grammar,
        prop=prop,
    )
    evaluation_results: DataFrame = evaluator.run()

    for result in evaluation_results:
        print(
            f"Name: {result['name']}",
            f", Time: {result['time']}",
            f", Accuracy: {result['accuracy']}",
            f", Precision: {result['precision']}",
            f", Recall: {result['recall']}",
        )
