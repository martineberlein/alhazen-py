import math

from alhazen.alhazen import Alhazen

from alhazen.data.input import AlhazenInput, OracleResult
from alhazen.data.grammar import Grammar
from alhazen.data.feature_collector import Collector

from alhazen.learner.learner import DecisionTreeLearner, show_tree

from dbg.runner.runner import ExecutionHandler, SingleExecutionHandler


grammar_alhazen = {
    "<start>": ["<arith_expr>"],
    "<arith_expr>": ["<function>(<number>)"],
    "<function>": ["sqrt", "sin", "cos", "tan"],
    "<number>": ["<maybe_minus><onenine><maybe_digits><maybe_frac>"],
    "<maybe_minus>": ["", "-"],
    "<onenine>": [str(num) for num in range(1, 10)],
    "<digit>": [str(num) for num in range(0, 10)],
    "<maybe_digits>": ["", "<digits>"],
    "<digits>": ["<digit>", "<digit><digits>"],
    "<maybe_frac>": ["", ".<digits>"],
}


def arith_eval(inp) -> float:
    return eval(
        str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    )


def oracle(inp: AlhazenInput) -> OracleResult:
    try:
        arith_eval(str(inp))
        return OracleResult.PASSING
    except ValueError:
        return OracleResult.FAILING


if __name__ == "__main__":
    grammar = Grammar(grammar_alhazen)
    feature_collector = Collector(grammar=Grammar(grammar_alhazen))

    initial_inputs = [
        "cos(12)",
        "sqrt(-900)",
        "sin(-58)",
        "sqrt(-1)",
        "tan(1241)",
        "sqrt(123)",
    ]

    initial_inputs = {grammar.parse(inp) for inp in initial_inputs}
    print(initial_inputs)
    for inp in initial_inputs:
        inp.features = feature_collector.collect_features(inp)
        print(inp.features)

    runner = SingleExecutionHandler(oracle=oracle)
    runner.label(initial_inputs)
    for inp in initial_inputs:
        print(inp, inp.oracle)

    learner = DecisionTreeLearner()
    clf = learner.train(initial_inputs)
    show_tree(clf, feature_names=None)

    all_features = feature_collector.all_features
    feature_names = [f.name for f in all_features]

    new_input_specifications = learner.get_input_specifications(clf, all_features=all_features, feature_names=feature_names, data=learner.data)
    for spec in new_input_specifications:
        print(spec)


