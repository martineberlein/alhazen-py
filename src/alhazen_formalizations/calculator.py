import math

from fuzzingbook.Grammars import Grammar
from fuzzingbook.Parser import DerivationTree


def arith_eval(inp: DerivationTree) -> float:
    return eval(
        str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    )


def prop(inp: DerivationTree) -> bool:
    try:
        arith_eval(inp)
        return False
    except ValueError:
        return True


grammar: Grammar = {
    "<start>": ["<function>(<term>)"],
    "<function>": ["sqrt", "tan", "cos", "sin"],
    "<term>": ["-<value>", "<value>"],
    "<value>": ["<integer>.<integer>", "<integer>"],
    "<integer>": ["<digit><integer>", "<digit>"],
    "<digit>": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}


grammar_alhazen: Grammar = {
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

START_SYMBOL = "<start>"

initial_inputs = ["cos(12)", "sqrt(-900)"]
