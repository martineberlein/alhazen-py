import math
import logging

from fuzzingbook.Grammars import Grammar
from isla.derivation_tree import DerivationTree

from alhazen.oracle import OracleResult


def arith_eval(inp: DerivationTree) -> float:
    return eval(
        str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
    )


def prop_(inp: DerivationTree) -> bool:
    try:
        arith_eval(inp)
        return False
    except ValueError:
        return True


def prop(inp: DerivationTree):
    try:
        return OracleResult.BUG if prop_(inp) else OracleResult.NO_BUG
    except SyntaxError:
        logging.info(
            f"Input {str(inp)} is not a valid input of the program. You might want to rewrite your grammar!"
        )
        return OracleResult.UNDEF


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
