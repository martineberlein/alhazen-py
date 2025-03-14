import math

from dbg.data.oracle import OracleResult
from alhazen._data import AlhazenInput


def oracle(inp: AlhazenInput | str) -> OracleResult:
    try:
        eval(
            str(inp),
            {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan},
        )
        return OracleResult.PASSING
    except ValueError:
        return OracleResult.FAILING


grammar = {
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


initial_inputs = ["sqrt(-900)", "sin(-3)", "cos(10)", "tan(5)"]