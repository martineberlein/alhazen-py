from alhazen.alhazen import Alhazen

from alhazen.data.input import AlhazenInput
from alhazen.data.grammar import Grammar

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


if __name__ == "__main__":
    grammar = Grammar(grammar_alhazen)

    initial_inputs = [
        "cos(12)",
        "sqrt(-900)",
    ]

    initial_inputs = [grammar.parse(inp) for inp in initial_inputs]
    print(initial_inputs)


