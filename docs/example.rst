Examples
========
How to use *Alhazen*
--------------------
To illustrate *Alhazen*’s capabilities, we start with a quick motivating example.
First, let us introduce our program under test: The Calculator.

..  code-block:: python

    import math

    def arith_eval(inp) -> float:
        return eval(
            str(inp), {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan}
        )

This infamous program accepts arithmetic equations, trigonometric functions and allows us to calculate the square root. To help us determine faulty behavior, i.e., a crash, we implement an evaluation function

..  code-block:: python

    from alhazen.oracle import OracleResult

    def prop(inp: str) -> bool:
        try:
            arith_eval(inp)
            return OracleResult.NO_BUG
        except ValueError:
            return OracleResult.BUG
        return OracleResult.UNDEF

that takes an input file and returns whether a bug occurred during the evaluation of the mathematical equations (BUG=True, NO_BUG=False). We can now test the calculator with some sample inputs:

..  code-block:: python

    inputs = ['cos(10)', 'sqrt(28367)', 'tan(-12)', 'sqrt(-3)']
    print([(x, prop(x)) for x in inputs])

The output looks like this:

..  code-block:: python

    [('cos(10)', OracleResult.NO_BUG),
     ('sqrt(28367)', OracleResult.NO_BUG),
     ('tan(-12)', OracleResult.NO_BUG),
     ('sqrt(-3)', OracleResult.BUG)]

We see that sqrt(-3) results in the failure of our calculator program. We can now use **Alhazen** to learn the root causes of the program's failure.

First, we need to define the input format of the calculator with a grammar:

..  code-block:: python

    import string

    grammar = {
        "<start>": ["<arith_expr>"],
        "<arith_expr>": ["<function>(<number>)"],
        "<function>": ["sqrt", "sin", "cos", "tan"],
        "<number>": ["<maybe_minus><onenine><maybe_digits>"],
        "<maybe_minus>": ["", "-"],
        "<onenine>": [str(num) for num in range(1, 10)],
        "<digit>": list(string.digits),
        "<maybe_digits>": ["", "<digits>"],
        "<digits>": ["<digit>", "<digit><digits>"],
    }

Then, we can call **Alhazen** with the grammar, some sample inputs, and the evaluation function (program under test).

..  code-block:: python

    from alhazen import Alhazen

    alhazen = Alhazen(
        initial_inputs=inputs,
        grammar=grammar,
        evaluation_function=prop,
    )
    trees = alhazen.run()

By default, **Alhazen** will do 10 iterations of its refinement process. Finally, **Alhazen** returns the learned decision tree that describes the failure-inducing inputs.

For our calculator, the learned decision tree looks something like this:

.. image:: ../img/DecisionTree.png

We see that the failure occurs whenever we use the sqrt(x) function and the number x has a negative sign!

