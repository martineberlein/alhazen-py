import unittest

from isla.solver import ISLaSolver

from alhazen.input_specifications import Requirement, InputSpecification
from alhazen.generator import (
    SimpleGenerator,
    AdvancedGenerator,
    Generator,
    generate_samples_advanced,
    ISLAGenerator
)
from alhazen_formalizations.calculator import grammar
from alhazen.features import ExistenceFeature, NumericInterpretation


class TestGenerator(unittest.TestCase):
    def tests_simple_generator(self):
        generator = SimpleGenerator(grammar=grammar)
        assert isinstance(generator, Generator)
        inputs = [generator.generate() for _ in range(10)]
        self.assertEqual(len(inputs), 10)

    def test_advanced_generator(self):
        generator = AdvancedGenerator(grammar=grammar)
        assert isinstance(generator, Generator)
        req_sqrt = Requirement(
            ExistenceFeature("exists(<function>@0)", "<function>", "sqrt"), ">", "0.5"
        )
        req_term = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), ">", "-900.0"
        )
        input_specification = InputSpecification([req_sqrt, req_term])

        generator.generate(input_specification)

    def test_advanced_generator_function_old(self):
        """
        Old Function -> Deprecated
        """
        exist_sqrt = ExistenceFeature("exists(<function>@0)", "<function>", "sqrt")
        req_sqrt = Requirement(exist_sqrt, ">", "0.5")

        num_term = NumericInterpretation("num(<term>)", "<term>")
        req_term = Requirement(num_term, ">", "-31.0")

        test_spec1 = InputSpecification([req_sqrt, req_term])

        for _ in range(10):
            print(
                str(generate_samples_advanced(grammar, [test_spec1], 10))
            )  # TODO better test case and assertion

    def test_isla_input_specification_transformation(self):
        exist_sqrt = ExistenceFeature("exists(<function>@0)", "<function>", "sqrt")
        req_sqrt = Requirement(exist_sqrt, ">", "0.5")

        num_term = NumericInterpretation("num(<term>)", "<term>")
        req_term = Requirement(num_term, ">", "5.0")

        test_spec1 = InputSpecification([req_sqrt, req_term])
        generator = ISLAGenerator(grammar=grammar)
        constraint = generator.transform_constraints(test_spec1)
        self.assertEqual('''<function> = "sqrt" and str.to.int(<term>) > 5.0''', constraint)

    def test_isla_generator(self):
        from alhazen_formalizations.calculator import grammar_alhazen as grammar
        formula = '''
            str.to.int(<number>) < 12
            and
            <function> = "sqrt"
            and
            <maybe_frac> = ""
            '''
        #             forall <maybe_frac> c = ".<digits>" in start:
        #                 inside(c, start)

        #             inside(<maybe_frac>.<digits>, start)
        #             and
        solver = ISLaSolver(
            grammar=grammar,
            formula=formula
        )
        for _ in range(9):
            print(solver.solve())


if __name__ == "__main__":
    unittest.main()
