import unittest

from alhazen.input_specifications import Requirement, InputSpecification
from alhazen.generator import (
    SimpleGenerator,
    AdvancedGenerator,
    Generator,
    generate_samples_advanced,
    ISLAGenerator
)
from alhazen.isla_helper import input_specification_to_isla_constraint
from alhazen_formalizations.calculator import grammar
from alhazen.features import ExistenceFeature, NumericInterpretation
from tests.test_helper import assertInputSpecification


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

        input = generator.generate(input_specification)

        assertInputSpecification(grammar, input, input_specification)

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
        constraint = input_specification_to_isla_constraint(test_spec1)
        self.assertEqual('''<function> = "sqrt" and str.to.int(<term>) > 5.0''', constraint)

    def test_isla_generator(self):
        isla_generator = ISLAGenerator(grammar)

        req_sqrt = Requirement(
            ExistenceFeature("exists(<function>@0)", "<function>", "sqrt"), ">", "0.5"
        )
        req_term_min = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), ">", "900.0"
        )
        req_term_max = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), "<", "1023"
        )
        input_specification = InputSpecification([req_sqrt, req_term_min, req_term_max])

        input = isla_generator.generate(input_specification)

        assertInputSpecification(grammar, input, input_specification)
        print(input)

    def test_isla_generator_with_negativs(self):
        isla_generator = ISLAGenerator(grammar=grammar, allow_negatives=True)

        req_sqrt = Requirement(
            ExistenceFeature("exists(<function>@0)", "<function>", "sqrt"), ">", "0.5"
        )
        req_term_min = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), ">", "-900.0"
        )
        req_term_max = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), "<", "-890.0"
        )
        input_specification = InputSpecification([req_sqrt, req_term_min, req_term_max])

        input = isla_generator.generate(input_specification)

        assertInputSpecification(grammar, input, input_specification)
        print(input)


if __name__ == "__main__":
    unittest.main()
