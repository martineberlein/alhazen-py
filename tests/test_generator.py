import unittest

from alhazen.Activity1_1_FeatureExtraction import (
    ExistenceFeature,
    NumericInterpretation,
)
from alhazen.Activity3_RequirementExtraction import Requirement, InputSpecification
from alhazen.generator import SimpleGenerator, AdvancedGenerator, Generator
from alhazen_formalizations.calculator import grammar


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

        result = generator.generate(input_specification)
        print(result)


if __name__ == "__main__":
    unittest.main()
