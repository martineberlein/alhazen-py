import unittest

from fuzzingbook.Parser import EarleyParser
from isla.derivation_tree import DerivationTree

from alhazen.Activity1_1_FeatureExtraction import (
    ExistenceFeature,
    NumericInterpretation,
    collect_features,
)
from alhazen.Activity3_RequirementExtraction import Requirement, InputSpecification
from alhazen.generator import generate_samples_advanced, AdvancedGenerator
from alhazen_formalizations.calculator import grammar


class TestGenerator(unittest.TestCase):
    def test_validation_requirement(self):
        exist_sqrt = ExistenceFeature("exists(<function>@0)", "<function>", "sqrt")
        req_sqrt = Requirement(exist_sqrt, ">", "0.5")

        inp = "sqrt(-900)"
        derivation_tree = DerivationTree.from_parse_tree(
            next(EarleyParser(grammar).parse(inp))
        )
        inp_features = collect_features({derivation_tree}, grammar=grammar)

        generator = AdvancedGenerator(grammar)
        result = generator.validate_requirement(
            sample_features=inp_features, requirement=req_sqrt
        )
        self.assertTrue(result)

    def test_validation_specification(self):
        req_sqrt = Requirement(
            ExistenceFeature("exists(<function>@0)", "<function>", "sqrt"), ">", "0.5"
        )
        req_term = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), "<", "-31.0"
        )
        input_specification = InputSpecification([req_sqrt, req_term])

        inp = "sqrt(-900)"
        derivation_tree = DerivationTree.from_parse_tree(
            next(EarleyParser(grammar).parse(inp))
        )

        generator = AdvancedGenerator(grammar)
        result, _ = generator.validate(
            sample=derivation_tree, specification=input_specification
        )
        self.assertEqual(result, True)

        req_term_1 = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), ">", "10"
        )
        input_specification_1 = InputSpecification([req_sqrt, req_term_1])
        result, count = generator.validate(
            sample=derivation_tree, specification=input_specification_1
        )
        self.assertEqual(result, False)
        self.assertEqual(count, 1)

    def test_requirements(self):
        exist_sqrt = ExistenceFeature("exists(<function>@0)", "<function>", "sqrt")
        exist_digit = ExistenceFeature("exists(<digit>)", "<digit>", "<digit>")

        req_sqrt = Requirement(exist_sqrt, ">", "0.5")
        req_digit = Requirement(exist_digit, "<=", "0.5")

        test_spec_0 = InputSpecification([req_sqrt, req_digit])

        num_term = NumericInterpretation("num(<term>)", "<term>")
        req_term = Requirement(num_term, "<", "-31.0")
        test_spec1 = InputSpecification([req_sqrt, req_term])

        for _ in range(10):
            generate_samples_advanced(
                grammar, [test_spec1], 10
            )  # TODO better test case and assertion


if __name__ == "__main__":
    unittest.main()
