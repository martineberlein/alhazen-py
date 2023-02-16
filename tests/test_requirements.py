import unittest
import pandas

from sklearn.tree import DecisionTreeClassifier
from fuzzingbook.Parser import EarleyParser, tree_to_string, is_valid_grammar
from isla.derivation_tree import DerivationTree

from alhazen.requirementExtractionDT.requirements import tree_to_paths
from alhazen.features import (
    ExistenceFeature,
    NumericInterpretation,
    collect_features,
    extract_existence,
    extract_numeric,
)
from alhazen.input_specifications import (
    Requirement,
    InputSpecification,
    SPECIFICATION_GRAMMAR,
    create_new_input_specification,
)
from alhazen.generator import AdvancedGenerator
from alhazen_formalizations.calculator import grammar
from alhazen.input import Input


class TestInputSpecifications(unittest.TestCase):
    def setUp(self) -> None:
        features = [
            {"function-sqrt": 1, "function-cos": 0, "function-sin": 0, "number": -900},
            {"function-sqrt": 0, "function-cos": 1, "function-sin": 0, "number": 300},
            {"function-sqrt": 1, "function-cos": 0, "function-sin": 0, "number": -1},
            {"function-sqrt": 0, "function-cos": 1, "function-sin": 0, "number": -10},
            {"function-sqrt": 0, "function-cos": 0, "function-sin": 1, "number": 36},
            {"function-sqrt": 0, "function-cos": 0, "function-sin": 1, "number": -58},
            {"function-sqrt": 1, "function-cos": 0, "function-sin": 0, "number": 27},
        ]
        oracle = ["BUG", "NO_BUG", "BUG", "NO_BUG", "NO_BUG", "NO_BUG", "NO_BUG"]

        self.feature_names = ["function-sqrt", "function-cos", "function-sin", "number"]
        x_data = pandas.DataFrame.from_records(features)

        clf = DecisionTreeClassifier(random_state=10)
        self.clf = clf.fit(x_data, oracle)

        self.grammar_features = extract_existence(grammar) + extract_numeric(grammar)

    def test_validation_requirement(self):
        exist_sqrt = ExistenceFeature("exists(<function>@0)", "<function>", "sqrt")
        req_sqrt = Requirement(exist_sqrt, ">", "0.5")

        inp = "sqrt(-900)"
        derivation_tree = DerivationTree.from_parse_tree(
            next(EarleyParser(grammar).parse(inp))
        )
        test_input = Input(derivation_tree)
        inp_features = collect_features(test_input, self.grammar_features)

        generator = AdvancedGenerator(grammar)
        result = generator.validate_requirement(
            input_features=inp_features, requirement=req_sqrt
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
            test_input=Input(derivation_tree), specification=input_specification
        )
        self.assertEqual(result, True)

        req_term_1 = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), ">", "10"
        )
        input_specification_1 = InputSpecification([req_sqrt, req_term_1])
        result, count = generator.validate(
            test_input=Input(derivation_tree), specification=input_specification_1
        )
        self.assertEqual(result, False)
        self.assertEqual(count, 1)

    def test_tree_to_paths(self):
        expected_paths = [
            ("function-sqrt <= 0.5", False),
            ("function-sqrt > 0.5 number <= 13.0", True),
            ("function-sqrt > 0.5 number > 13.0", False),
        ]

        all_paths = tree_to_paths(self.clf, self.feature_names)

        for path, expected in zip(all_paths, expected_paths):
            string_path = path.get(0).get_str_ext()
            for box in range(1, len(path)):
                string_path += " " + path.get(box).get_str_ext()
            self.assertEqual((string_path, path.is_bug()), expected)

    def test_specification_grammar(self):
        self.assertTrue(is_valid_grammar(SPECIFICATION_GRAMMAR))

        plain_input_specifications = [
            "exists(<function>@0) > 0.5, exists(<term>@0) <= 0.5, exists(<value>@1) <= 0.5",
            "exists(<digit>@9) <= 0.5, exists(<function>@0) > 0.5, num(<term>) > 0.05000000074505806",
            "exists(<digit>@2) <= 0.5, exists(<function>@0) < 0.5, num(<term>) <= 0.05000000074505806",
            "exists(<function>@0) > 0.5, num(<term>) > -3965678.1875",
        ]

        earley = EarleyParser(SPECIFICATION_GRAMMAR)
        for sample in plain_input_specifications:
            for tree in earley.parse(sample):
                self.assertEqual(tree_to_string(tree), sample)

    def test_parse_specification(self):
        sample_prediction_paths = [
            "exists(<function>@0) > 0.5, num(<term>) <= -38244758.0",
            "exists(<digit>@7) <= 0.5, exists(<function>@0) > 0.5, num(<term>) <= 0.05000000074505806",
            "exists(<digit>) > 1.5, exists(<function>@0) > 0.5, num(<term>) <= 0.21850000321865082",
            "exists(<function>@0) > 0.5",
        ]

        expected_input_specifications = [
            "NewInputSpecification(Requirement(exists(<function>@0) > 0.5), Requirement(num(<term>) <= -38244758.0))",
            "NewInputSpecification(Requirement(exists(<digit>@7) <= 0.5), Requirement(exists(<function>@0) > 0.5), Requirement(num(<term>) <= 0.05000000074505806))",
            "NewInputSpecification(Requirement(exists(<digit>) > 1.5), Requirement(exists(<function>@0) > 0.5), Requirement(num(<term>) <= 0.21850000321865082))",
            "NewInputSpecification(Requirement(exists(<function>@0) > 0.5))",
        ]

        all_features = extract_existence(grammar) + extract_numeric(grammar)

        earley = EarleyParser(SPECIFICATION_GRAMMAR)
        for sample, expected in zip(
            sample_prediction_paths, expected_input_specifications
        ):
            for tree in earley.parse(sample):
                input_specification = create_new_input_specification(tree, all_features)
                self.assertEqual(str(input_specification), expected)


if __name__ == "__main__":
    unittest.main()
