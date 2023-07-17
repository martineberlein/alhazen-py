import operator
import unittest

from isla.derivation_tree import DerivationTree
from fuzzingbook.Grammars import Grammar
from fuzzingbook.Parser import EarleyParser

from alhazen.input import Input
from alhazen.input_specifications import InputSpecification, Requirement
from alhazen.feature_collector import Collector
from alhazen_formalizations.calculator import grammar
from alhazen.features import EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE, ExistenceFeature, NumericInterpretation

FEATURES = {EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE}


def assert_input_specification(grammar: Grammar, input: Input, input_specification: InputSpecification):
    collector = Collector(grammar=grammar, features=FEATURES)
    input_features = collector.collect_features(input)

    for requirement in input_specification.requirements:
        key = requirement.feature.name

        operator_map = {
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
            "==": operator.eq,
        }

        if requirement.quant not in operator_map:
            raise ValueError("Invalid requirement quantifier")

        boolean_operator = operator_map[requirement.quant]

        print("%f %s %f" % (input_features.get(key), requirement.quant, float(requirement.value)))
        if not boolean_operator(input_features.get(key), float(requirement.value)):
            raise AssertionError("%f %s %f" % (input_features.get(key), requirement.quant, float(requirement.value)))


class TestInputSpec(unittest.TestCase):

    def test_input_spec(self):
        input = Input(
            DerivationTree.from_parse_tree(
                next(EarleyParser(grammar).parse("sqrt(901)"))
            )
        )
        req_sqrt = Requirement(
            ExistenceFeature("exists(<function>@0)", "<function>", "sqrt"), ">", "0.5"
        )
        req_term = Requirement(
            NumericInterpretation("num(<term>)", "<term>"), ">", "900.0"
        )
        input_specification = InputSpecification([req_sqrt, req_term])
        assert_input_specification(grammar, input, input_specification=input_specification)


if __name__ == "__main__":
    unittest.main()
