import unittest

from alhazen.features import (
    ExistenceFeature,
    NumericInterpretation,
    get_all_features,
    compute_feature_values,
)
from alhazen_formalizations.calculator import grammar


class TestFeatures(unittest.TestCase):
    def test_features(self):
        features = get_all_features(grammar)

        existence_features = 0
        numeric_features = 0

        for feature in features:
            if isinstance(feature, ExistenceFeature):
                existence_features += 1
            elif isinstance(feature, NumericInterpretation):
                numeric_features += 1

        self.assertEqual(existence_features, 27)
        self.assertEqual(numeric_features, 4)

        expected_feature_names = [
            "exists(<start>)",
            "exists(<start> == <function>(<term>))",
            "exists(<function>)",
            "exists(<function> == sqrt)",
            "exists(<function> == tan)",
            "exists(<function> == cos)",
            "exists(<function> == sin)",
            "exists(<term>)",
            "exists(<term> == -<value>)",
            "exists(<term> == <value>)",
            "exists(<value>)",
            "exists(<value> == <integer>.<integer>)",
            "exists(<value> == <integer>)",
            "exists(<integer>)",
            "exists(<integer> == <digit><integer>)",
            "exists(<integer> == <digit>)",
            "exists(<digit>)",
            "exists(<digit> == 0)",
            "exists(<digit> == 1)",
            "exists(<digit> == 2)",
            "exists(<digit> == 3)",
            "exists(<digit> == 4)",
            "exists(<digit> == 5)",
            "exists(<digit> == 6)",
            "exists(<digit> == 7)",
            "exists(<digit> == 8)",
            "exists(<digit> == 9)",
            "num(<term>)",
            "num(<value>)",
            "num(<digit>)",
            "num(<integer>)",
        ]

        actual_feature_names = set(map(lambda f: f.name_rep(), features))
        self.assertEqual(set(actual_feature_names), set(expected_feature_names))

    def test_feature_values(self) -> None:
        sample_list = ["sqrt(-900)", "sin(24)", "cos(-3.14)"]

        expected_feature_values = {
            "sqrt(-900)": {
                "exists(<start>)": 1,
                "exists(<start> == <function>(<term>))": 1,
                "exists(<function>)": 1,
                "exists(<function> == sqrt)": 1,
                "exists(<function> == tan)": 0,
                "exists(<function> == cos)": 0,
                "exists(<function> == sin)": 0,
                "exists(<term>)": 1,
                "exists(<term> == -<value>)": 1,
                "exists(<term> == <value>)": 0,
                "exists(<value>)": 1,
                "exists(<value> == <integer>.<integer>)": 0,
                "exists(<value> == <integer>)": 1,
                "exists(<integer>)": 1,
                "exists(<integer> == <digit><integer>)": 1,
                "exists(<integer> == <digit>)": 1,
                "exists(<digit>)": 1,
                "exists(<digit> == 0)": 1,
                "exists(<digit> == 1)": 0,
                "exists(<digit> == 2)": 0,
                "exists(<digit> == 3)": 0,
                "exists(<digit> == 4)": 0,
                "exists(<digit> == 5)": 0,
                "exists(<digit> == 6)": 0,
                "exists(<digit> == 7)": 0,
                "exists(<digit> == 8)": 0,
                "exists(<digit> == 9)": 1,
                "num(<term>)": -900.0,
                "num(<value>)": 900.0,
                "num(<digit>)": 9.0,
                "num(<integer>)": 900.0,
            },
            "sin(24)": {
                "exists(<start>)": 1,
                "exists(<start> == <function>(<term>))": 1,
                "exists(<function>)": 1,
                "exists(<function> == sqrt)": 0,
                "exists(<function> == tan)": 0,
                "exists(<function> == cos)": 0,
                "exists(<function> == sin)": 1,
                "exists(<term>)": 1,
                "exists(<term> == -<value>)": 0,
                "exists(<term> == <value>)": 1,
                "exists(<value>)": 1,
                "exists(<value> == <integer>.<integer>)": 0,
                "exists(<value> == <integer>)": 1,
                "exists(<integer>)": 1,
                "exists(<integer> == <digit><integer>)": 1,
                "exists(<integer> == <digit>)": 1,
                "exists(<digit>)": 1,
                "exists(<digit> == 0)": 0,
                "exists(<digit> == 1)": 0,
                "exists(<digit> == 2)": 1,
                "exists(<digit> == 3)": 0,
                "exists(<digit> == 4)": 1,
                "exists(<digit> == 5)": 0,
                "exists(<digit> == 6)": 0,
                "exists(<digit> == 7)": 0,
                "exists(<digit> == 8)": 0,
                "exists(<digit> == 9)": 0,
                "num(<term>)": 24.0,
                "num(<value>)": 24.0,
                "num(<digit>)": 4.0,
                "num(<integer>)": 24.0,
            },
            "cos(-3.14)": {
                "exists(<start>)": 1,
                "exists(<start> == <function>(<term>))": 1,
                "exists(<function>)": 1,
                "exists(<function> == sqrt)": 0,
                "exists(<function> == tan)": 0,
                "exists(<function> == cos)": 1,
                "exists(<function> == sin)": 0,
                "exists(<term>)": 1,
                "exists(<term> == -<value>)": 1,
                "exists(<term> == <value>)": 0,
                "exists(<value>)": 1,
                "exists(<value> == <integer>.<integer>)": 1,
                "exists(<value> == <integer>)": 0,
                "exists(<integer>)": 1,
                "exists(<integer> == <digit><integer>)": 1,
                "exists(<integer> == <digit>)": 1,
                "exists(<digit>)": 1,
                "exists(<digit> == 0)": 0,
                "exists(<digit> == 1)": 1,
                "exists(<digit> == 2)": 0,
                "exists(<digit> == 3)": 1,
                "exists(<digit> == 4)": 1,
                "exists(<digit> == 5)": 0,
                "exists(<digit> == 6)": 0,
                "exists(<digit> == 7)": 0,
                "exists(<digit> == 8)": 0,
                "exists(<digit> == 9)": 0,
                "num(<term>)": -3.14,
                "num(<value>)": 3.14,
                "num(<digit>)": 4.0,
                "num(<integer>)": 14.0,
            },
        }

        all_features = get_all_features(grammar)
        for sample in sample_list:
            input_features = compute_feature_values(sample, grammar)

            for feature in all_features:
                key = feature.name_rep()
                expected = expected_feature_values[sample][key]
                actual = input_features[key]
                assert (
                    expected == actual
                ), f"Wrong feature value for sample {sample} and feature {key}: expected {expected} but is {actual}."

        print("All checks passed!")


if __name__ == "__main__":
    unittest.main()
