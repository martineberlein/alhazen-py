import pandas
import numpy

# from antlr4 import *
# from specsparser import SpecsParser, SpecsLexer, SpecsVisitor
from typing import List


class Feature:
    def __init__(self, key, readable_name):
        self._key = key
        self.__readable_name = readable_name

    def key(self):
        return self._key

    def readable_name(self) -> str:
        return self.__readable_name

    def name(self) -> str:
        raise AssertionError("Implement in subclass!")

    def is_valid(self, value):
        return True

    def clean_data(self, data: pandas.DataFrame) -> pandas.DataFrame:
        return data

    def is_binary(self):
        return False

    def is_feasible(self, threshold: float) -> bool:
        return True

    def __repr__(self):
        return self.__readable_name

    def __str__(self):
        return self.name()

    def node_label(self, threshold):
        rn = self.readable_name().replace('"', '\\"')
        return f"{rn} <= {threshold}"

    def left_label(self):
        return "yes"

    def right_label(self):
        return "no"


class ExistenceFeature(Feature):
    def __init__(self, key: str, readable_name: str):
        super().__init__(key, readable_name)

    def is_valid(self, value):
        return value in [0, 1]

    def is_binary(self) -> bool:
        return True

    def node_label(self, threshold):
        return self.readable_name().replace('"', "'")

    def left_label(self):
        return "no"

    def right_label(self):
        return "yes"

    def name(self) -> str:
        return self._key


class CharLengthFeature(Feature):
    def __init__(self, key: str, readable_name: str):
        super().__init__(key, readable_name)

    def is_feasible(self, threshold: float) -> bool:
        return threshold > 0

    def is_valid(self, value):
        return value >= 0

    def clean_data(self, data: pandas.DataFrame) -> pandas.DataFrame:
        # use min for nan, so non-existing symbols have negative length
        data[self.name()] = data[self.name()].fillna(numpy.finfo("float32").min)
        return data

    def name(self):
        return f"char-length({self._key})"


class QaLengthFeature(Feature):
    def __init__(self, key: str, readable_name: str):
        super().__init__(key, readable_name)

    def is_valid(self, value):
        return value >= 0

    def clean_data(self, data: pandas.DataFrame) -> pandas.DataFrame:
        # use min for nan, so non-existing symbols have negative length
        data[self.name()] = data[self.name()].fillna(numpy.finfo("float32").min)
        return data

    def is_feasible(self, threshold: float) -> bool:
        return threshold > 0

    def name(self):
        return f"qa-length({self._key})"


class MaxCharFeature(Feature):
    def __init__(self, key: str, readable_name: str):
        super().__init__(key, readable_name)

    def is_valid(self, value):
        return True

    def clean_data(self, data: pandas.DataFrame) -> pandas.DataFrame:
        # use the mean for nan, so non-existing things have no influence on the outcome
        mean = data[self.name()].mean()
        if numpy.isnan(mean):
            mean = 0
        data[self.name()] = data[self.name()].fillna(mean)
        return data

    def name(self):
        return f"max-char({self._key})"


class MaxNumberFeature(Feature):
    def __init__(self, key: str, readable_name: str):
        super().__init__(key, readable_name)

    def is_valid(self, value):
        return True

    def clean_data(self, data: pandas.DataFrame) -> pandas.DataFrame:
        # use the mean for nan, so non-existing things have no influence on the outcome
        mean = data[self.name()].mean()
        if numpy.isnan(mean):
            mean = 0
        data[self.name()] = data[self.name()].fillna(mean)
        return data

    def name(self):
        return f"max-num({self._key})"


""""

class FeatureVisitor(SpecsVisitor.SpecsVisitor):

    def __init__(self, readable_name: str):
        self.__readable_name: str = readable_name

    def visitKPathFeature(self, ctx: SpecsParser.SpecsParser.KPathFeatureContext):
        return ExistenceFeature(ctx.getText(), self.__readable_name)

    def visitParseableKeyFeature(self, ctx: SpecsParser.SpecsParser.Binary_featureContext):
        return ExistenceFeature(ctx.getText(), self.__readable_name)

    def visitCharLength(self, ctx: SpecsParser.SpecsParser.CharLengthContext):
        return CharLengthFeature(ctx.argument().parseable_key().getText(), self.__readable_name)

    def visitQaLength(self, ctx: SpecsParser.SpecsParser.QaLengthContext):
        return QaLengthFeature(ctx.argument().parseable_key().getText(), self.__readable_name)

    def visitMaxNumber(self, ctx: SpecsParser.SpecsParser.MaxNumberContext):
        return MaxNumberFeature(ctx.argument().parseable_key().getText(), self.__readable_name)

    def visitMaxChar(self, ctx: SpecsParser.SpecsParser.MaxCharContext):
        return MaxCharFeature(ctx.argument().parseable_key().getText(), self.__readable_name)


def parse_feature(feature: str, readable_name: str) -> Feature:
    input_stream = InputStream(feature)
    lexer = SpecsLexer.SpecsLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = SpecsParser.SpecsParser(token_stream)
    feat_ctx = parser.feature()
    feat = feat_ctx.accept(FeatureVisitor(readable_name))
    if feat is None:
        raise AssertionError(f"Could not parse: {readable_name}")
    return feat
"""


def find_existence_index(features: List[Feature], feature: Feature):
    for idx, f in enumerate(features):
        if isinstance(f, ExistenceFeature) and f.key() == feature.key():
            return idx
    raise AssertionError("There is no existence feature with this key!")


def extract_features(feature_file) -> List[Feature]:
    """This function extracts a sorted list of features from the features file, which is generated
    by substr grammar or the depth tool."""
    depth_data = pandas.read_csv(
        feature_file, dtype={"name": str, "depth": numpy.int32}, keep_default_na=False
    )

    depth_data["feature"] = depth_data["name"]
    sorted_features = depth_data.sort_values(by=["name"])
    return list(
        map(
            lambda r: parse_feature(r[1]["name"], r[1]["readable name"]),
            sorted_features.iterrows(),
        )
    )
