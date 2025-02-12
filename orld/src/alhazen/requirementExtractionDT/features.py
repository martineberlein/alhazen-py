import pandas
import numpy
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


class ExistenceFeature(Feature):
    def __init__(self, key: str, readable_name: str):
        super().__init__(key, readable_name)

    def is_valid(self, value):
        return value in [0, 1]

    def is_binary(self) -> bool:
        return True

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


def find_existence_index(features: List[Feature], feature: Feature):
    for idx, f in enumerate(features):
        if isinstance(f, ExistenceFeature) and f.key() == feature.key():
            return idx
    raise AssertionError("There is no existence feature with this key!")
