import logging
from pathlib import Path

import numpy as np
from typing import List, Optional

from alhazen.oracle import OracleResult
from . import treetools
from .features import Feature


class Requirement:
    def __init__(self, feature: Feature, mini, maxi):
        self.__feature: Feature = feature
        self.__mini = mini
        self.__maxi = maxi

    def feature(self) -> Feature:
        return self.__feature

    def mini(self):
        return self.__mini

    def maxi(self):
        return self.__maxi

    def get_key(self) -> str:
        return self.__feature.key()

    def is_binary(self) -> bool:
        return self.__feature.is_binary()

    def get_str(self, bounds) -> str:
        if self.is_binary():
            if self.__mini < 0 <= self.__maxi:
                # feature is NOT included
                return f"!{self.__feature.name()}"
            if self.__mini < 1 <= self.__maxi:
                # feature is included
                return self.__feature.name()
            raise AssertionError("How is this possible?")
        else:
            if (not np.isinf(self.__mini)) and (not np.isinf(self.__maxi)):
                return f"{self.__feature.name()} in [{self.__mini}, {self.__maxi}]"
            elif not np.isinf(self.__maxi):
                return f"{self.__feature.name()} <= {self.__maxi}"
            else:
                return f"{self.__feature.name()} > {self.__mini}"

    def get_str_ext(self) -> str:
        if (not np.isinf(self.__mini)) and (not np.isinf(self.__maxi)):
            return f"{self.__feature} in [{self.__mini}, {self.__maxi}]"
        elif not np.isinf(self.__maxi):
            return f"{self.__feature} <= {self.__maxi}"
        else:
            return f"{self.__feature} > {self.__mini}"

    def get_neg(self, bounds) -> List[str]:
        if self.is_binary():
            if self.__mini < 0 <= self.__maxi:
                # feature is NOT included, so, the negated condition is to include it
                return [self.__feature.name()]
            if self.__mini < 1 <= self.__maxi:
                # feature is included, so exclude it
                return [f"!{self.__feature.name()}"]
            raise AssertionError("How is this possible?")
        else:
            if (not np.isinf(self.__mini)) and (not np.isinf(self.__maxi)):
                return [
                    f"{self.__feature.name()} in [{bounds.at['min', self.__feature.name()]},{self.__mini}]",
                    f"{self.__feature.name()} in [{self.__maxi}, {bounds.at['max', self.__feature.name()]}]",
                ]
            elif not np.isinf(self.__maxi):
                return [f"{self.__feature.name()} <= {self.__maxi}"]
            else:
                return [f"{self.__feature.name()} > {self.__mini}"]

    def get_neg_ext(self, bounds) -> List[str]:
        if (not np.isinf(self.__mini)) and (not np.isinf(self.__maxi)):
            return [
                f"{self.__feature} in [{bounds.at['min', self.__feature]},{self.__mini}]",
                f"{self.__feature} in [{self.__maxi}, {bounds.at['max', self.__feature]}]",
            ]
        elif not np.isinf(self.__maxi):
            return [f"{self.__feature} > {self.__maxi}"]
        else:
            return [f"{self.__feature} <= {self.__mini}"]


class TreePath:
    def __init__(
        self, samplefile: Optional[Path], is_bug: bool, requirements: List[Requirement]
    ):
        self.__sample = samplefile
        self.__is_bug = is_bug
        self.__requirements: List[Requirement] = requirements

    def is_bug(self) -> bool:
        return self.__is_bug

    def get(self, idx):
        return self.__requirements[idx]

    @property
    def requirements(self) -> List[Requirement]:
        return self.__requirements

    def __len__(self) -> int:
        return len(self.__requirements)


def lower_middle(start, end):
    if start == end:
        return start - abs(start)
    return start + ((end - start) / 2)


def upper_middle(start, end):
    if start == end:
        return end + abs(end)
    return start + ((end - start) / 2)


def min_digits(mini):
    return int("1" + "".join([0] * int(mini - 1)))


def max_digits(maxi):
    return int("".join([9] * int(maxi)))


def tree_to_paths(tree, features: List[Feature | str], classes=None) -> List[TreePath]:
    logging.info("Extracting requirements from tree ...")
    paths = []
    # go through tree leaf by leaf
    for path in treetools.all_path(tree):
        requirements = []
        is_bug = OracleResult.BUG == treetools.prediction_for_path(tree, path, classes)
        # find the requirements
        box = treetools.box(tree, path, feature_names=features).transpose()
        for feature, row in box.iterrows():
            mini = row["min"]
            maxi = row["max"]
            if (not np.isinf(mini)) or (not np.isinf(maxi)):
                requirements.append(Requirement(feature, mini, maxi))
        paths.append(TreePath(None, is_bug, requirements))

    return paths
