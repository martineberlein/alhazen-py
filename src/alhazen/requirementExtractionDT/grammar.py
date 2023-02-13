import re
import logging
from pathlib import Path
import subprocess
from .generators import ReachTargetGenerator, SampleGenerator
import numpy as np
import pandas
from typing import List

from . import requirements, bug_class
from .features import Feature

column_name_pattern = re.compile(r"([a-zA-Z0-9_]+)//([0-9]+)//[a-z]+")


def turn_into_prob(feature):
    if feature.endswith("//prob"):
        return feature
    elif feature.endswith("//present"):
        return feature[: -len("//present")] + "//prob"
    else:
        raise AssertionError(
            "I have no idea how to find probabilities for {feature}".format(
                feature=feature
            )
        )


def construct_feature_map(features):
    feature_map = {}
    for c in features:
        colname = re.match(column_name_pattern, str(c))
        if colname is not None:
            rulename = colname.group(1)
            alternative = int(colname.group(2))

            if rulename in feature_map:
                alt = feature_map[rulename]
                alt = max(alternative, alt)
            else:
                alt = alternative
            feature_map[rulename] = alt
    return feature_map


class BasiliskBasedGenerator(SampleGenerator):
    def __init__(
        self, features: List[Feature], bug: bug_class.Bug, grammar_file, random_seed
    ):
        self._callcount = 1
        self.__bug = bug
        self._features = features
        self.__grammar_file = grammar_file
        self.__random_seed = random_seed
        assert "run_basilisk" in dir(
            self.__bug
        ), "This subject is not prepared for basilisk!"

    def generate_samples(self, iter_dir, feature_file, tree, data):
        logging.info("Creating new samples with info from the tree only ...")
        # determine the bounds
        bounds = (
            pandas.DataFrame(
                [
                    {
                        "feature": c.name(),
                        "min": data[c.name()].min(),
                        "max": data[c.name()].max(),
                    }
                    for c in self._features
                ],
                columns=["feature", "min", "max"],
            )
            .set_index(["feature"])
            .transpose()
        )

        # go through tree leaf by leaf
        all_reqs = set()
        samples = set()
        for path in requirements.tree_to_paths(tree, self._features):
            # generate conditions
            reqs = []
            for p in range(0, len(path)):
                r = path.get(p)
                reqs.append(r.get_str(bounds))
            str_req = " ".join(sorted(reqs))
            # add the path to a sample for this requirement
            sample = Path(path.find_sample(data))
            if sample is not None:
                samples.add(sample)
                str_req = str_req + f" (in/{sample.name})"
            all_reqs.add(str_req)
        self._run_specs(feature_file, iter_dir, all_reqs, samples)

    def _run_specs(
        self, feature_file: Path, iter_dir: Path, all_reqs: List[str], samples
    ) -> None:
        # output spec
        spec_file = iter_dir / f"specs{self._callcount}"
        with open(spec_file, "w") as spec:
            for r in all_reqs:
                spec.write(r)
                spec.write("\n")
        self._generate_samples(spec_file, feature_file, iter_dir, samples)

    def _generate_samples(
        self, spec_file: Path, feature_file: Path, iter_dir: Path, samples
    ) -> None:
        """
        Run basilisk as a sample generator.
        spec_file: File which contains the specs for reach target
        feature_file: File to write the features in the new samples to (basilisk won't do this, will it?).
        iterdir: Directory for this iteration. New samples will be written to iter_dir / "samples".
        """
        logging.info("Generating more samples ...")
        assert iter_dir / "features.csv" == feature_file
        try:
            self.__bug.run_basilisk(
                self.__grammar_file,
                spec_file,
                iter_dir,
                iter_dir / f"basilisk_{self._callcount}.log.bz2",
                samples,
                random_seed=(self._callcount * self.__random_seed),
            )
            self._callcount = self._callcount + 1
        except subprocess.CalledProcessError:
            # just log it and have alhazen deal with the fact that there are no samples
            logging.exception(f"Basilisk failed in {iter_dir}")
        except subprocess.TimeoutExpired:
            # just log it and have alhazen deal with the fact that there are no samples
            logging.exception(f"Basilisk timed out in {iter_dir}")


class OccurrenceBasedGenerator(ReachTargetGenerator):
    def __init__(
        self, features: List[Feature], bug: bug_class.Bug, grammar_file, random_seed
    ):
        super().__init__(bug, grammar_file, random_seed)
        self._features: List[Feature] = features

    def generate_samples(self, iter_dir, feature_file, tree, data):
        logging.info("Creating new samples with info from the tree only ...")
        # determine the bounds
        bounds = (
            pandas.DataFrame(
                [
                    {
                        "feature": c.name(),
                        "min": data[c.name()].min(),
                        "max": data[c.name()].max(),
                    }
                    for c in self._features
                ],
                columns=["feature", "min", "max"],
            )
            .set_index(["feature"])
            .transpose()
        )

        # go through tree leaf by leaf
        all_reqs = set()
        for path in requirements.tree_to_paths(tree, self._features):
            # generate conditions
            for i in range(0, len(path) + 1):
                reqs_list = []
                bins = format(i, "#0{}b".format(len(path) + 2))[2:]
                for p, b in zip(range(0, len(bins)), bins):
                    r = path.get(p)
                    if "1" == b:
                        reqs_list.append(r.get_neg(bounds))
                    else:
                        reqs_list.append([r.get_str(bounds)])
                for reqs in self.all_combinations(reqs_list):
                    all_reqs.add(" ".join(sorted(reqs)))
        self._run_specs(feature_file, iter_dir, all_reqs)

    def all_combinations(self, reqs_lists):
        result = [[]]
        for reqs in reqs_lists:
            t = []
            for r in reqs:
                for i in result:
                    t.append(i + [r])
            result = t
        return result


class CorrelationOccurrenceBasedGenerator(OccurrenceBasedGenerator):
    def __init__(
        self, features: List[Feature], bug: bug_class.Bug, grammar_file, random_seed
    ):
        super().__init__(features, bug, grammar_file, random_seed)

    def generate_samples(self, iter_dir, feature_file, tree, data):
        logging.info("Checking correlations ...")
        feature_names = list(map(lambda f: f.name(), self._features))
        corr = data[feature_names].corr(method="spearman")

        # delete small effects and diagonal
        for f1 in feature_names:
            for f2 in feature_names:
                if f1 == f2 or abs(corr[f1][f2]) < 0.6:
                    corr[f1][f2] = np.nan

        # find maximum correlation
        self.__adds = {}
        maxcorrs = corr.idxmax().fillna("")
        for f in self._features:
            # within maxcorrs, features with no correlations map to NaN
            if 0 != len(maxcorrs[f.name()]):
                f2 = [f2 for f2 in self._features if f2.name() == maxcorrs[f.name()]][0]
                mean = data[f.name()].mean()
                if corr[f.name()][f2.name()] < 0:
                    logging.info(f"{f2} enables {f}")
                    if f.is_binary():
                        self.__adds[f2] = f.name()
                    elif f.is_valid(mean):
                        self.__adds[f2] = f"{f.name()} <= {mean}"
                else:
                    logging.info(f"{f2} disables {f}")
                    if f.is_binary():
                        self.__adds[f2] = f"!{f.name()}"
                    elif f.is_valid(mean):
                        self.__adds[f2] = f"{f.name()} > {mean}"
        super().generate_samples(iter_dir, feature_file, tree, data)

    def _run_specs(self, feature_file, iter_dir, all_reqs):
        # add enabled stuff
        p_all_reqs = set()
        for r in all_reqs:
            p_all_reqs.add(r)
            for trigger, additional in self.__adds.items():
                if trigger.name() in r:
                    r = f"{r} {additional}"
                    p_all_reqs.add(r)
        super()._run_specs(feature_file, iter_dir, p_all_reqs)
