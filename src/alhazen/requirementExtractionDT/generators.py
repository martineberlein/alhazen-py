from abc import ABC, abstractmethod
from pathlib import Path
from pandas import DataFrame
from alhazen import tools
from sklearn.tree import DecisionTreeClassifier
from typing import List
import logging
import subprocess


class SampleGenerator(ABC):
    @abstractmethod
    def generate_samples(self, iterdir: Path, feature_file: Path, tree: DecisionTreeClassifier, data: DataFrame) -> None:
        """
        This method is called to generate samples with this generator.
        iterdir: Directory for this iteration. New samples will be written to iter_dir / "samples".
        feature_file: File to write the features in the new samples to.
        tree: The classifier to be analyzed.
        data: DataFrame with features and execution data from all known samples.
        """
        raise NotImplementedError("You must implement generate_samples!")


class MultiGenerator(SampleGenerator):
    """
    This generator combines multiple generators such that they act as if they were just one generator.
    """

    def __init__(self, generators):
        self.__generators = generators

    def generate_samples(self, iterdir, feature_file, tree, data):
        for gen in self.__generators:
            gen.generate_samples(iterdir, feature_file, tree, data)


class ReachTargetGenerator(SampleGenerator):

    def __init__(self, bug, grammar_file, random_seed):
        self._callcount = 1
        self.__bug = bug
        self.__grammar_file = grammar_file
        self.__random_seed = random_seed

    def _run_specs(self, feature_file: Path, iter_dir: Path, all_reqs: List[str]) -> None:
        # output spec
        spec_file = iter_dir / f"specs{self._callcount}"
        with open(spec_file, 'w') as spec:
            for r in all_reqs:
                spec.write(r)
                spec.write("\n")
        self._generate_samples(spec_file, feature_file, iter_dir)

    def _generate_samples(self, spec_file: Path, feature_file: Path, iter_dir: Path) -> None:
        """
        Run tribble to generate samples from all grammars in grammar dir.
        spec_file: File which contains the specs for reach target
        feature_file: File to write the features in the new samples to.
        iterdir: Directory for this iteration. New samples will be written to iter_dir / "samples".
        """
        logging.info("Generating more samples ...")
        try:
            tools.run_reach_target(self.__grammar_file, spec_file, feature_file, iter_dir / "samples",
                                   iter_dir / f"reach_target_spec{self._callcount}.log",
                                   random_seed=(self._callcount * self.__random_seed), suffix=self.__bug.suffix())
            self._callcount = self._callcount + 1
        except subprocess.CalledProcessError:
            # just log it and have alhazen deal with the fact that there are no samples
            logging.exception(f"ReachTargets failed in {iter_dir}")
        except subprocess.TimeoutExpired:
            # just log it and have alhazen deal with the fact that there are no samples
            logging.exception(f"ReachTargets timed out in {iter_dir}")
