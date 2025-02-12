from typing import Set

from dbg.core import HypothesisBasedExplainer
from dbg.data.input import Input
from dbg.explanation.candidate import ExplanationSet


class Alhazen(HypothesisBasedExplainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn_candidates(self, test_inputs: Set[Input]) -> ExplanationSet:
        pass

    def generate_test_inputs(self, candidates: ExplanationSet) -> Set[Input]:
        pass

