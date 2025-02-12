from typing import Iterable, Optional

from dbg.data.input import Input
from dbg.explanation.candidate import ExplanationSet
from dbg.learner.learner import Learner


class AlhazenLearner(Learner):

    def learn_explanation(self, test_inputs: Iterable[Input], **kwargs) -> Optional[ExplanationSet]:
        pass

    def get_explanations(self) -> Optional[ExplanationSet]:
        pass

    def get_best_candidates(self) -> Optional[ExplanationSet]:
        pass

