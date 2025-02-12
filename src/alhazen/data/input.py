from typing import Optional

from dbg.data.input import Input
from dbg.data.oracle import OracleResult


class AlhazenInput(Input):

    def traverse(self):
        pass

    def __hash__(self) -> int:
        pass

    @classmethod
    def from_str(cls, grammar, input_string, oracle: Optional[OracleResult] = None):
        pass