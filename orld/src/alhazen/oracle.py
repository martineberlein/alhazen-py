from enum import Enum


class OracleResult(Enum):
    BUG = "BUG"
    NO_BUG = "NO_BUG"
    UNDEF = "UNDEF"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value
