import subprocess
from os import PathLike, path
from pathlib import Path
import string
from fuzzingbook.Grammars import Grammar, srange, is_valid_grammar

from alhazen.t4p_common import (
    Environment,
    API,
    TestResult,
)


file_dir = Path(__file__).parent.absolute()
HARNESS_FILE = str(Path(file_dir / "harness.py"))


class PySnooperAPI(API):
    def __init__(self, expected_error: bytes, default_timeout: int = 5):
        self.expected_error = expected_error
        # self.translator = python.ToASTVisitor(python.GENERATIVE_GRAMMAR)
        super().__init__(default_timeout=default_timeout)

    # noinspection PyBroadException
    def run(self, system_test_path: PathLike, environ: Environment) -> TestResult:
        try:
            with open(system_test_path, "r") as fp:
                test = fp.read()
            if test:
                test = test.split("\n")
            else:
                test = []
            process = subprocess.run(
                ["python", HARNESS_FILE] + test,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=self.default_timeout,
                env=environ,
            )
            if process.returncode:
                if self.expected_error.decode() in process.stderr.decode():
                    return TestResult.FAILING
                else:
                    return TestResult.UNDEFINED
            else:
                return TestResult.PASSING
        except subprocess.TimeoutExpired:
            return TestResult.UNDEFINED
        except Exception:
            return TestResult.UNDEFINED


PYSNOOPER_2_API = PySnooperAPI(
    expected_error=b"TypeError: Tracer.__init__() got an unexpected keyword argument 'custom_repr'\n"
)

PYSNOOPER_3_API = PySnooperAPI(
    expected_error=b"NameError: name 'output_path' is not defined"
)


PYSNOOPER_GRAMMAR: Grammar = {
    "<start>": ["<options>"],
    "<options>": ["", "<option_list>"],
    "<option_list>": ["<option>", "<option_list>\n<option>"],
    "<option>": [
        "<variables>",
        "<depth>",
        "<prefix>",
        "<watch>",
        "<custom_repr>",
        "<overwrite>",
        "<thread_info>",
        "<output>",
    ],
    "<output>": ["-o", "-o<path>"],
    "<variables>": ["-v<variable_list>"],
    "<depth>": ["-d<int>"],
    "<prefix>": ["-p<str>"],
    "<watch>": ["-w<variable_list>"],
    "<custom_repr>": ["-c<predicate_list>"],
    "<overwrite>": ["-O"],
    "<thread_info>": ["-T"],
    "<path>": ["<location>", "<location>.<str>"],
    "<location>": ["<str>", path.join("<path>", "<str>")],
    "<variable_list>": ["<variable>", "<variable_list>,<variable>"],
    "<variable>": ["<name>", "<variable>.<name>"],
    "<name>": ["<letter><chars>"],
    "<chars>": ["", "<chars><char>"],
    "<letter>": srange(string.ascii_letters),
    "<digit>": srange(string.digits),
    "<char>": ["<letter>", "<digit>", "_"],
    "<int>": ["<nonzero><digits>", "0"],
    "<digits>": ["", "<digits><digit>"],
    "<nonzero>": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<str>": ["<char><chars>"],
    "<predicate_list>": ["<predicate>", "<predicate_list>,<predicate>"],
    "<predicate>": ["<p_function>=<t_function>"],
    "<p_function>": ["int", "str", "float", "bool"],
    "<t_function>": ["repr", "str", "int"],
}

assert is_valid_grammar(PYSNOOPER_GRAMMAR)
