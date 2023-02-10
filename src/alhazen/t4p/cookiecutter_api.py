import os
import random
import re
import shutil
import string
import subprocess
import sys
from abc import abstractmethod, ABC
from os import PathLike
from pathlib import Path
from subprocess import Popen
from typing import List, Optional, Tuple
import subprocess
from os import PathLike, path
import string
from fuzzingbook.Grammars import Grammar, srange, is_valid_grammar
from isla.language import DerivationTree

from alhazen.t4p_common import Environment, HARNESS_FILE, API, TestResult, GrammarVisitor


class CookieCutterAPI(API, GrammarVisitor):
    REPO_PATH = "tests4py_repo"

    def __init__(self, default_timeout: int = 5):
        API.__init__(self, default_timeout=default_timeout)
        GrammarVisitor.__init__(self, grammar=grammar)
        self.config = None
        self.pre_hooks = []
        self.post_hooks = []
        self.path = []
        self.pre_hook_crash = False
        self.post_hook_crash = False

    def visit_hooks(self, node: DerivationTree):
        self.pre_hooks = []
        self.post_hooks = []
        self.pre_hook_crash = False
        self.post_hook_crash = False
        for children in node.children:
            self.visit(children)

    def visit_config(self, node: DerivationTree):
        self.config = node.to_string()
        for child in node.children:
            self.visit(child)

    def visit_repo_name(self, node: DerivationTree):
        self.path = list(
            map(lambda x: x.replace('"', ""), node.children[1].to_string().split(","))
        )

    def _set_hook_crash(self, hook: str, pre: bool = True):
        c, v = hook.split(",")
        if c == "exit" and v != "0":
            if pre:
                self.pre_hook_crash = True
            else:
                self.post_hook_crash = True

    def visit_pre_hook(self, node: DerivationTree):
        hook = node.children[1].to_string()
        self._set_hook_crash(hook)
        self.pre_hooks.append(hook)

    def visit_post_hook(self, node: DerivationTree):
        hook = node.children[1].to_string()
        self._set_hook_crash(hook, pre=False)
        self.post_hooks.append(hook)

    @staticmethod
    def _write_hook(hooks_path, hooks, file):
        for i, hook in enumerate(hooks):
            c, v = hook.split(",")
            with open(os.path.join(hooks_path, f"{file}.{i}"), "w") as fp:
                if sys.platform.startswith("win"):
                    if c == "exit":
                        fp.write(f"exit \\b {v}\n")
                    else:
                        fp.write("@echo off\n")
                        fp.write(f"echo {v}\n")
                else:
                    fp.write("#!/bin/sh\n")
                    if c == "exit":
                        fp.write(f"exit {v}\n")
                    else:
                        fp.write(f'echo "{v}"\n')

    def _setup(self):
        if os.path.exists(self.REPO_PATH):
            if os.path.isdir(self.REPO_PATH):
                shutil.rmtree(self.REPO_PATH, ignore_errors=True)
            else:
                os.remove(self.REPO_PATH)

        os.makedirs(self.REPO_PATH)

        with open(os.path.join(self.REPO_PATH, "cookiecutter.json"), "w") as fp:
            fp.write(self.config)

        repo_path = os.path.join(self.REPO_PATH, "{{cookiecutter.repo_name}}")
        os.makedirs(repo_path)

        with open(os.path.join(repo_path, "README.rst"), "w") as fp:
            fp.write(
                "============\nFake Project\n============\n\n"
                "Project name: **{{ cookiecutter.project_name }}**\n\n"
                "Blah!!!!\n"
            )

        if self.pre_hooks or self.post_hooks:
            hooks_path = os.path.join(self.REPO_PATH, "hooks")
            os.makedirs(hooks_path)
            if self.pre_hooks:
                self._write_hook(hooks_path, self.pre_hooks, "pre_gen_project")
            if self.post_hooks:
                self._write_hook(hooks_path, self.post_hooks, "post_gen_project")

    @abstractmethod
    def _validate(self, process: subprocess.Popen, stdout, stderr) -> TestResult:
        pass

    @abstractmethod
    def _get_command_parameters(self) -> List[str]:
        return []

    def _communicate(self, process: Popen) -> Tuple[bytes, bytes] | Tuple[str, str]:
        return process.communicate(20 * b"\n", self.default_timeout)

    # noinspection PyBroadException
    def run(self, system_test_path: PathLike, environ: Environment) -> TestResult:
        try:
            with open(system_test_path, "r") as fp:
                content = fp.read()
            self.visit_source(content)
            self._setup()
            if self.path:
                for p in self.path:
                    shutil.rmtree(p, ignore_errors=True)
            process = subprocess.Popen(
                ["cookiecutter"] + self._get_command_parameters() + [self.REPO_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environ,
            )
            # print(process)
            stdout, stderr = self._communicate(process)
            return self._validate(process, stdout, stderr)
        except subprocess.TimeoutExpired as e:
            # print(e)
            return TestResult.UNDEFINED
        except Exception as e:
            # print(e)
            return TestResult.UNDEFINED
        finally:
            if self.path:
                for p in self.path:
                    shutil.rmtree(p, ignore_errors=True)
            shutil.rmtree(self.REPO_PATH, ignore_errors=True)


class CookieCutter2API(CookieCutterAPI):

    def _get_command_parameters(self) -> List[str]:
        return ["--no-input", "-v"]

    def _validate(
        self, process: subprocess.Popen, stdout: bytes | str, stderr: bytes | str
    ) -> TestResult:
        if process.returncode != 0:
            return TestResult.UNDEFINED
        if isinstance(stdout, str):
            output = stdout
        else:
            output = stdout.decode("utf-8")
        for hook in self.pre_hooks + self.post_hooks:
            command, hook = hook.split(",")
            if command == "echo":
                hook_repr = "\n" + hook + "\n"
                if hook_repr in output:
                    output = output.replace(hook_repr, "\n", 1)
                else:
                    return TestResult.FAILING
            else:
                return TestResult.UNDEFINED
        return TestResult.PASSING


class CookieCutter3API(CookieCutterAPI):
    def __init__(self, default_timeout: int = 5):
        super().__init__(default_timeout=default_timeout)
        self.choice_pattern = re.compile(r"Choose from \d+(, \d)+ \(\d+(, \d)+\)")

    def _get_command_parameters(self) -> List[str]:
        return []

    def _validate(
        self, process: subprocess.Popen, stdout: bytes | str, stderr: bytes | str
    ) -> TestResult:
        if process.returncode != 0:
            return TestResult.UNDEFINED
        if isinstance(stdout, str):
            output = stdout
        else:
            output = stdout.decode("utf-8")
        if self.choice_pattern.search(output):
            return TestResult.FAILING
        return TestResult.PASSING


class CookieCutter4API(CookieCutterAPI):
    def __init__(self, default_timeout: int = 5):
        super().__init__(default_timeout=default_timeout)

    def _get_command_parameters(self) -> List[str]:
        return ["-v"]

    def _validate(
        self, process: subprocess.Popen, stdout: bytes | str, stderr: bytes | str
    ) -> TestResult:
        if isinstance(stderr, str):
            output = stderr
        else:
            output = stderr.decode("utf-8")
        captured = True
        if self.pre_hook_crash:
            if (
                "Stopping generation because pre_gen_project hook script didn't exit sucessfully"
                in output
            ):
                return TestResult.PASSING
            else:
                return TestResult.FAILING
        if self.post_hook_crash:
            if (
                "cookiecutter.exceptions.FailedHookException: Hook script failed"
                in output
            ):
                return TestResult.PASSING
            if self.post_hook_crash:
                return TestResult.FAILING
        return TestResult.PASSING


grammar: Grammar = {
    "<start>": ["<config>\n<hooks>"],
    "<config>": ["{<pairs>}", "{}"],
    "<hooks>": ["", "<hook_list>"],
    "<hook_list>": ["<hook>", "<hook_list>\n<hook>"],
    "<hook>": ["<pre_hook>", "<post_hook>"],
    "<pre_hook>": ["pre:<hook_content>"],
    "<post_hook>": ["post:<hook_content>"],
    "<hook_content>": ["echo,<str_with_spaces>", "exit,<int>"],
    "<pairs>": ["<pair>", "<pairs>,<pair>"],
    "<pair>": [
        "<full_name>",
        "<email>",
        "<github_username>",
        "<project_name>",
        "<repo_name>",
        "<project_short_description>",
        "<release_date>",
        "<year>",
        "<version>",
    ],
    "<full_name>": [
        '"full_name":"<str_with_spaces>"',
        '"full_name":[<str_with_spaces_list>]',
    ],
    "<email>": ['"email":"<email_address>"', '"email":[<email_list>]'],
    "<github_username>": [
        '"github_username":"<str>"',
        '"github_username":[<str_list>]',
    ],
    "<project_name>": [
        '"project_name":"<str_with_spaces>"',
        '"project_name":[<str_with_spaces_list>]',
    ],
    "<repo_name>": ['"repo_name":"<str>"', '"repo_name":[<str_list>]'],
    "<project_short_description>": [
        '"project_short_description":"<str_with_spaces>"',
        '"project_short_description":[<str_with_spaces_list>]',
    ],
    "<release_date>": ['"release_date":"<date>"', '"release_date":[<date_list>]'],
    "<year>": ['"year":"<int>"', '"year":[<int_list>]'],
    "<version>": ['"version":"<v>"', '"version":[<version_list>]'],
    "<str_with_spaces_list>": [
        '"<str_with_spaces>"',
        '<str_with_spaces_list>,"<str_with_spaces>"',
    ],
    "<email_list>": ['"<email_address>"', '<email_list>,"<email_address>"'],
    "<str_list>": ['"<str>"', '<str_list>,"<str>"'],
    "<int_list>": ['"<int>"', '<int_list>,"<int>"'],
    "<date_list>": ['"<date>"', '<date_list>,"<date>"'],
    "<version_list>": ['"<v>"', '<version_list>,"<v>"'],
    "<chars>": ["", "<chars><char>"],
    "<char>": srange(string.ascii_letters + string.digits + "_"),
    "<chars_with_spaces>": ["", "<chars_with_spaces><char_with_spaces>"],
    "<char_with_spaces>": srange(string.ascii_letters + string.digits + "_ "),
    "<str>": ["<char><chars>"],
    "<str_with_spaces>": ["<char_with_spaces><chars_with_spaces>"],
    "<email_address>": ["<str>@<str>.<str>"],
    "<date>": ["<day>.<month>.<int>", "<int>-<month>-<day>"],
    "<month>": ["0<nonzero>", "<nonzero>", "10", "11", "12"],
    "<day>": [
        "0<nonzero>",
        "<nonzero>",
        "10",
        "1<nonzero>",
        "20",
        "2<nonzero>",
        "30",
        "31",
    ],
    "<v>": ["<digit><digits>", "<v>.<digit><digits>"],
    "<int>": ["<nonzero><digits>", "0"],
    "<digits>": ["", "<digits><digit>"],
    "<nonzero>": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "<digit>": srange(string.digits),
}

assert is_valid_grammar(grammar)
