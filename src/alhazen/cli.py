import argparse
import sys
from datetime import datetime
from importlib.metadata import version
import importlib.util

from alhazen import Alhazen


def main():

    parser = argparse.ArgumentParser(description="Alhazen (v%s) is a debugging tool, that tries to automatically learn "
                                                 "associations between program failures in connection to their "
                                                 "inputs. Results are be printed to stdout. You can change the output "
                                                 "format."
                                                 % (version("alhazen-py")),
                                     epilog="To find out more, check out the repository at "
                                            "https://github.com/martineberlein/alhazen-py .")

    parser.add_argument("program",
                        help="path to a python file under test, currently only one source file is supported by the CLI",
                        type=str)

    # optional arguments, sorted alphabetically
    parser.add_argument("-e", "--evaluation_function",
                        help="specify the name of the method under test (default = evaluation_function)",
                        type=str,
                        metavar="EVAL_FUNC",
                        default="evaluation_function")
    parser.add_argument("-g", "--grammar",
                        help="specify the name of the grammar to be used (default = grammar)",
                        type=str,
                        default="grammar")
    parser.add_argument("-i", "--initial_inputs",
                        help="specify the name of the initial inputs to be used (default initial_inputs)",
                        type=str,
                        metavar="INIT_INPUTS",
                        default="initial_inputs")

    # more optional arguments, sorted alphabetically
    # parser.add_argument("-f", "--features") TODO
    parser.add_argument("-F", "--format",
                        type=int,
                        default=0,
                        help="define the output form to stdout (0: clf text, 1: dot model)")
    # parser.add_argument("-G", "--generator") TODO
    parser.add_argument("-t", "--generator_timeout",
                        type=int,
                        metavar="GEN_TIMEOUT",
                        default=10)
    # parser.add_argument("-l", "--learner") TODO
    parser.add_argument("-m", "--max_iter",
                        type=int,
                        default=10)
    parser.add_argument("-v", "--verbosity",
                        action="count",
                        default=0,
                        help="increase output verbosity")
    parser.add_argument("-V", "--version",
                        action="version",
                        version="alhazen-py %s" % (version("alhazen-py")))

    args = parser.parse_args()

    module_name = args.program
    file_path = args.program

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if args.verbosity >= 1:
        print("Starting up Alhazen with %s." % args.program)
        print("Started at %s with the following configuration:" % str(datetime.now()))
        print("evaluation_function: %s, grammar: %s, initial_inputs: %s"
              % (args.evaluation_function, args.grammar, args.initial_inputs))

    alhazen = Alhazen(initial_inputs=getattr(module, args.initial_inputs),
                      grammar=getattr(module, args.grammar),
                      evaluation_function=getattr(module, args.evaluation_function))

    alhazen.run()

    if args.verbosity >= 1:
        print("\nResult start here:\n")

    if args.format == 0:
        print(alhazen.get_clf_model())
    elif args.format == 1:
        print(alhazen.get_dot_model())


if __name__ == "__main__":
    exit(main())
