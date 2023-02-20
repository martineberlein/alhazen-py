import sys
import logging

import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.derivation_tree import DerivationTree

from alhazen import Alhazen

from alhazen_formalizations.calculator import initial_inputs, grammar_alhazen as grammar, prop
from alhazen.requirementExtractionDT.treetools import remove_unequal_decisions
from alhazen.oracle import OracleResult
from alhazen.helper import show_tree
from alhazen.input import Input
from alhazen.feature_collector import Collector
from alhazen.features import EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE

MAX_ITERATION = 30


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(level=logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")
    # log.setLevel(logging.DEBUG)
    alhazen = Alhazen(
        initial_inputs=initial_inputs,
        grammar=grammar,
        evaluation_function=prop,
        max_iter=MAX_ITERATION,
        # generator=AdvancedGenerator(grammar)
    )
    trees = alhazen.run()

    collector = Collector(grammar=grammar, features={EXISTENCE_FEATURE, NUMERIC_INTERPRETATION_FEATURE})
    all_features = collector.get_all_features()

    show_tree(trees[MAX_ITERATION-1], all_features)
    show_tree(remove_unequal_decisions(trees[MAX_ITERATION - 1]), all_features)

    evaluation_data = set()
    g = GrammarFuzzer(grammar)
    for i in range(1000):
        inp = Input(DerivationTree.from_parse_tree(g.fuzz_tree()))
        evaluation_data.add(inp)

    evaluation_exec_data = alhazen._execute_input_files(evaluation_data)

    sample_bug_count = len(evaluation_exec_data[(evaluation_exec_data["oracle"].astype(str) == "BUG")])
    sample_count = len(evaluation_exec_data)

    print(f"{sample_bug_count} samples of {sample_count} generated inputs trigger the bug.")

    d = []
    for inp in evaluation_data:
        d.append(collector.collect_features(inp))

    eval_feature_data = pandas.DataFrame.from_records(data=d)

    # Clean up the evaluation data
    joined_data = evaluation_exec_data.join(eval_feature_data)

    # Only add valid data
    new_data = joined_data[(joined_data['oracle'] != OracleResult.UNDEF)]
    clean_data = joined_data.drop(joined_data[joined_data.oracle.astype(str) == "UNDEF"].index)
    clean_data = clean_data.fillna(0)

    eval_iteration = MAX_ITERATION - 1
    final_tree = remove_unequal_decisions(trees[eval_iteration])
    predictions = final_tree.predict(clean_data.drop(['oracle'], axis=1))

    accuracy = accuracy_score(clean_data['oracle'].astype(str), predictions, normalize=True)
    accuracy = round(accuracy * 100, 3)
    print(f"The decision tree at iteration {str(eval_iteration)} achieved an accuracy of {accuracy} %")

    precision = precision_score(clean_data['oracle'].astype(str), predictions, pos_label='BUG', average='binary')
    precision = round(precision * 100, 3)
    recall = recall_score(clean_data['oracle'].astype(str), predictions, pos_label='BUG', average='binary')
    recall = round(recall * 100, 3)

    print(f"The decision tree at iteration {str(eval_iteration)} achieved a precision of {precision} %")
    print(f"The decision tree at iteration {str(eval_iteration)} achieved a recall of {recall} %")

    f1 = f1_score(clean_data['oracle'].astype(str), predictions, pos_label='BUG', average='binary')
    print(f"The decision tree at iteration {str(eval_iteration)} achieved a f1-score of {round(f1, 3)}")
