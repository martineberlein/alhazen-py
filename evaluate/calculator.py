import sys
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from fuzzingbook.GrammarFuzzer import GrammarFuzzer
from isla.derivation_tree import DerivationTree

from alhazen import Alhazen

from alhazen_formalizations.calculator import initial_inputs, grammar, prop
from alhazen.requirementExtractionDT.treetools import remove_unequal_decisions
from alhazen.features import extract_existence, extract_numeric, collect_features
from alhazen.helper import show_tree, OracleResult

MAX_ITERATION = 30


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")

    alhazen = Alhazen(
        initial_inputs=initial_inputs,
        grammar=grammar,
        evaluation_function=prop,
        max_iter=MAX_ITERATION
    )
    trees = alhazen.run()

    all_features = extract_existence(grammar) + extract_numeric(grammar)

    show_tree(trees[MAX_ITERATION-1], all_features)
    show_tree(remove_unequal_decisions(trees[MAX_ITERATION - 1]), all_features)

    evaluation_data = []
    g = GrammarFuzzer(grammar)
    for i in range(1000):
        evaluation_data.append(DerivationTree.from_parse_tree(g.fuzz_tree()))

    evaluation_exec_data = alhazen._execute_input_files(evaluation_data)

    sample_bug_count = len(evaluation_exec_data[(evaluation_exec_data["oracle"].astype(str) == "BUG")])
    sample_count = len(evaluation_exec_data)

    print(f"{sample_bug_count} samples of {sample_count} generated inputs trigger the bug.")

    eval_feature_data = collect_features(evaluation_data, grammar)

    # Clean up the evaluation data
    joined_data = evaluation_exec_data.join(eval_feature_data.drop(['sample'], axis=1))

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
