#!/usr/bin/env python
# coding: utf-8
from IPython.core.display_functions import display

display_output = False

from alhazen.helper import CALC_GRAMMAR

for rule in CALC_GRAMMAR:
    print(rule.ljust(15), CALC_GRAMMAR[rule])


# feature extraction
from alhazen.Activity1_1_FeatureExtraction import extract_existence, extract_numeric, collect_features
from alhazen.Activity1_2_GrammarTransformation import transform_grammar
from alhazen.Activity2_DecisionTreeLearner import train_tree
from alhazen.Activity4_GenerateSamples import generate_samples_random as generate_samples
from alhazen.Activity3_RequirementExtraction import get_all_input_specifications
from alhazen.Activity5_ExecuteSamples import execute_samples

from typing import List, Callable
import pandas

from fuzzingbook.Grammars import Grammar
from alhazen.helper import OracleResult
from alhazen.helper import show_tree

GENERATOR_TIMEOUT = 10  # timeout in seconds


class Alhazen:

    def __init__(
            self, initial_inputs: List[str],
            grammar: Grammar,
            evaluation_function: Callable,
            max_iter: int = 10,
            generator_timeout: int = 10
    ):

        self._initial_inputs = initial_inputs
        self._grammar = grammar
        self._prop = evaluation_function
        self._max_iter = max_iter
        self._previous_samples = None
        self._data = None
        self._trees = []
        self._generator_timeout = generator_timeout
        self._setup()

    def _setup(self):
        self._previous_samples = self._initial_inputs

        self._all_features = extract_existence(self._grammar) + extract_numeric(self._grammar)
        self._feature_names = [f.name for f in self._all_features]

    def run(self):
        raise NotImplementedError()

    def _add_new_data(self, exec_data, feature_data):
        joined_data = exec_data.join(feature_data.drop(['sample'], axis=1))

        # Only add valid data
        new_data = joined_data[(joined_data['oracle'] != OracleResult.UNDEF)]
        new_data = joined_data.drop(joined_data[joined_data.oracle.astype(str) == "UNDEF"].index)
        if 0 != len(new_data):
            if self._data is None:
                self._data = new_data
            else:
                self._data = pandas.concat([self._data, new_data], sort=False)

    def _finalize(self):
        return self._trees


# In[ ]:


class Alhazen(Alhazen):

    def run(self) -> List:
        for iteration in range(self._max_iter):
            print(f"Starting Iteration: " + str(iteration))
            self._loop(self._previous_samples)

        return self._finalize()

    def _loop(self, sample_list):
        # obtain labels, execute samples (Initial Step, Activity 5)
        exec_data = execute_samples(sample_list, self._prop)

        # collect features from the new samples (Activity 1)
        feature_data = collect_features(sample_list, self._grammar)

        # combine the new data with the already existing data
        self._add_new_data(exec_data, feature_data)

        # train a tree (Activity 2)
        dec_tree = train_tree(self._data)
        self._trees.append(dec_tree)

        # extract new requirements from the tree (Activity 3)
        new_input_specifications = get_all_input_specifications(dec_tree,
                                                                self._all_features,
                                                                self._feature_names,
                                                                self._data.drop(['oracle'], axis=1))

        # generate new inputs according to the new input specifications
        # (Activity 4)
        new_samples = generate_samples(self._grammar, new_input_specifications, self._generator_timeout)
        self._previous_samples = new_samples


MAX_ITERATION = 20

# let's initialize Alhazen
# let's use the previously used sample_list (['sqrt(-16)', 'sqrt(4)'])

if __name__ == '__main__':

    sample_list = ['sqrt(-16)', 'sqrt(4)']
    alhazen = Alhazen(sample_list, CALC_GRAMMAR, MAX_ITERATION, GENERATOR_TIMEOUT)

    # and run it
    # Alhazen returns a list of all the iteratively learned decision trees
    trees = alhazen.run()

    #
    # </hr>
    #
    # Let's display the final decision tree learned by Alhazen. You can use the function `show_tree(decison_tree, features)` to display the final tree.

    # In[ ]:

    from alhazen.requirementExtractionDT.treetools import remove_unequal_decisions

    all_features = extract_existence(CALC_GRAMMAR) + extract_numeric(CALC_GRAMMAR)
    # show_tree(trees[MAX_ITERATION-1], all_features)

    # <div class="alert alert-info" role="alert">
    # [Info] The decision tree may contain unneccesary long paths, where the bug-class does not change. You can use the function 'remove_unequal_decisions(decision_tree)' to remove those nodes.
    # </div>

    # In[ ]:

    from alhazen.requirementExtractionDT.treetools import remove_unequal_decisions

    show_tree(remove_unequal_decisions(trees[MAX_ITERATION - 1]), all_features)

    # You should now be able to identify the features that are responsible for the caluclator's failue!

    # `Real Solution`: The failure occurs whenever the function 'sqrt(x)' is used and x is between '-12' and '-42'

    # # Evaluation

    # <hr/>
    # Let's evaluate the learned classification model! We judge the quality of the learned decision tree learner by assessing its capabilities of predicting the behavior of newly generated inputs.

    # ## Evaluation Setup (Generating an Evaluation Data Set)

    # In the first step of evaluation of the learned classifier, we generate a evaluation data set.

    # In[ ]:

    # We import the GrammarFuzzer
    from fuzzingbook.GrammarFuzzer import GrammarFuzzer

    evaluation_data = []

    # And generate 1000 input samples
    g = GrammarFuzzer(CALC_GRAMMAR)
    for i in range(1000):
        evaluation_data.append(str(g.fuzz()))

    # In[ ]:

    # Lets obtain the actuall program behavior of the evaluation data ['BUG', 'NO_BUG']
    evaluation_exec_data = execute_samples(evaluation_data)
    print(evaluation_exec_data)

    # Is the data set imbalanced?
    sample_bug_count = len(evaluation_exec_data[(evaluation_exec_data["oracle"].astype(str) == "BUG")])
    sample_count = len(evaluation_exec_data)

    print(f"{sample_bug_count} samples of {sample_count} generated inputs trigger the bug.")

    # In[ ]:

    # let us obtain the features from the generated inputs
    eval_feature_data = collect_features(evaluation_data, CALC_GRAMMAR)
    # display(eval_feature_data)

    # In[ ]:

    # Clean up the evaluation data
    joined_data = evaluation_exec_data.join(eval_feature_data.drop(['sample'], axis=1))

    # Only add valid data
    new_data = joined_data[(joined_data['oracle'] != OracleResult.UNDEF)]
    clean_data = joined_data.drop(joined_data[joined_data.oracle.astype(str) == "UNDEF"].index)
    # display(clean_data)

    # ## Evaluation Results
    #
    # <hr/>
    # Let's use the generated evaluation set to measure the accuracy, precision, recall and f1-score of your learned machine learning model.

    # <div class="alert alert-info" role="alert">
    # [Info] We use <a href="https://scikit-learn.org/stable/">scikit-learn</a> to evalute the classifier.
    # </div>

    # In[ ]:

    eval_iteration = MAX_ITERATION - 1
    final_tree = remove_unequal_decisions(trees[eval_iteration])

    # In[ ]:

    # We use the final decision tree to predict the behavior of the evaluation data set.
    predictions = final_tree.predict(clean_data.drop(['oracle'], axis=1))

    # Let's measure the accuracy of the learned decision tree

    # <div class="alert alert-info" role="alert">
    # [Info] We start by measuering the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html">accuracy</a> of the classifier.
    # </div>

    # In[ ]:

    # We calculate the accuracy by comparing how many predictions match the actual program behavior
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(clean_data['oracle'].astype(str), predictions, normalize=True)
    # we round the accuracy to three digits
    accuracy = round(accuracy * 100, 3)
    print(f"The decison tree at iteration {str(eval_iteration)} achieved an accuracy of {accuracy} %")

    # <div class="alert alert-info" role="alert">
    # [Info] We use the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html">precision-score</a> and the <a href="https://scikit-learn.org/stable/">recall-score</a>.
    # </div>

    # In[ ]:

    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score

    precision = precision_score(clean_data['oracle'].astype(str), predictions, pos_label='BUG', average='binary')
    precision = round(precision * 100, 3)
    recall = recall_score(clean_data['oracle'].astype(str), predictions, pos_label='BUG', average='binary')
    recall = round(recall * 100, 3)

    print(f"The decison tree at iteration {str(eval_iteration)} achieved a precision of {precision} %")
    print(f"The decison tree at iteration {str(eval_iteration)} achieved a recall of {recall} %")

    # <div class="alert alert-info" role="alert">
    # [Info] To counteract the imbalanced data set, we use the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html">F1-score</a>.
    # </div>

    # In[ ]:

    from sklearn.metrics import f1_score

    f1 = f1_score(clean_data['oracle'].astype(str), predictions, pos_label='BUG', average='binary')
    print(f"The decison tree at iteration {str(eval_iteration)} achieved a f1-score of {round(f1, 3)}")

    # ## Congratulations

    # You did it, congratulations!

    # In[ ]:
