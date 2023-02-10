#!/usr/bin/env python
# coding: utf-8

# # Activity 1.1: Feature Extraction

# <div class="alert alert-success alertsuccess">
# [Task] Implement the functions <i>extract_existence</i>, <i>extract_numeric</i>, and <i>collect_features</i> to extract all possible features from a grammar and to parse each input file into its individual features.
# </div>

# ## Overview
# 
# In this module, we are concerned with the problem of extracting semantic features from inputs. In particular, Alhazen defines various features based on the input grammar, such as *existance* and *numeric interpretation*. These features are then extracted from the parse trees of the inputs (see [Section 3 of the paper](https://publications.cispa.saarland/3107/7/fse2020-alhazen.pdf) for more details).
# 
# The implementation of the feature extraction module consists of the following three tasks:
# 1. Implementation of individual feature classes, whose instances allow to derive specific feature values from inputs
# 2. Extraction of features from the grammar through instantiation of the aforementioned feature classes
# 3. Computation of feature vectors from a set of inputs, which will then be used as input for the decision tree

# <div class="alert alert-info">
# [Info]: For more information about parsing inputs with a grammar, we recommand to have a look at the chapters <a href="https://www.fuzzingbook.org/html/Grammars.html">Fuzzing with Grammars</a> and <a href="https://www.fuzzingbook.org/html/Parser.html">Parsing Inputs</a> of the fuzzingbook.
# </div>

# In[ ]:


# For type hints
from typing import Tuple, List, Optional, Any, Union, Set, Callable, Dict
DerivationTree = Tuple[str, Optional[List[Any]]]


# ## The calc grammar

# In[ ]:


# Custom Calculator Grammar from Kampmann et al. (See paper - with out regex)
# Lets load the grammar from the util-notebook
from alhazen.helper import CALC_GRAMMAR
from fuzzingbook.Grammars import Grammar

if __name__ == "__main__":
    for rule in CALC_GRAMMAR:
        print(rule.ljust(15), CALC_GRAMMAR[rule])


# ## Task 1: Implementing the feature classes

# In[ ]:


from abc import ABC, abstractmethod

class Feature(ABC):
    '''
    The abstract base class for grammar features.
    
    Args:
        name : A unique identifier name for this feature. Should not contain Whitespaces. 
               e.g., 'type(<feature>@1)'
        rule : The production rule (e.g., '<function>' or '<value>').
        key  : The feature key (e.g., the chosen alternative or rule itself).
    '''
    
    def __init__(self, name: str, rule: str, key: str) -> None:
        self.name = name
        self.rule = rule
        self.key = key
        super().__init__()
        
    def __repr__(self) -> str:
        '''Returns a printable string representation of the feature.'''
        return self.name_rep()
    
    @abstractmethod
    def name_rep(self) -> str:
        pass
    
    @abstractmethod
    def get_feature_value(self, derivation_tree) -> float:
        '''Returns the feature value for a given derivation tree of an input.'''
        pass


# Possible solution for the feature classes `ExistenceFeature` and `NumericInterpretation`:

# In[ ]:


from fuzzingbook.GrammarFuzzer import expansion_to_children

class ExistenceFeature(Feature):
    '''
    This class represents existence features of a grammar. Existence features indicate 
    whether a particular production rule was used in the derivation sequence of an input. 
    For a given production rule P -> A | B, a production existence feature for P and 
    alternative existence features for each alternative (i.e., A and B) are defined.
    
    name : A unique identifier name for this feature. Should not contain Whitespaces. 
           e.g., 'exist(<digit>@1)'
    rule : The production rule.
    key  : The feature key, equal to the rule attribute for production features, 
           or equal to the corresponding alternative for alternative features.
    '''
    def __init__(self, name: str, rule: str, key: str) -> None:
        super().__init__(name, rule, key)
    
    def name_rep(self) -> str:
        if self.rule == self.key:
            return f"exists({self.rule})"
        else:
            return f"exists({self.rule} == {self.key})"
    
    
    def get_feature_value(self, derivation_tree) -> float:
        '''Returns the feature value for a given derivation tree of an input.'''
        raise NotImplementedError
    

    def get_feature_value(self, derivation_tree: DerivationTree) -> float:
        '''Counts the number of times this feature was matched in the derivation tree.'''
        (node, children) = derivation_tree
        
        # The local match count (1 if the feature is matched for the current node, 0 if not)
        count = 0
        
        # First check if the current node can be matched with the rule
        if node == self.rule:
            
            # Production existance feature
            if self.rule == self.key:
                count = 1
            
            # Production alternative existance feature
            # We compare the children of the expansion with the actual children
            else:
                expansion_children = list(map(lambda x: x[0], expansion_to_children(self.key)))
                node_children = list(map(lambda x: x[0], children))
                if expansion_children == node_children:
                    count= 1
        
        # Recursively compute the counts for all children and return the sum for the whole tree
        for child in children:
            count = max(count, self.get_feature_value(child)) 
        
        return count


# In[ ]:


from fuzzingbook.GrammarFuzzer import tree_to_string
from numpy import nanmax, isnan

class NumericInterpretation(Feature):
    '''
    This class represents numeric interpretation features of a grammar. These features
    are defined for productions that only derive words composed of the characters
    [0-9], '.', and '-'. The returned feature value corresponds to the maximum
    floating-point number interpretation of the derived words of a production.

    name : A unique identifier name for this feature. Should not contain Whitespaces. 
           e.g., 'num(<integer>)'
    rule : The production rule.
    '''
    def __init__(self, name: str, rule: str) -> None:
        super().__init__(name, rule, rule)
    
    def name_rep(self) -> str:
        return f"num({self.key})"
    
    def get_feature_value(self, derivation_tree) -> float:
        '''Returns the feature value for a given derivation tree of an input.'''
        raise NotImplementedError
    
    def get_feature_value(self, derivation_tree: DerivationTree) -> float:
        '''Determines the maximum float of this feature in the derivation tree.'''
        (node, children) = derivation_tree
        
        value = float('nan')
        if node == self.rule:
            try:
                #print(self.name, float(tree_to_string(derivation_tree)))
                value = float(tree_to_string(derivation_tree))
            except ValueError:
                #print(self.name, float(tree_to_string(derivation_tree)), "err")
                pass
            
        # Return maximum value encountered in tree, ignoring all NaNs
        tree_values = [value] + [self.get_feature_value(c) for c in children]
        if all(isnan(tree_values)):
            return value
        else:
            return nanmax(tree_values)


# ## Task 2: Extracting the feature set from the grammar

# <div class="alert alert-success alertsuccess">
# [Task] Implement the function <i>extract_existence</i>. The function should extracts all existence features from the grammar and returns them as a list.
# </div>

# In[ ]:


def extract_existence(grammar: Grammar) -> List[ExistenceFeature]:
    '''
        Extracts all existence features from the grammar and returns them as a list.
        grammar : The input grammar.
    '''
    
    # Your code goes here
    raise NotImplementedError("Func. extract_existence: Function not Implemented")


# Possible solution for the function `extract_existence`:

# In[ ]:


def extract_existence(grammar: Grammar) -> List[ExistenceFeature]:
    '''
        Extracts all existence features from the grammar and returns them as a list.
        grammar : The input grammar.
    '''
    
    features = []
    
    for rule in grammar:
        # add the rule
        features.append(ExistenceFeature(f"exists({rule})", rule, rule))
        # add all alternatives
        for count, expansion in enumerate(grammar[rule]):
            features.append(ExistenceFeature(f"exists({rule}@{count})", rule, expansion))
    
    return features


# <div class="alert alert-success alertsuccess">
# [Task] Implement the function <i>extract_numeric</i>. The function should extract all numeric interpretation features from the grammar and returns them as a list.
# </div>

# In[ ]:


def extract_numeric(grammar: Grammar) -> List[NumericInterpretation]:
    '''
        Extracts all numeric interpretation features from the grammar and returns them as a list.
        
        grammar : The input grammar.
    '''
    
    # Your code goes here
    raise NotImplementedError("Func. extract_numeric: Function not Implemented")


# Possible solution for the function `extract_numeric`:

# In[ ]:


from fuzzingbook.Grammars import reachable_nonterminals
from collections import defaultdict
import re

# Regex for non-terminal symbols in expansions
RE_NONTERMINAL = re.compile(r'(<[^<> ]*>)')

def extract_numeric(grammar: Grammar) -> List[NumericInterpretation]:
    '''
        Extracts all numeric interpretation features from the grammar and returns them as a list.
        
        grammar : The input grammar.
    '''
    
    features = []
    
    # Mapping from non-terminals to derivable terminal chars
    derivable_chars = defaultdict(set)
    
    for rule in grammar:
        for expansion in grammar[rule]:
            
            # Remove non-terminal symbols and whitespace from expansion
            terminals = re.sub(RE_NONTERMINAL, '', expansion).replace(' ', '')
            
            # Add each terminal char to the set of derivable chars
            for c in terminals:
                derivable_chars[rule].add(c)
    
    # Repeatedly update the mapping until convergence
    while True:
        updated = False
        for rule in grammar:
            for r in reachable_nonterminals(grammar, rule):
                before = len(derivable_chars[rule])
                derivable_chars[rule].update(derivable_chars[r])
                after = len(derivable_chars[rule])
                
                # Set of derivable chars was updated
                if after > before:
                    updated = True
        
        if not updated:
            break
    
    numeric_chars = set(['0','1','2','3','4','5','6','7','8','9','.','-'])
    
    for key in derivable_chars:
        # Check if derivable chars contain only numeric chars
        if len(derivable_chars[key] - numeric_chars) == 0:
            features.append(NumericInterpretation(f"num({key})", key))
            
    return features


# <div class="alert alert-danger" role="alert">
# [Note] For the 'Feature.name' attribute, please use a unique identifier that does not contain any whitespaces. For instance, the identifier name can be something similar to 'exists(&lt;feature&gt;@1)' or 'exists(&lt;digit&gt;@0)'. @i corresponds to the i-te derivation alternative of a rule. 
# 
# For instance, exists(&lt;digit&gt;@0) correspondes to exists(&lt;digit&gt; == 0), or exists(&lt;feature&gt;@1) corresponds to exists(&lt;feature&gt; == tan). We use these identifier to speed up the parsing process. The nice representation of 'exists({self.rule} == {self.key})' is only used for us humans to give us easier to grasp explanations. For further information, please look at the gitlab issue #2.
# </div>

# ## Test 1: Confirm that we have extracted the right number of features

# In[ ]:


def get_all_features(grammar: Grammar) -> List[Feature]:
    return extract_existence(grammar) + extract_numeric(grammar)


# In[ ]:


def test_features(features: List[Feature]) -> None:
    existence_features = 0
    numeric_features = 0
    
    for feature in features:
        if isinstance(feature, ExistenceFeature):
            existence_features += 1
        elif isinstance(feature, NumericInterpretation):
            numeric_features += 1
            
    assert(existence_features == 27)
    assert(numeric_features == 4)
    
    expected_feature_names = ["exists(<start>)",
        "exists(<start> == <function>(<term>))",
        "exists(<function>)",
        "exists(<function> == sqrt)",
        "exists(<function> == tan)",
        "exists(<function> == cos)",
        "exists(<function> == sin)",
        "exists(<term>)",
        "exists(<term> == -<value>)",
        "exists(<term> == <value>)",
        "exists(<value>)",
        "exists(<value> == <integer>.<integer>)",
        "exists(<value> == <integer>)",
        "exists(<integer>)",
        "exists(<integer> == <digit><integer>)",
        "exists(<integer> == <digit>)",
        "exists(<digit>)",
        "exists(<digit> == 0)",
        "exists(<digit> == 1)",
        "exists(<digit> == 2)",
        "exists(<digit> == 3)",
        "exists(<digit> == 4)",
        "exists(<digit> == 5)",
        "exists(<digit> == 6)",
        "exists(<digit> == 7)",
        "exists(<digit> == 8)",
        "exists(<digit> == 9)",
        "num(<term>)",
        "num(<value>)",
        "num(<digit>)",
        "num(<integer>)"
    ]
    
    actual_feature_names = list(map(lambda f: f.name_rep(), features))
    
    for feature_name in expected_feature_names:
        assert (feature_name in actual_feature_names), f"Missing feature with name: {feature_name}"
        
    print("All checks passed!")


# In[ ]:


# Uncomment to execute test
if __name__ == "__main__":
    test_features(get_all_features(CALC_GRAMMAR))


# ## Task 3:  Extracting feature values from inputs

# <div class="alert alert-success alertsuccess">
# [Task] Implement the function <i>collect_features(sample_list, grammar)</i>. The function should parse all inputs from a list of samples into its features.
# </div>

# **INPUT**:
# the function requires the following parameter:
# - sample_list: a list of samples that should be parsed
# - grammar: the corresponding grammar of the syntactical features

# **OUTPUT**: the function should return a pandas Dataframe of the parsed features for all inputs in the sample list:

# | feature_1     | feature_2     | ...    |feature_n|
# | ------------- |-------------|-------------|-----|
# | 1     | 0 | ...| -900 |
# | 0     | 1 | ...| 20 |

# <div class="alert alert-info">
# [Hint]: It might be usefull to use the implement the abstract functions get_feature_value(self, derivation_tree) of the Feature class for each of the feature types (Existence, Numeric). Given a derivation tree, these functions return the value of the feature.
# </div>

# In[ ]:


from fuzzingbook.Parser import EarleyParser
from fuzzingbook.Grammars import Grammar
import pandas
from pandas import DataFrame

def collect_features(sample_list: List[str],
                     grammar: Grammar) -> DataFrame:
    
    # write your code here
    raise NotImplementedError("Func. collect_features: Function not Implemented")


# Possible solution for `collect_features`:
# 
# <div class="alert alert-danger" role="alert">
# [Note] This is a rather slow implementation, for many grammars with many syntactically features, the feature collection can be optimized
# </div>

# In[ ]:


from fuzzingbook.Parser import EarleyParser
from fuzzingbook.Grammars import Grammar
import pandas
from pandas import DataFrame

def collect_features(sample_list: List[str],
                     grammar: Grammar) -> DataFrame:
    
    data = []
    
    # parse grammar and extract features
    all_features = get_all_features(grammar)
    
    # iterate over all samples
    for sample in sample_list:
        parsed_features = {}
        parsed_features["sample"] = sample
        # initate dictionary
        for feature in all_features:
            parsed_features[feature.name] = 0
        
        # Obtain the parse tree for each input file
        earley = EarleyParser(grammar)
        for tree in earley.parse(sample):
            
            for feature in all_features:
                parsed_features[feature.name] = feature.get_feature_value(tree)
        
        data.append(parsed_features)
    
    return pandas.DataFrame.from_records(data)


# ## Test 2: Check whether we produce the correct feature values

# `TEST`

# In[ ]:


sample_list = ["sqrt(-900)", "sin(24)", "cos(-3.14)"]
df1 = collect_features(sample_list, CALC_GRAMMAR)

if __name__ == "__main__":
    display(df1)


# In[ ]:


# TODO: Implement max values for multiple parse trees
def get_feature_vector(sample, grammar, features):
    '''Returns the feature vector of the sample as a dictionary of feature values'''
    
    feature_dict = defaultdict(int)
    
    earley = EarleyParser(grammar)
    for tree in earley.parse(sample):
        for feature in features:
            feature_dict[feature.name] = feature.get_feature_value(tree)
    
    return feature_dict

from sklearn.feature_extraction import DictVectorizer
import pandas as pd

# Features for each input, one dict per input
feature_vectors = [get_feature_vector(sample, CALC_GRAMMAR, get_all_features(CALC_GRAMMAR)) for sample in sample_list]

# Transform to numpy array
vec = DictVectorizer()
X = vec.fit_transform(feature_vectors).toarray()

df2 = pd.DataFrame(X, columns = vec.get_feature_names_out())

# Check if both dataframes are equal by element-wise comparing each column
if __name__ == "__main__":
    assert all(map(lambda col: all(df1[col] == df2[col]), df2.head()))


# In[ ]:


# TODO: handle multiple trees
from fuzzingbook.Parser import EarleyParser

def compute_feature_values(sample: str, grammar: Grammar, features: List[Feature]) -> Dict[str, float]:
    '''
        Extracts all feature values from an input.
        
        sample   : The input.
        grammar  : The input grammar.
        features : The list of input features extracted from the grammar.
        
    '''
    earley = EarleyParser(CALC_GRAMMAR)
    
    features = {}
    for tree in earley.parse(sample):
        for feature in get_all_features(CALC_GRAMMAR):
            features[feature.name_rep()] = feature.get_feature_value(tree)
    return features


# In[ ]:


def test_feature_values() -> None:

    sample_list = ["sqrt(-900)", "sin(24)", "cos(-3.14)"]

    expected_feature_values = {
        "sqrt(-900)": {
            "exists(<start>)" : 1,
            "exists(<start> == <function>(<term>))" : 1,
            "exists(<function>)" : 1,
            "exists(<function> == sqrt)" : 1,
            "exists(<function> == tan)" : 0,
            "exists(<function> == cos)" : 0,
            "exists(<function> == sin)" : 0,
            "exists(<term>)" : 1,
            "exists(<term> == -<value>)" : 1,
            "exists(<term> == <value>)" : 0,
            "exists(<value>)" : 1,
            "exists(<value> == <integer>.<integer>)" : 0,
            "exists(<value> == <integer>)" : 1,
            "exists(<integer>)" : 1,
            "exists(<integer> == <digit><integer>)" : 1,
            "exists(<integer> == <digit>)" : 1,
            "exists(<digit>)" : 1,
            "exists(<digit> == 0)" : 1,
            "exists(<digit> == 1)" : 0,
            "exists(<digit> == 2)" : 0,
            "exists(<digit> == 3)" : 0,
            "exists(<digit> == 4)" : 0,
            "exists(<digit> == 5)" : 0,
            "exists(<digit> == 6)" : 0,
            "exists(<digit> == 7)" : 0,
            "exists(<digit> == 8)" : 0,
            "exists(<digit> == 9)" : 1,
            "num(<term>)" : -900.0,
            "num(<value>)" : 900.0,
            "num(<digit>)" : 9.0,
            "num(<integer>)" : 900.0
        }, 
        "sin(24)": {
            "exists(<start>)" : 1,
            "exists(<start> == <function>(<term>))" : 1,
            "exists(<function>)" : 1,
            "exists(<function> == sqrt)" : 0,
            "exists(<function> == tan)" : 0,
            "exists(<function> == cos)" : 0,
            "exists(<function> == sin)" : 1,
            "exists(<term>)" : 1,
            "exists(<term> == -<value>)" : 0,
            "exists(<term> == <value>)" : 1,
            "exists(<value>)" : 1,
            "exists(<value> == <integer>.<integer>)" : 0,
            "exists(<value> == <integer>)" : 1,
            "exists(<integer>)" : 1,
            "exists(<integer> == <digit><integer>)" : 1,
            "exists(<integer> == <digit>)" : 1,
            "exists(<digit>)" : 1,
            "exists(<digit> == 0)" : 0,
            "exists(<digit> == 1)" : 0,
            "exists(<digit> == 2)" : 1,
            "exists(<digit> == 3)" : 0,
            "exists(<digit> == 4)" : 1,
            "exists(<digit> == 5)" : 0,
            "exists(<digit> == 6)" : 0,
            "exists(<digit> == 7)" : 0,
            "exists(<digit> == 8)" : 0,
            "exists(<digit> == 9)" : 0,
            "num(<term>)" : 24.0,
            "num(<value>)" : 24.0,
            "num(<digit>)" : 4.0,
            "num(<integer>)" : 24.0
        },
        "cos(-3.14)": {
            "exists(<start>)" : 1,
            "exists(<start> == <function>(<term>))" : 1,
            "exists(<function>)" : 1,
            "exists(<function> == sqrt)" : 0,
            "exists(<function> == tan)" : 0,
            "exists(<function> == cos)" : 1,
            "exists(<function> == sin)" : 0,
            "exists(<term>)" : 1,
            "exists(<term> == -<value>)" : 1,
            "exists(<term> == <value>)" : 0,
            "exists(<value>)" : 1,
            "exists(<value> == <integer>.<integer>)" : 1,
            "exists(<value> == <integer>)" : 0,
            "exists(<integer>)" : 1,
            "exists(<integer> == <digit><integer>)" : 1,
            "exists(<integer> == <digit>)" : 1,
            "exists(<digit>)" : 1,
            "exists(<digit> == 0)" : 0,
            "exists(<digit> == 1)" : 1,
            "exists(<digit> == 2)" : 0,
            "exists(<digit> == 3)" : 1,
            "exists(<digit> == 4)" : 1,
            "exists(<digit> == 5)" : 0,
            "exists(<digit> == 6)" : 0,
            "exists(<digit> == 7)" : 0,
            "exists(<digit> == 8)" : 0,
            "exists(<digit> == 9)" : 0,
            "num(<term>)" : -3.14,
            "num(<value>)" : 3.14,
            "num(<digit>)" : 4.0,
            "num(<integer>)" : 14.0
        }
    }

    all_features = get_all_features(CALC_GRAMMAR)
    for sample in sample_list:
        input_features = compute_feature_values(sample, CALC_GRAMMAR, all_features)

        for feature in all_features:
            key = feature.name_rep()
            #print(f"\t{key.ljust(50)}: {input_features[key]}")
            #print('"' + key + '"' + ' : ' + str(input_features[key]) + ',')
            expected = expected_feature_values[sample][key]
            actual = input_features[key]
            assert (expected == actual), f"Wrong feature value for sample {sample} and feature {key}: expected {expected} but is {actual}."
            
    print("All checks passed!")


# In[ ]:


# Uncomment to execute test
if __name__ == "__main__":
    test_feature_values()

