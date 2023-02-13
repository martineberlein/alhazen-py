#!/usr/bin/env python
# coding: utf-8

# # Helper Functions

# These functions are used throughout the tool.

# ### Custom Calculator Grammar

# In[1]:


# Load Grammar
from fuzzingbook.Grammars import Grammar

# Custom Calculator Grammar from Kampmann et al. (See paper - with out regex)
CALC_GRAMMAR: Grammar = {
    "<start>": ["<function>(<term>)"],
    "<function>": ["sqrt", "tan", "cos", "sin"],
    "<term>": ["-<value>", "<value>"],
    "<value>": ["<integer>.<integer>", "<integer>"],
    "<integer>": ["<digit><integer>", "<digit>"],
    "<digit>": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
}
START_SYMBOL = "<start>"


# ### Oracle/Label Definition

# In[2]:


from enum import Enum


class OracleResult(Enum):
    BUG = "BUG"
    NO_BUG = "NO_BUG"
    UNDEF = "UNDEF"

    def __str__(self):
        return self.value


# In[3]:


from typing import List, Dict


def read_files(sample_list: List[str]):
    data = []
    for file in sample_list:
        f = open(file, "r")
        data.append(str(f.read()))

    return data


# ### Show Decision Tree

# In[4]:


import graphviz
from sklearn import tree


def show_tree(clf, feature_names):
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=["BUG", "NO_BUG"],
        filled=True,
        rounded=True,
    )
    return graphviz.Source(dot_data)


def get_dot_data(clf, feature_names):
    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=["BUG", "NO_BUG"],
        filled=True,
        rounded=True,
    )
    return dot_data


# In[ ]:
