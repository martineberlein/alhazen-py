#!/usr/bin/env python
# coding: utf-8
from IPython.core.display_functions import display

# # Executing Inpt Files
#
# This notebook contains the functionallity to execute the calculator subject with a list of input samples.

# <div class="alert alert-info">
# [Info]: This notebook does not contain any tasks for you, however, it can be usefull to get a better understanding of the 'execute_samples(sample_list)' function.
# </div>
# <hr/>

# We start by importing the oracle/label definition from the utility notebook:

# In[ ]:


from alhazen.helper import OracleResult

# Next we implement the function `sample_runner(sample)` that lets us execute the calculator for a single sample. `sample_runner(sample)` returns the, in the pervious step imported, `OracleResult` for the sample.

# In[ ]:


import pandas

# load the calculator as task
import alhazen.calculator.task as task

SUBJECT = "calculator"


def sample_runner(sample):
    testcode = sample

    try:
        exec(
            testcode,
            {"sqrt": task.sqrt, "tan": task.tan, "sin": task.sin, "cos": task.cos},
            {},
        )
        return OracleResult.NO_BUG
    except ZeroDivisionError:
        return OracleResult.BUG
    except:
        return OracleResult.UNDEF


# Let's test the function:

# In[ ]:


sample = "sqrt(-16)"
sample_runner(sample)

# As expected, the sample `sqrt(-16)` triggers the calculator bug. Let's try some more samples:

# In[ ]:


assert sample_runner("sqrt(-23)") == OracleResult.BUG
assert sample_runner("sqrt(44)") == OracleResult.NO_BUG
assert sample_runner("cos(-9)") == OracleResult.NO_BUG

# <hr/>
# What happens if we parse inputs to calculator, that do not conform to its input format?

# In[ ]:


sample_runner("undef_function(QUERY)")

# The function `sample_runner(sample)` returns an `OracleResult.UNDEF` whenever the runner is not able to execute the sample.

# <div class="alert alert-danger" role="alert">
# To work reliably, you have to remove all samples from the learning set of Alhazen that do not conform to the grammar.
# </div>

# <hr/>
# The finally we provide the function 'execute_samples(sample_list)' that obtians the oracle/label for a list of samples.

# In[ ]:

from typing import Callable
import uuid

from fuzzingbook.Parser import DerivationTree, tree_to_string


def prop_(prop, sample: DerivationTree):
    result = prop(tree_to_string(sample)) if isinstance(sample, DerivationTree) else prop(sample)
    # result = prop(tree_to_string(sample))
    if isinstance(result, bool):
        if result:
            return OracleResult.BUG
        return OracleResult.NO_BUG
    else:
        return OracleResult.UNDEF


# executes a list of samples and return the execution outcome (label)
# the functions returns a pandas dataframe
def execute_samples(sample_list, prop: Callable = None):
    data = []
    for sample in sample_list:
        id = uuid.uuid1()
        if prop == None:
            result = sample_runner(sample)
        else:
            result = prop_(prop=prop, sample=sample)
        data.append(
            {
                # "sample_id": id.hex,
                # "sample": sample,
                # "subject": SUBJECT,
                "oracle": result
            }
        )
    return pandas.DataFrame.from_records(data)


# In[ ]:


# let us define a list of samples to execute
sample_list = ["sqrt(-20)", "cos(2)", "sqrt(-100)", "undef_function(foo)"]

# In[ ]:


# we obtain the execution outcome
labels = execute_samples(sample_list, None)
display(labels)

# In[ ]:


# combine with the sample_list
for i, row in enumerate(labels["oracle"]):
    print(sample_list[i].ljust(30) + str(row))

# To remove the undefined input samples, you could invoke something similar to this:

# In[ ]:


# clean up data
clean_data = labels.drop(labels[labels.oracle.astype(str) == "UNDEF"].index)
display(clean_data)
