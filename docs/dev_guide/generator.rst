The Generator Class
-------------------

.. autoclass:: alhazen.generator.SimpleGenerator
.. autoclass:: alhazen.generator.AdvancedGenerator
.. autoclass:: alhazen.generator.ISLAGenerator

    In development feature to transform Decision trees into First order logic

Main Functions
^^^^^^^^^^^^^^

.. autofunction:: alhazen.generator.best_trees
.. literalinclude:: ../../src/alhazen/generator.py
   :pyobject: best_trees
   :language: python
   :caption: Description: Selects best Decision tree.

.. autofunction:: alhazen.generator.generate_samples_advanced
.. literalinclude:: ../../src/alhazen/generator.py
   :pyobject: generate_samples_advanced
   :language: python
   :caption: Description: Generating samples

.. autofunction:: alhazen.generator.generate_samples_random

   -> List of DerivationTree

.. literalinclude:: ../../src/alhazen/generator.py
   :pyobject: generate_samples_random
   :language: python
   :caption: Description: Generates samples randomly