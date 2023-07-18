The Learner Class
-----------------

.. autoclass:: alhazen.learner.Learner
.. autoclass:: alhazen.learner.DecisionTreeLearner
.. autoclass:: alhazen.learner.RandomForestLearner
.. autoclass:: alhazen.learner.XGBTLearner

   in development

Main Functions of all Learners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: alhazen.learner.Learner.train
.. literalinclude:: ../../src/alhazen/learner.py
   :pyobject: Learner.train
   :caption: trains model on data

.. automethod:: alhazen.learner.Learner.get_input_specifications
.. literalinclude:: ../../src/alhazen/learner.py
   :pyobject: Learner.get_input_specifications

.. automethod:: alhazen.learner.Learner.visualize
.. literalinclude:: ../../src/alhazen/learner.py
   :pyobject: Learner.visualize

.. automethod:: alhazen.learner.Learner.predict
.. literalinclude:: ../../src/alhazen/learner.py
   :pyobject: Learner.predict

.. automethod:: alhazen.learner.DecisionTreeLearner.train
.. literalinclude:: ../../src/alhazen/learner.py
   :pyobject: DecisionTreeLearner.train
   :caption: Description: trains model on data

.. automethod:: alhazen.learner.DecisionTreeLearner.get_input_specifications
.. literalinclude:: ../../src/alhazen/learner.py
   :pyobject: DecisionTreeLearner.get_input_specifications
   :caption: Description: gets input specification

