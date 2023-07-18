The Feature Collector Class
---------------------------

.. autoclass:: alhazen.feature_collector.Collector

Main Functions
^^^^^^^^^^^^^^

This method gives back all features defined by the grammar.
You can find in :doc:`features` what kind of features are implemented.

.. automethod:: alhazen.feature_collector.Collector.get_all_features

   -> List of Features as Dict{feature.name:feature.value}

.. literalinclude:: ../../src/alhazen/feature_collector.py
   :pyobject: Collector.get_all_features
   :caption:  Colletcts all the grammar based features.

.. automethod:: alhazen.feature_collector.Collector.collect_features_from_list
.. literalinclude:: ../../src/alhazen/feature_collector.py
   :pyobject: Collector.collect_features_from_list
   :caption:  Description: Wrapper for collect features method

.. automethod:: alhazen.feature_collector.Collector.collect_features
.. literalinclude:: ../../src/alhazen/feature_collector.py
   :pyobject: Collector.collect_features
   :caption: Description: Collects Features from a Test Input.

.. automethod:: alhazen.feature_collector.Collector.feature_collection
.. literalinclude:: ../../src/alhazen/feature_collector.py
   :pyobject: Collector.feature_collection
   :caption: Description: Gets all one and two dimensional features.

