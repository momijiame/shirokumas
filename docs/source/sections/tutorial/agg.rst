********************************************************************************
AggregateEncoder
********************************************************************************

This implements an encoding method that makes an aggregate feature.
It is a bit similar to Target Encoding, but it is an unsupervised feature extraction method.

.. literalinclude:: ../../sources/tutorial/agg.txt
  :language: python
  :start-after: <prepare-dataframe>
  :end-before: </prepare-dataframe>

The encoder requires ``cols`` and ``agg_exprs`` arguments.
``cols`` is specified the columns to be grouped.
``agg_exprs`` is specified the expressions used to br aggregated.
``agg_exprs`` is a dictionary.
The dictionary keys are used for the suffix of the column names.

.. literalinclude:: ../../sources/tutorial/agg.txt
  :language: python
  :start-after: <instantiate-encoder>
  :end-before: </instantiate-encoder>

The mean and maximum values for each category are encoded.

.. literalinclude:: ../../sources/tutorial/agg.txt
  :language: python
  :start-after: <fit-transform>
  :end-before: </fit-transform>
