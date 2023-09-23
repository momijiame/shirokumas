********************************************************************************
OneHotEncoder
********************************************************************************

This implements an encoding method called One-hot Encoding.
This encoding method creates as many columns as there are unique values, each with a true/false indicator representing whether a categorical variable belongs to a specific class.

.. literalinclude:: ../../sources/tutorial/binarize.txt
  :language: python
  :start-after: <one-hot-prepare-dataframe-and-fit-transform>
  :end-before: </one-hot-prepare-dataframe-and-fit-transform>

********************************************************************************
MultiLabelBinarizer
********************************************************************************

This implements an encoding method called Multi-hot Encoding.
This encoding method is designed for scenarios where each instance can belong to multiple classes.
In this method, multiple columns can have a true value for a single instance.

.. literalinclude:: ../../sources/tutorial/binarize.txt
  :language: python
  :start-after: <multi-hot-prepare-dataframe-and-fit-transform>
  :end-before: </multi-hot-prepare-dataframe-and-fit-transform>
