********************************************************************************
OrdinalEncoder
********************************************************************************

This implements an encoding method called Ordinal/Label Encoding.
Specifically, it maps a specific class of categorical variables to a specific integer.

Prepare a sample DataFrame.
This contains a column named "fruits" of a category variable.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <prepare-dataframe>
  :end-before: </prepare-dataframe>

Instantiate the encoder.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <instantiate-encoder>
  :end-before: </instantiate-encoder>

Then fit and transform the DataFrame.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <fit-transform>
  :end-before: </fit-transform>

Since the mapping is saved in the instance, test data can also be transformed.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <transform-test>
  :end-before: </transform-test>

If you want to give explicit mappings, specify ``mapping`` option.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <mappings-option>
  :end-before: </mappings-option>

By default, unknown values are replaced to -1 and missing values to -2.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <none-and-unknown>
  :end-before: </none-and-unknown>

If you want to transform specific columns, specify ``cols`` option.

.. literalinclude:: ../../sources/tutorial/ordinal.txt
  :language: python
  :start-after: <cols>
  :end-before: </cols>
