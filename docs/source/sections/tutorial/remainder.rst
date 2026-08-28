********************************************************************************
Handling of the other columns
********************************************************************************

Every encoder accepts a ``remainder`` option.
This decides what happens to the columns that are not encoded.

Prepare a sample DataFrame.
This contains a column named "fruits" of a category variable, and a column named "price" that is not a category variable.

.. literalinclude:: ../../sources/tutorial/remainder.txt
  :language: python
  :start-after: <prepare-dataframe>
  :end-before: </prepare-dataframe>

By default, ``"drop"`` is used and the output contains only the encoded columns.

.. literalinclude:: ../../sources/tutorial/remainder.txt
  :language: python
  :start-after: <default-drop>
  :end-before: </default-drop>

If you want to keep the other columns, specify ``"passthrough"``.
The encoded columns come first, and the remaining columns follow in their original order.

.. literalinclude:: ../../sources/tutorial/remainder.txt
  :language: python
  :start-after: <passthrough>
  :end-before: </passthrough>

This also works for the encoders that generate new column names.

.. literalinclude:: ../../sources/tutorial/remainder.txt
  :language: python
  :start-after: <passthrough-generated-columns>
  :end-before: </passthrough-generated-columns>

Note that ``"passthrough"`` requires the DataFrame passed to ``transform()`` to have the same columns as the one used for ``fit()``.
Otherwise the encoder would silently produce a different set of columns for the training data and the test data.
