********************************************************************************
TargetEncoder
********************************************************************************

This implements an encoding method called Target/Likelihood (Mean) Encoding.
It is a supervised method that uses the objective variable for feature extraction.

Therefore, not only explanatory variables but also objective variables should be prepared.

.. literalinclude:: ../../sources/tutorial/target.txt
  :language: python
  :start-after: <prepare-dataframe>
  :end-before: </prepare-dataframe>

In addition, Out-of-Fold (OOF) feature extraction is required to reduce data leakage on cross validation.
Use scikit-learn's BaseCrossValidator subclass to define how to split data.

.. literalinclude:: ../../sources/tutorial/target.txt
  :language: python
  :start-after: <prepare-folds>
  :end-before: </prepare-folds>

In the following, four rows of training data are divided into four parts.
In turn, three of the training data rows are used to compute the target statistics, and one row is replaced with the computed target statistics.

.. literalinclude:: ../../sources/tutorial/target.txt
  :language: python
  :start-after: <fit-and-transform>
  :end-before: </fit-and-transform>

Unknown value will be replaced to global mean by default.
In the following, ``cherry`` is replaced by ``3 / 4 = 0.75`` because it is an unknown value

.. literalinclude:: ../../sources/tutorial/target.txt
  :language: python
  :start-after: <transform-test-data>
  :end-before: </transform-test-data>

Smoothing is available to reduce over-fitting.
Shirokumas implements two types of smoothing: Empirical Bayesian and M-probability Estimate [#]_ .

.. [#]
  https://dl.acm.org/doi/10.1145/507533.507538

To use Empirical Bayesian, you can set ``"eb"`` for ``smoothing_method`` argument.
The parameters ``"k"`` and ``"f"`` are used to regulate the smoothing.
If not specified, ``20`` and ``10`` are used, respectively by default.
``k`` and ``f`` are equivalent to ``min_samples_leaf`` and ``smoothing`` in ``TargetEncoder`` of category_encoders [#]_.

.. [#]
  https://contrib.scikit-learn.org/category_encoders/targetencoder.html

.. literalinclude:: ../../sources/tutorial/target.txt
  :language: python
  :start-after: <empirical-bayesian>
  :end-before: </empirical-bayesian>

To use M-probability Estimate, you can set ``"m-estimate"`` for ``smoothing_method`` argument.
The parameter ``"m"`` is used to regulate the smoothing.
If not specified, ``1.0`` is used by default.
``m`` is equivalent to ``m`` in ``MEstimateEncoder`` of category_encoders [#]_.

.. [#]
  https://contrib.scikit-learn.org/category_encoders/mestimate.html

.. literalinclude:: ../../sources/tutorial/target.txt
  :language: python
  :start-after: <m-estimate>
  :end-before: </m-estimate>