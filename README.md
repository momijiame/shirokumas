# Shirokumas

A set of scikit-learn style transformers for Polars.
The transformers have the following property:

- Support for polars DataFrame as an input and output
- Can explicitly configure which columns will be encoded

## How to install

```sh
$ pip install git+https://github.com/momijiame/shirokumas.git
```

## How to use

```python
import shirokumas as sk

encoder = sk.AggregateEncoder(...)
encoder = sk.CountEncoder(...)
encoder = sk.NullEncoder(...)
encoder = sk.OneHotEncoder(...)
encoder = sk.OrdinalEncoder(...)
encoder = sk.TargetEncoder(...)

train_x, train_y, test_x = ...

encoder.fit(train_x, train_y)
encoded_train_x = encoder.transform(train_x, train_y)
encoded_test_x = encoder.transform(test_x)
```
