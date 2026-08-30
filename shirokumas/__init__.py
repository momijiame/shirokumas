from importlib.metadata import version

from ._agg import AggregateEncoder
from ._binarize import MultiLabelBinarizer
from ._binarize import OneHotEncoder
from ._count import CountEncoder
from ._null import NullEncoder
from ._oof import OutOfFoldEncodeWrapper
from ._ordinal import OrdinalEncoder
from ._target import TargetEncoder

__version__ = version("shirokumas")
__all__ = [
    "AggregateEncoder",
    "CountEncoder",
    "MultiLabelBinarizer",
    "NullEncoder",
    "OneHotEncoder",
    "OrdinalEncoder",
    "OutOfFoldEncodeWrapper",
    "TargetEncoder",
]
