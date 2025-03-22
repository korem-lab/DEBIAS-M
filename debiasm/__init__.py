__version__ = "0.0.2"

import warnings
warnings.filterwarnings("ignore",
                        ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", '.*In the future*')
from .sklearn_functions import DebiasMClassifier, OnlineDebiasMClassifier, DebiasMClassifierLogAdd
from .dm_regression import DebiasMRegressor
from .dmc_multitask_regression import MultitaskDebiasMRegressor
from .multitask import MultitaskDebiasMClassifier
