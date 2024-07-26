"""Automated script to train models with fresh data"""

import pandas as pd

from xtream.ml_pipeline.data_extraction import extract_from_csv
from xtream.ml_pipeline.preprocessing import (
    dummy_encode,
    filter_numeric,
)
from xtream.ml_pipeline.training import linear_regression_train
