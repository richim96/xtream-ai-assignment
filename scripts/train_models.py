"""Automated script to train models with fresh data"""

import pandas as pd

from diamonds.ml_pipeline.data_extraction import extract_from_csv
from diamonds.ml_pipeline.data_preprocessing import (
    filter_numerical,
    dummy_encode,
    cols_drop,
)
from diamonds.ml_pipeline.model_training import *
