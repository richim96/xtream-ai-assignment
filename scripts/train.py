"""Automated script to train models with fresh data"""

import pandas as pd

from service.ml_pipeline.data_extraction import extract_from_csv
from service.ml_pipeline.preprocessing import (
    dummy_encode,
    filter_numeric,
)
from service.ml_pipeline.training import linear_model_train
