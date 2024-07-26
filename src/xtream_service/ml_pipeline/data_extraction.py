"""Module for the data extraction workflow"""

from pathlib import Path
from typing import IO

import pandas as pd

from xtream_service.ml_pipeline import LOGGER


def extract_from_csv(source: str | Path | IO[str] | IO[bytes] | bytes) -> pd.DataFrame:
    """Load csv source data into a pandas dataframe.

    Parameters
    ----------
    source : str | Path | IO[str] | IO[bytes] | bytes
        Path to a file or a file-like object. URLs are also accepted. For file-like
        objects, stream position may not be updated accordingly after reading

    Returns
    ----------
    pd.DataFrame
        The newly extracted pandas dataframe.
    """
    LOGGER.info("Reading dataset from .csv file...")
    return pd.read_csv(filepath_or_buffer=source)
