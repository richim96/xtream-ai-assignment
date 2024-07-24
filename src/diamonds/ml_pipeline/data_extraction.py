"""Sub-module for the data extraction workflow"""

import polars as pl

from pathlib import Path
from typing import IO


def extract_from_csv(source: str | Path | IO[str] | IO[bytes] | bytes) -> pl.DataFrame:
    """Load csv source data into a polars dataframe.

    Parameters
    ----------
    source : str | Path | IO[str] | IO[bytes] | bytes
        Path to a file or a file-like object. If `fsspec` is installed, it will
        be used to open remote files. For file-like objects, stream position may
        not be updated accordingly after reading.

    Returns
    ----------
    pl.DataFrame
        The newly extracted polars dataframe.
    """
    return pl.read_csv(source=source)
