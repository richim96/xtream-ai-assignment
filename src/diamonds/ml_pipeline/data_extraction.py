"""Extract training data"""

import io
import  polars as pl
from pathlib import Path

def data_extract(source: str | Path | list[str] | list[Path]) -> pl.LazyFrame:
    """Lazy loads data into a polars Lazy Frame, from a given source.
    """
    return pl.scan_csv(source)

