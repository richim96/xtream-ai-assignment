"""Constants for the ML pipeline tests"""

CSV_SOURCE: str = "assets/data/raw/diamonds/diamonds.csv"

COLS_CATEGORICAL: list[str] = ["cut", "color", "clarity"]
COLS_TO_DROP: list[str] = ["depth", "table", "y", "z"]
COLS_NUMERIC: list[str] = ["carat", "price", "x"]
