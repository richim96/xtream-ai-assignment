"""Constants for the ML pipeline tests"""

CSV_SOURCE: str = "assets/data/raw/diamonds/diamonds.csv"

COLS_CATEGORICAL: list[str] = ["cut", "color", "clarity"]
COLS_TO_DROP: list[str] = ["depth", "table", "y", "z"]
COLS_NUMERIC: list[str] = ["carat", "price", "x", "y", "z"]

XGB_TARGETS: dict[str, list[str]] = {
    "cut": ["Fair", "Good", "Very Good", "Ideal", "Premium"],
    "color": ["D", "E", "F", "G", "H", "I", "J"],
    "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
}
