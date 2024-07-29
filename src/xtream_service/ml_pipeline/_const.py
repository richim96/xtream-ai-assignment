"""Constants for processing the `diamonds.csv` dataset"""

NUMERIC: list[str] = ["carat", "price", "x", "y", "z"]
CATEGORICAL: list[str] = ["cut", "color", "clarity"]

LINEAR_TO_DROP: list[str] = ["depth", "table", "y", "z"]
XGB_TO_CATG: dict[str, list[str]] = {
    "cut": ["Fair", "Good", "Very Good", "Ideal", "Premium"],
    "color": ["D", "E", "F", "G", "H", "I", "J"],
    "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
}

TARGET: str = "price"
