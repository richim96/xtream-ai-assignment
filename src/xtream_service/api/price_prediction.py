"""Price prediction route"""

import os
import pickle
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter

from xtream_service.utils import uuid_get, utc_time_get
from xtream_service.ml_pipeline import data_processing
from xtream_service.db.put import response_db_put

from xtream_service.api import LOGGER
from xtream_service.api.const import NUMERIC, CATEGORICAL, LINEAR_TO_DROP, XGB_TO_CATG
from xtream_service.api.models import Diamond, DiamondPriceResponse


load_dotenv(find_dotenv())
model_source: str = os.getenv("SOTA_SOURCE", "")

price_router: APIRouter = APIRouter()


@price_router.post("/price")
async def diamond_price_predict(diamond_obj: Diamond) -> DiamondPriceResponse:
    """Query the price of a diamond and request n similar samples from the
    original dataset.

    Parameters
    ----------
    diamond : Diamond
        The diamond data to process.

    Return
    ----------
    DiamondPriceResponse
        The response with the price prediction.
    """
    # Convert request data into a pandas dataframe to process the queries
    diamond_df: pd.DataFrame = pd.DataFrame(
        {col: [row] for col, row in diamond_obj.model_dump().items()}
    )
    price: int = price_predict(diamond_df)
    LOGGER.info("Predicted a price of %s€ for diamond {%s}", price, diamond_df)

    response: DiamondPriceResponse = DiamondPriceResponse(
        response_id=uuid_get(),
        predicted_price=price,
        model=model_source.split("/")[-1],  # name of the model
        source_model=model_source,
        request=diamond_obj,
        created_at=utc_time_get(),
    )
    response_db_put(response)
    return response


def price_predict(diamond_obj: pd.DataFrame) -> int:
    """Prepare input data for the appropriate model.

    Parameters
    ----------
    diamond_obj : pd.DataFrame
        Dataframe representing the diamond object received via API request.

    Return
    ----------
    int
        The predicted price of the diamond.
    """
    # Prepare data appropriately for the model
    diamond_obj = data_processing.filter_numeric(diamond_obj, cols=NUMERIC, n=0)
    if "linear" in model_source:
        diamond_obj = diamond_obj.drop(columns=LINEAR_TO_DROP)
        diamond_obj = data_processing.dummy_encode(diamond_obj, cols=CATEGORICAL)
    elif "xgb" in model_source:
        diamond_obj = data_processing.to_categorical_dtype(diamond_obj, XGB_TO_CATG)

    # Load model from source
    with open(model_source, "rb") as file:
        model = pickle.load(file)

    return int(model.predict(diamond_obj))
