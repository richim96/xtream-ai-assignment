"""Price prediction route"""

import os
import pickle

import numpy as np
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter, BackgroundTasks

from xtream_service.utils import uuid_get, utc_time_get
from xtream_service.ml_pipeline import data_processing
from xtream_service.db.data_put import request_db_put, response_db_put

from xtream_service.api import LOGGER
from xtream_service.api._const import NUMERIC, CATEGORICAL, LINEAR_TO_DROP, XGB_TO_CATG
from xtream_service.api.pydantic_models import (
    Diamond,
    DiamondPriceRequest,
    DiamondPriceResponse,
)

load_dotenv(find_dotenv())
MODEL_SOURCE: str = os.getenv("SOTA_SOURCE", "")

price_router: APIRouter = APIRouter()


@price_router.post("/price/", response_model=DiamondPriceResponse)
async def diamond_price_predict(
    diamond_obj: Diamond, background_tasks: BackgroundTasks
) -> DiamondPriceResponse:
    """Query the price of a given diamond.

    Parameters
    ----------
    diamond : Diamond
        The diamond data to process.

    Return
    ----------
    DiamondPriceResponse
        The response with the price prediction.
    """
    request: DiamondPriceRequest = DiamondPriceRequest(
        id=uuid_get(),
        response_id=uuid_get(),
        diamond=diamond_obj,
        created_at=utc_time_get(),
    )
    background_tasks.add_task(request_db_put, request)

    # Convert request data into a pandas dataframe and process query
    diamond_df: pd.DataFrame = pd.DataFrame(
        {col: [row] for col, row in diamond_obj.model_dump().items()}
    )
    price: int = await price_predict(diamond_df)
    LOGGER.info("Predicted a price of %s€ for diamond {%s}", price, diamond_obj)

    response: DiamondPriceResponse = DiamondPriceResponse(
        id=request.response_id,
        request_id=request.id,
        predicted_price=price,
        model=MODEL_SOURCE.split("/")[-1],  # name of the model
        source_model=MODEL_SOURCE,
        created_at=utc_time_get(),
    )
    background_tasks.add_task(response_db_put, response)
    return response


async def price_predict(diamond_obj: pd.DataFrame) -> int:
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
    # Prepare data for the model
    diamond_obj = data_processing.filter_numeric(diamond_obj, cols=NUMERIC, n=0)

    if "linear" in MODEL_SOURCE:
        diamond_obj = diamond_obj.drop(columns=LINEAR_TO_DROP)
        diamond_obj = data_processing.dummy_encode(diamond_obj, cols=CATEGORICAL)
    elif "xgb" in MODEL_SOURCE:
        diamond_obj = data_processing.to_categorical_dtype(diamond_obj, XGB_TO_CATG)

    # Load model from source and predict price
    with open(MODEL_SOURCE, "rb") as obj:
        model = pickle.load(obj)

    pred: float = model.predict(diamond_obj).item()
    pred = int(np.expm1(pred)) if "_log" in MODEL_SOURCE else int(pred)

    return pred
