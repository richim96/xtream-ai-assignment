"""Diamond sampling route"""

import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter, BackgroundTasks

from xtream_service.utils import uuid_get, utc_time_get
from xtream_service.db.data_put import request_db_put, response_db_put

from xtream_service.api import LOGGER
from xtream_service.api._const import QUALITIES, CARAT
from xtream_service.api.pydantic_models import (
    DiamondSampleRequest,
    DiamondSampleResponse,
)

load_dotenv(find_dotenv())
DATA_SOURCE: str = os.getenv("DATA_SOURCE", "")

sampling_router: APIRouter = APIRouter()


@sampling_router.get("/samples/", response_model=DiamondSampleResponse)
async def diamond_sample_get(
    carat: float,
    cut: str,
    color: str,
    clarity: str,
    n_samples: int,
    background_tasks: BackgroundTasks,
) -> DiamondSampleResponse:
    """Request `n` similar diamond samples from the original dataset, according
    to the specified characteristics.

    Parameters
    ----------
    carat : float
        The weight of the diamond.
    cut : str
        The light performance indicator of the diamond.
    color : str
        The color indicator of the diamond.
    clarity : str
        The clarity indicator of the diamond.
    n_samples : int,
        Number of similar samples to retrieve.

    Return
    ----------
    DiamondSampleResponse
        The response containing the diamond samples.
    """
    # Place request data in a dictionary and store the request
    diamond_obj = {"carat": carat, "cut": cut, "color": color, "clarity": clarity}
    request: DiamondSampleRequest = DiamondSampleRequest(
        id=uuid_get(),
        response_id=uuid_get(),
        diamond=diamond_obj,
        n_samples=n_samples,
        created_at=utc_time_get(),
    )
    background_tasks.add_task(request_db_put, request)

    # Convert data into a pandas dataframe and process query
    diamond_df: pd.DataFrame = pd.DataFrame([diamond_obj], index=[0])
    n_samples = max(n_samples, 0)  # n_samples must be >= 0
    samples: list[dict] = await samples_get(diamond_df, n_samples)
    LOGGER.info("Retrieved %s samples from: %s", len(samples), DATA_SOURCE)

    response: DiamondSampleResponse = DiamondSampleResponse(
        id=request.response_id,
        request_id=request.id,
        n_samples=len(samples),
        samples=samples,
        dataset=DATA_SOURCE.split("/")[-1],  # name of the dataset
        source_dataset=DATA_SOURCE,
        created_at=utc_time_get(),
    )
    background_tasks.add_task(response_db_put, response)
    return response


async def samples_get(diamond_obj: pd.DataFrame, n_samples: int) -> list[dict]:
    """Retrieve `n` diamonds with the same qualities and similar weight (carat)
    from the available data. A carat difference > 0.1 is considered significant.

    Parameters
    ----------
    diamond_obj : pd.DataFrame
        Dataframe representing the diamond object received via API request.
    n_samples : int
        Number of similar samples to retrieve.

    Return
    ----------
    list[dict]
        A list containing the sample objects.
    """
    diamond_data: pd.DataFrame = pd.read_csv(DATA_SOURCE)

    # Masks to match all dataset entries to the requested values
    is_equal = (diamond_data[QUALITIES] == diamond_obj.loc[0, QUALITIES]).all(axis=1)
    carat_diff = (diamond_data[CARAT] - diamond_obj.loc[0, CARAT]).abs() <= 0.1

    matching_entries: pd.DataFrame = diamond_data[is_equal & carat_diff]
    samples = matching_entries.sample(min(n_samples, len(matching_entries)))

    return samples.to_dict(orient="records")
