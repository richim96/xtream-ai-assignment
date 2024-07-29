"""Diamond sampling route"""

import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter

from xtream_service.utils import uuid_get, utc_time_get
from xtream_service.db.put import response_db_put

from xtream_service.api import LOGGER
from xtream_service.api.const import QUALITIES, CARAT
from xtream_service.api.pydantic_models import (
    DiamondSampleRequest,
    DiamondSampleResponse,
)

load_dotenv(find_dotenv())
DATA_SOURCE: str = os.getenv("DATA_SOURCE", "")

sampling_router: APIRouter = APIRouter()


@sampling_router.get("/samples")
async def diamond_samples_get(
    carat: float, cut: str, color: str, clarity: str, n_samples: int
) -> DiamondSampleResponse:
    """Request `n` similar diamond samples from the original dataset, according
    to the specified characteristics. Async because loading data may be expensive.

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
    # Convert request data into a pandas dataframe to process the query
    diamond_obj: dict[str, float | str] = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
    }
    diamond_df: pd.DataFrame = pd.DataFrame([diamond_obj], index=[0])

    n_samples = max(n_samples, 0)  # n_samples must be >= 0
    samples: list[dict] = await samples_get(diamond_df, n_samples)
    LOGGER.info("Retrieved %s samples from: %s", len(samples), DATA_SOURCE)

    response: DiamondSampleResponse = DiamondSampleResponse(
        response_id=uuid_get(),
        n_samples=len(samples),
        samples=samples,
        dataset=DATA_SOURCE.split("/")[-1],  # name of the dataset
        source_dataset=DATA_SOURCE,
        request_data=DiamondSampleRequest(diamond=diamond_obj, n_samples=n_samples),
        created_at=utc_time_get(),
    )
    response_db_put(response)
    return response


async def samples_get(diamond_obj: pd.DataFrame, n_samples: int) -> list[dict]:
    """Retrieve `n` diamonds with similar qualities from the available data.

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
    carat_diff = (diamond_data[CARAT] - diamond_obj.loc[0, CARAT]).abs() <= 0.11

    matching_entries: pd.DataFrame = diamond_data[is_equal & carat_diff]
    samples = matching_entries.sample(min(n_samples, len(matching_entries)))

    return samples.to_dict(orient="records")
