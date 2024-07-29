"""Diamond sampling route"""

import os
import pandas as pd

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter

from xtream_service.utils import uuid_get, utc_time_get
from xtream_service.db.put import response_db_put

from xtream_service.api import LOGGER
from xtream_service.api.const import ATTRIBUTES, WEIGHT
from xtream_service.api.models import (
    Diamond,
    DiamondSampleRequest,
    DiamondSampleResponse,
)


load_dotenv(find_dotenv())
data_source: str = os.getenv("DATA_SOURCE", "")
diamond_data: pd.DataFrame = pd.read_csv(data_source)

sampling_router: APIRouter = APIRouter()


@sampling_router.post("/samples")
async def diamond_samples_get(
    diamond_obj: Diamond, n_samples: int
) -> DiamondSampleResponse:
    """Request n similar diamond samples from the original dataset.

    Parameters
    ----------
    diamond_obj : Diamond
        The diamond data to process.
    n_samples : int,
        Number of similar diamonds to retrieve from the original dataset.

    Return
    ----------
    DiamondSampleResponse
        The response containing the diamond samples.
    """
    # Convert request data into a pandas dataframe to process the queries
    diamond_df: pd.DataFrame = pd.DataFrame(
        {col: [row] for col, row in diamond_obj.model_dump().items()}
    )
    n_samples = max(n_samples, 0)  # n_samples must be >= 0
    samples: list[dict] = samples_get(diamond_df, n_samples)
    LOGGER.info("Retrieved %s samples from %s", len(samples), data_source)

    response: DiamondSampleResponse = DiamondSampleResponse(
        response_id=uuid_get(),
        dataset_samples=samples,
        dataset=data_source.split("/")[-1],  # name of the dataset
        source_dataset=data_source,
        request=DiamondSampleRequest(diamond=diamond_obj, n_samples=n_samples),
        created_at=utc_time_get(),
    )
    response_db_put(response)
    return response


def samples_get(diamond_obj: pd.DataFrame, n_samples: int) -> list[dict]:
    """Get n samples of diamonds with similar qualities from the available data.

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
    diamond_obj = diamond_obj.reindex(diamond_data.index)
    is_equal = (diamond_data[ATTRIBUTES] == diamond_obj[ATTRIBUTES]).all(axis=1)
    is_similar = (diamond_data[WEIGHT] - diamond_obj[WEIGHT]).abs() <= 0.11

    matching_rows: pd.DataFrame = diamond_data[is_equal & is_similar]
    samples = matching_rows.sample(min(n_samples, len(matching_rows)))

    return samples.to_dict(orient="records")
