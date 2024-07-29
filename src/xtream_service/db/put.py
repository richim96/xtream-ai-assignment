"""Module collecting the database put functions"""

from xtream_service.db import LOGGER
from xtream_service.api.pydantic_models import (
    DiamondPriceRequest,
    DiamondPriceResponse,
    DiamondSampleRequest,
    DiamondSampleResponse,
)


def request_db_put(request: DiamondPriceRequest | DiamondSampleRequest) -> None:
    """Save the api request to the db.

    Parameters
    ----------
    response : DiamondPriceRequest | DiamondSampleRequest
        API request body.
    """
    LOGGER.info("Request saved to database: {%s}", request)


def response_db_put(response: DiamondPriceResponse | DiamondSampleResponse) -> None:
    """Save the api response to the db.

    Parameters
    ----------
    response : DiamondPriceResponse | DiamondSampleResponse
        API reponse body.
    """
    LOGGER.info("Response saved to database: {%s}", response)
