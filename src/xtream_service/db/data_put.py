"""Module collecting the database put functions"""

import os
import pymongo

from dotenv import load_dotenv, find_dotenv

from xtream_service.db import LOGGER
from xtream_service.api.pydantic_models import (
    DiamondPriceRequest,
    DiamondPriceResponse,
    DiamondSampleRequest,
    DiamondSampleResponse,
)

load_dotenv(find_dotenv())
MONGO_HOST: str = os.getenv("MONGO_HOST", "mongodb://localhost:27017")


async def request_db_put(request: DiamondPriceRequest | DiamondSampleRequest) -> None:
    """Save the api request to the db.

    Parameters
    ----------
    response : DiamondPriceRequest | DiamondSampleRequest
        API request body.
    """
    try:
        with pymongo.MongoClient(MONGO_HOST) as client:  # type: pymongo.MongoClient
            # Create or select the database and the collection
            database = client["diamond_db"]
            diamond_request_collection = database["diamond_request"]
            # Insert the new request
            diamond_request_collection.insert_one(request.model_dump())

        LOGGER.info("Request saved to MongoDB database: {%s}", request)

    except pymongo.errors.ServerSelectionTimeoutError as e:
        LOGGER.error("Request not saved. MongoDB server selection timeout: %s", e)
    except pymongo.errors.ConnectionFailure as e:
        LOGGER.error("Request not saved. MongoDB connection error: %s", e)
    except Exception as e:
        LOGGER.error("Request not saved. MongoDB threw an unexpected error: %s", e)


async def response_db_put(
    response: DiamondPriceResponse | DiamondSampleResponse,
) -> None:
    """Save the api response to the db.

    Parameters
    ----------
    response : DiamondPriceResponse | DiamondSampleResponse
        API reponse body.
    """
    try:
        with pymongo.MongoClient(MONGO_HOST) as client:  # type: pymongo.MongoClient
            # Create or select the database and the collection
            database = client["diamond_db"]
            diamond_response_collection = database["diamond_response"]
            # Insert the new request
            diamond_response_collection.insert_one(response.model_dump())

        LOGGER.info("Response saved to MongoDB database: {%s}", response)

    except pymongo.errors.ServerSelectionTimeoutError as e:
        LOGGER.error("Response not saved. MongoDB server selection timeout: %s", e)
    except pymongo.errors.ConnectionFailure as e:
        LOGGER.error("Response not saved. MongoDB connection error: %s", e)
    except Exception as e:
        LOGGER.error("Response not saved. MongoDB threw an unexpected error: %s", e)
