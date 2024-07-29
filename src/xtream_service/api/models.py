"""Pydantic models for the api service"""

from pydantic import BaseModel


class Diamond(BaseModel):
    """Diamond base object model for the API."""

    # Weight
    carat: float
    # Attributes
    cut: str
    color: str
    clarity: str
    # Physical measurements
    depth: float
    table: float
    # Dimensions
    x: float
    y: float
    z: float


class DiamondPriceResponse(BaseModel):
    """Response model for the diamonds price prediction api."""

    response_id: str
    predicted_price: int
    model: str
    source_model: str
    request: Diamond
    created_at: str


class DiamondSampleRequest(BaseModel):
    """Request model for the diamonds api."""

    diamond: Diamond
    n_samples: int


class DiamondSampleResponse(BaseModel):
    """Response model for the diamond sampling api."""

    response_id: str
    dataset_samples: list[dict]
    dataset: str
    source_dataset: str
    request: DiamondSampleRequest
    created_at: str
