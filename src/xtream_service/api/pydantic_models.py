"""Pydantic models for the api service"""

from pydantic import BaseModel


class Diamond(BaseModel):
    """Diamond base object model for the API."""

    carat: float
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


class DiamondPriceRequest(BaseModel):
    """Request model for the diamonds price prediction api."""

    id: str
    response_id: str
    diamond: Diamond
    request_type: str = "price_predict"
    created_at: str


class DiamondPriceResponse(BaseModel):
    """Response model for the diamonds price prediction api."""

    id: str
    request_id: str
    predicted_price: int
    model: str
    source_model: str
    response_type: str = "price_predict"
    created_at: str


class DiamondSampleRequest(BaseModel):
    """Request model for the diamonds sampling api."""

    id: str
    response_id: str
    diamond: dict
    n_samples: int
    request_type: str = "sample_get"
    created_at: str


class DiamondSampleResponse(BaseModel):
    """Response model for the diamond sampling api."""

    id: str
    request_id: str
    n_samples: int
    samples: list[dict]
    dataset: str
    source_dataset: str
    response_type: str = "sample_get"
    created_at: str
