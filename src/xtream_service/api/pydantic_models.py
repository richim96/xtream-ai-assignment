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

    diamond: Diamond


class DiamondPriceResponse(BaseModel):
    """Response model for the diamonds price prediction api."""

    response_id: str
    predicted_price: int
    model: str
    source_model: str
    request_data: DiamondPriceRequest
    created_at: str


class DiamondSampleRequest(BaseModel):
    """Request model for the diamonds sampling api."""

    diamond: dict[str, float | str]
    n_samples: int


class DiamondSampleResponse(BaseModel):
    """Response model for the diamond sampling api."""

    response_id: str
    n_samples: int
    samples: list[dict]
    dataset: str
    source_dataset: str
    request_data: DiamondSampleRequest
    created_at: str
