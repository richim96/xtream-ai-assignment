"""Entry point for the diamond model serving API - runs on FastAPI"""

from fastapi import FastAPI

from xtream_service.api.price_prediction import price_router
from xtream_service.api.sampling import sampling_router


app: FastAPI = FastAPI(
    title="Diamond P&S",
    description="API for diamond price prediction and sample retrieval.",
)
prefix: str = "/diamonds"
app.include_router(price_router, prefix=prefix)
app.include_router(sampling_router, prefix=prefix)
