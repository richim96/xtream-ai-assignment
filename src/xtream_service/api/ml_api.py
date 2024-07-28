"""Entry point for the ML app serving the models."""

import fastapi

from xtream_service.utils.utils import uuid_get

ml_app = fastapi.FastAPI()


@ml_app.post("/diamonds/predict_value")
def diamond_price_predict():
    request_uid: str = uuid_get()
    response_uid: str = uuid_get()
    # save request

    # save response
    return  # response


@ml_app.post("/diamond/get_similar")
def diamond_samples_get():
    # save request

    # save response
    return  # response
