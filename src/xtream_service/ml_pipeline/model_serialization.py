"""Module for serializing ML models."""

import pickle
from uuid import uuid4, UUID

from xtream_service.ml_pipeline import LOGGER


# TODO: convert this function into an object
def model_object_make(model, data_uid: UUID) -> dict:
    """Creates a model object and its relevant metadata for logging. The model
    is also serialized and saved in persistent storage.

    Parameters
    ----------
    model
        Any trained model.
    data_uid: UUID
        The unique indentifier of the data used to train the model.

    Return
    ----------
    dict
        The model object, containing the model itself and the relevant metadata.
    """
    model_uid: UUID = uuid4()
    model_obj: dict = {
        "model_uid": model_uid,
        "data_uid": data_uid,
        "timestamp": None,
        "metrics": {},
    }

    with open(f"assets/models/{model_uid}.pkl", "wb") as f:
        LOGGER.info("Model %s saved in persistent storage.", model_uid)
        pickle.dump(model, f)

    return model_obj
