"""Module for the model selection workflow"""


def sota_update(models: list, log: dict) -> dict:
    """Flag the current state-of-the-art model.

    Parameters
    ----------
    models : list
        List of models to iterate to find the current SOTA.
    log : dict
        Log file with data about previously trained models.

    Return
    ----------
    dict
        The log file with updated information about the SOTA, if any.
    """
    # Find current SOTA
    sota = models[0]
    for m in models[1:]:
        if (
            m.metrics["mean_absolute_error"] < sota.metrics["mean_absolute_error"]
            and m.metrics["r2_score"] > sota.metrics["r2_score"]
        ):
            sota.is_sota = False
            m.is_sota = True
            sota = m

    # Compare with training history and determine SOTA
    for model_info in log["data"]:
        if model_info["is_sota"]:
            if (
                model_info["metrics"]["mean_absolute_error"]
                < sota.metrics["mean_absolute_error"]
                and model_info["metrics"]["r2_score"] > sota.metrics["r2_score"]
            ):
                sota.is_sota = False
            else:
                model_info["is_sota"] = False

    return log
