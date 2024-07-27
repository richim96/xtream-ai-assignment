"""Module for the model selection workflow"""


def sota_set(models: list) -> None:
    """Flag the current state-of-the-art model.

    Parameters
    ----------
    models : list
        List of models to iterate to find the current SOTA.
    """
    sota = models[0]
    for model in models[1:]:
        if model.metrics.mae < sota.metrics.mae:
            sota.is_sota = False
            sota = model
            sota.is_sota = True
