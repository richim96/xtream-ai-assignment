"""Logging module. Only the logger `LOGGER` is meant for import."""

import logging


def _std_logger_get() -> logging.Logger:
    """Create a `Logger` instance with basic configuration. The default logging
    level is set to `INFO`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="""%(name)s
        %(asctime)s - %(levelname)s - %(message)s * [%(filename)s, line %(lineno)d]""",
    )

    return logging.getLogger(__name__)


LOGGER: logging.Logger = _std_logger_get()
