"""Logging module"""

import logging


def cli_logger_get(current_module: str) -> logging.Logger:
    """Create a `Logger` instance with basic configuration. The default logging
    level is set to `INFO`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="""%(name)s
        %(asctime)s - %(levelname)s - %(message)s * [%(filename)s, line %(lineno)d]""",
    )

    return logging.getLogger(current_module)
