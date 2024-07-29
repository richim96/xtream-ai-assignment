"""Utilities for xtream services"""

from datetime import datetime
from uuid import uuid4

import tzlocal


def uuid_get() -> str:
    """Return a unique identifier as a string."""
    return str(uuid4())


def utc_time_get() -> str:
    """Return the current UTC time as a string."""
    return datetime.now(tzlocal.get_localzone()).isoformat()
