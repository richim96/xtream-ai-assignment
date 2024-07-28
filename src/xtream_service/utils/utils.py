"""Utilities for xtream services."""

from datetime import datetime, timezone
from uuid import uuid4


def uuid_get() -> str:
    """Return a unique identifier as a string."""
    return str(uuid4())


def utc_time_get() -> str:
    """Return the current UTC time as a string."""
    return str(datetime.now(timezone.utc))
