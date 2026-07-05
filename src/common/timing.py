"""Shared timing instrumentation for logging per-step start/end/duration."""
from contextlib import contextmanager
from datetime import datetime
import logging
import time


@contextmanager
def timed_step(logger: logging.Logger, step_name: str):
    """Log start timestamp, end timestamp, and duration for a wrapped block."""
    start_ts = datetime.now()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        end_ts = datetime.now()
        logger.info(
            "[TIMING] %s | start=%s end=%s duration=%.3fs",
            step_name,
            start_ts.isoformat(timespec="milliseconds"),
            end_ts.isoformat(timespec="milliseconds"),
            duration,
        )
