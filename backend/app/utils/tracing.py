import time
from contextlib import contextmanager

from app.utils.env import load_backend_env


@contextmanager
def trace_span(name: str):
    load_backend_env()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        print(f"[trace] {name} completed in {duration_ms:.1f}ms")

