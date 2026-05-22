import time
from contextlib import contextmanager
import os
from typing import Any

from app.utils.env import load_backend_env

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"


def configure_langsmith() -> tuple[bool, str | None]:
    load_backend_env()

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    endpoint = normalize_langsmith_endpoint(
        os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    )

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    if project:
        os.environ["LANGSMITH_PROJECT"] = project
        os.environ["LANGCHAIN_PROJECT"] = project

    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint

    return bool(api_key and project), project


def normalize_langsmith_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return None

    normalized = endpoint.strip().rstrip("/")
    if normalized in {
        "https://smith.langchain.com",
        "https://smith.langchain.com/api/v1",
        "https://api.smith.langchain.com/api/v1",
    }:
        return DEFAULT_LANGSMITH_ENDPOINT
    return normalized


@contextmanager
def trace_span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    enabled, project = configure_langsmith()
    start = time.perf_counter()
    trace_manager = None
    run_tree = None
    trace_closed = False

    if enabled:
        try:
            from langsmith.run_helpers import trace

            trace_manager = trace(
                name,
                run_type=run_type,
                inputs=inputs,
                project_name=project,
                metadata=metadata,
            )
            run_tree = trace_manager.__enter__()
        except Exception as exc:
            trace_manager = None
            run_tree = None
            print(f"[trace] LangSmith disabled for {name}: {exc}")

    try:
        yield run_tree
    except Exception as exc:
        if trace_manager is not None:
            try:
                trace_manager.__exit__(type(exc), exc, exc.__traceback__)
                trace_closed = True
            except Exception:
                pass
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        if run_tree is not None and trace_manager is not None and not trace_closed:
            try:
                run_tree.end(
                    outputs={"duration_ms": round(duration_ms, 1)},
                    metadata={"duration_ms": round(duration_ms, 1)},
                )
                trace_manager.__exit__(None, None, None)
            except Exception as exc:
                print(f"[trace] LangSmith patch failed for {name}: {exc}")
        print(f"[trace] {name} completed in {duration_ms:.1f}ms")
