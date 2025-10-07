from contextvars import ContextVar
from fastapi import FastAPI

_current_app: ContextVar[FastAPI] = ContextVar("current_app", default=None)

def set_app_context(app: FastAPI):
    """Set the current running FastAPI app (during lifespan or request)."""
    _current_app.set(app)

def get_app_context() -> FastAPI:
    """Get the current FastAPI app context anywhere in code."""
    app = _current_app.get()
    if app is None:
        raise RuntimeError("App context not set. Make sure you're inside FastAPI lifespan or request.")
    return app