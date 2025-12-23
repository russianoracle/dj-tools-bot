"""FastAPI health and metrics endpoints."""

from .health import router as health_router

__all__ = ['health_router']
