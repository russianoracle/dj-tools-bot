"""Correlation ID middleware for request tracing."""

import uuid
import logging
import contextvars
from typing import Callable, Awaitable, Any

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update


# Context variable for correlation ID (thread-safe, async-safe)
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for user ID
user_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "user_id", default=None
)

# Context variable for job ID
job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "job_id", default=None
)


def generate_correlation_id() -> str:
    """Generate unique correlation ID."""
    return str(uuid.uuid4())[:8]


def get_correlation_id() -> str | None:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(cid: str):
    """Set correlation ID in context."""
    correlation_id_var.set(cid)


def get_user_id() -> int | None:
    """Get current user ID from context."""
    return user_id_var.get()


def set_user_id(uid: int):
    """Set user ID in context."""
    user_id_var.set(uid)


def get_job_id() -> str | None:
    """Get current job ID from context."""
    return job_id_var.get()


def set_job_id(jid: str):
    """Set job ID in context."""
    job_id_var.set(jid)


class CorrelationMiddleware(BaseMiddleware):
    """
    Aiogram middleware that sets correlation ID for each update.

    Generates unique correlation ID per Telegram update for request tracing.
    Also extracts user_id from update for logging context.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        # Generate correlation ID
        cid = generate_correlation_id()
        set_correlation_id(cid)

        # Extract user_id from update
        if isinstance(event, Update):
            user = None
            if event.message:
                user = event.message.from_user
            elif event.callback_query:
                user = event.callback_query.from_user
            elif event.inline_query:
                user = event.inline_query.from_user

            if user:
                set_user_id(user.id)

        # Add to handler data for easy access
        data["correlation_id"] = cid

        try:
            return await handler(event, data)
        finally:
            # Clear context after request
            correlation_id_var.set(None)
            user_id_var.set(None)
            job_id_var.set(None)


class CorrelationLogFilter(logging.Filter):
    """
    Logging filter that adds correlation_id, user_id, job_id to log records.

    Use with standard logging to auto-inject context vars.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        record.user_id = get_user_id()
        record.job_id = get_job_id()
        return True
