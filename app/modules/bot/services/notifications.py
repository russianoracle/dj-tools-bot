"""Notification helper for temporary auto-delete messages."""

import asyncio
import logging
from aiogram.types import Message

logger = logging.getLogger(__name__)


async def send_auto_delete_notification(
    message: Message,
    text: str,
    delete_after: int = 4,
    parse_mode: str = "HTML",
) -> None:
    """
    Send temporary notification that auto-deletes after a delay.

    This creates a "toast-like" notification in Telegram chat
    without disrupting the current menu or FSM state.

    Args:
        message: Original user message to respond to
        text: Notification text
        delete_after: Seconds before auto-delete (default: 4)
        parse_mode: Parse mode for text (default: HTML)

    Example:
        await send_auto_delete_notification(
            message,
            "✓ 3 tracks queued for analysis"
        )
    """
    try:
        # Send notification message
        notification = await message.answer(text, parse_mode=parse_mode)

        # Schedule auto-delete
        await asyncio.sleep(delete_after)

        try:
            await notification.delete()
        except Exception as delete_error:
            # Message might be already deleted by user
            logger.debug(f"Could not delete notification: {delete_error}")

    except Exception as e:
        logger.error(f"Error sending auto-delete notification: {e}")
