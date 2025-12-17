"""
Main router - combines all bot handlers.

Uses ARQ for async task queue (Redis 8.x/Valkey compatible).
"""

import os

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from ..handlers import start, analyze, admin, profile, generate
from app.common.logging import get_logger, CorrelationMiddleware

logger = get_logger(__name__)

# Environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def create_bot() -> Bot:
    """Create bot instance."""
    if not BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

    return Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )


def create_dispatcher() -> Dispatcher:
    """Create dispatcher with all routers."""
    dp = Dispatcher()

    # Add correlation ID middleware for request tracing
    dp.update.middleware(CorrelationMiddleware())

    # Create main router
    main_router = Router()

    # Include all handler routers
    main_router.include_router(start.router)
    main_router.include_router(analyze.router)
    main_router.include_router(admin.router)
    main_router.include_router(profile.router)
    main_router.include_router(generate.router)

    # Add to dispatcher
    dp.include_router(main_router)

    return dp


async def start_bot():
    """Start the bot."""
    bot = create_bot()
    dp = create_dispatcher()

    logger.info("Starting bot...")

    # Start polling
    await dp.start_polling(bot)


__all__ = [
    'create_bot',
    'create_dispatcher',
    'start_bot',
]
