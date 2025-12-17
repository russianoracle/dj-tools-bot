#!/usr/bin/env python3
"""
DJ Tools Bot - Main Entry Point

Clean Architecture application for DJ set analysis.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file (for local development)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Load secrets from Yandex Lockbox (production)
from app.core.secrets import init_secrets
init_secrets(
    secret_id=os.getenv("YC_LOCKBOX_SECRET_ID"),
    validate=True,
    fail_on_missing=os.getenv("REQUIRE_SECRETS", "true").lower() == "true",
)

# Setup centralized logging with YC Cloud Logging support
from app.common.logging import setup_logging, get_logger

setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE"),
    json_format=os.getenv("LOG_JSON_FORMAT", "true").lower() == "true",
    enable_yc_logging=True,
    yc_resource_type="bot",
)

logger = get_logger(__name__)


async def main():
    """Main entry point."""
    logger.info("Starting DJ Tools Bot...")

    from app.modules.bot.routers.main import start_bot
    await start_bot()


if __name__ == "__main__":
    asyncio.run(main())
