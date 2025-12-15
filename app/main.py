#!/usr/bin/env python3
"""
DJ Tools Bot - Main Entry Point

Clean Architecture application for DJ set analysis.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point."""
    logger.info("Starting DJ Tools Bot...")

    from app.modules.bot.routers.main import start_bot
    await start_bot()


if __name__ == "__main__":
    asyncio.run(main())
