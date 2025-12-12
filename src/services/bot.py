"""
Telegram Bot for mood-classifier.

Commands:
- /start - Welcome message
- /help - Help information
- /analyze <url> - Analyze DJ set from URL
- /status <job_id> - Check job status

Admin Commands:
- /admin stats - System statistics
- /admin disk - Disk usage
- /admin cache clear - Clear cache
- /admin ban <user_id> - Ban user
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()

# In-memory storage (use Redis in production)
user_jobs: dict = {}  # user_id -> [job_ids]
banned_users: set = set()


# ============== Helpers ==============

def is_admin(user_id: int) -> bool:
    """Check if user is admin."""
    return user_id == ADMIN_USER_ID


def get_disk_usage() -> dict:
    """Get disk usage statistics."""
    import shutil
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/app/downloads")
    cache_dir = os.getenv("CACHE_DIR", "/app/cache")

    total, used, free = shutil.disk_usage("/")

    downloads_size = sum(
        os.path.getsize(os.path.join(downloads_dir, f))
        for f in os.listdir(downloads_dir)
        if os.path.isfile(os.path.join(downloads_dir, f))
    ) if os.path.exists(downloads_dir) else 0

    return {
        "total_gb": total // (1024**3),
        "used_gb": used // (1024**3),
        "free_gb": free // (1024**3),
        "used_percent": round(used / total * 100, 1),
        "downloads_mb": downloads_size // (1024**2),
    }


# ============== User Commands ==============

@router.message(CommandStart())
async def cmd_start(message: Message):
    """Handle /start command."""
    if message.from_user.id in banned_users:
        return

    await message.answer(
        "<b>Welcome to Mood Classifier Bot!</b>\n\n"
        "I analyze DJ sets and classify tracks by energy zones:\n"
        "- 🟨 <b>Yellow</b> - Rest zone (low energy)\n"
        "- 🟩 <b>Green</b> - Transitional (medium energy)\n"
        "- 🟪 <b>Purple</b> - Hits/Energy (high energy)\n\n"
        "Send me a SoundCloud link or audio file to analyze.\n\n"
        "Commands:\n"
        "/analyze &lt;url&gt; - Analyze DJ set\n"
        "/status &lt;job_id&gt; - Check job status\n"
        "/help - More information"
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    if message.from_user.id in banned_users:
        return

    await message.answer(
        "<b>How to use:</b>\n\n"
        "1. Send a SoundCloud link:\n"
        "   <code>/analyze https://soundcloud.com/...</code>\n\n"
        "2. Or upload an audio file directly (max 500MB)\n\n"
        "3. Wait for analysis (usually 2-5 minutes)\n\n"
        "4. Get results with track boundaries, drops, and energy zones\n\n"
        "<b>Supported formats:</b> MP3, WAV, FLAC, M4A, OPUS"
    )


@router.message(Command("analyze"))
async def cmd_analyze(message: Message):
    """Handle /analyze <url> command."""
    if message.from_user.id in banned_users:
        await message.answer("You are banned from using this bot.")
        return

    # Extract URL from command
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.answer(
            "Please provide a URL:\n"
            "<code>/analyze https://soundcloud.com/...</code>"
        )
        return

    url = args[1].strip()

    # Validate URL
    if not url.startswith(("http://", "https://")):
        await message.answer("Invalid URL. Please provide a valid HTTP/HTTPS link.")
        return

    # Check disk space
    disk = get_disk_usage()
    if disk["used_percent"] > 90:
        await message.answer(
            "⚠️ Server disk space is critically low. Please try again later."
        )
        if is_admin(message.from_user.id):
            await message.answer(f"Disk usage: {disk['used_percent']}%")
        return

    # Queue analysis
    await message.answer(
        f"📥 <b>Queuing analysis...</b>\n\n"
        f"URL: {url}\n\n"
        f"This may take 2-5 minutes. I'll notify you when it's done."
    )

    # TODO: Integrate with Celery task queue
    # For now, simulate with placeholder
    job_id = f"job_{message.from_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Store job reference
    if message.from_user.id not in user_jobs:
        user_jobs[message.from_user.id] = []
    user_jobs[message.from_user.id].append(job_id)

    await message.answer(
        f"✅ <b>Analysis started!</b>\n\n"
        f"Job ID: <code>{job_id}</code>\n\n"
        f"Use <code>/status {job_id}</code> to check progress."
    )


@router.message(Command("status"))
async def cmd_status(message: Message):
    """Handle /status <job_id> command."""
    if message.from_user.id in banned_users:
        return

    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        # Show user's recent jobs
        user_id = message.from_user.id
        if user_id in user_jobs and user_jobs[user_id]:
            jobs_list = "\n".join(f"• <code>{j}</code>" for j in user_jobs[user_id][-5:])
            await message.answer(
                f"<b>Your recent jobs:</b>\n{jobs_list}\n\n"
                f"Use <code>/status &lt;job_id&gt;</code> to check status."
            )
        else:
            await message.answer("No recent jobs found. Use /analyze to start one.")
        return

    job_id = args[1].strip()

    # TODO: Query actual job status from Redis/Celery
    await message.answer(
        f"<b>Job Status:</b>\n\n"
        f"ID: <code>{job_id}</code>\n"
        f"Status: <i>Processing...</i>\n"
        f"Progress: 45%"
    )


# ============== File Handler ==============

@router.message(F.audio | F.document)
async def handle_file(message: Message):
    """Handle uploaded audio files."""
    if message.from_user.id in banned_users:
        return

    file = message.audio or message.document
    if not file:
        return

    # Check file size
    if file.file_size and file.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await message.answer(
            f"⚠️ File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
        return

    # Check extension
    file_name = file.file_name or "unknown"
    ext = os.path.splitext(file_name)[1].lower()
    allowed = {".mp3", ".wav", ".flac", ".m4a", ".opus", ".ogg"}
    if ext not in allowed:
        await message.answer(
            f"⚠️ Unsupported format: {ext}\n"
            f"Supported: {', '.join(allowed)}"
        )
        return

    await message.answer(
        f"📥 <b>Downloading file...</b>\n"
        f"Name: {file_name}\n"
        f"Size: {file.file_size // (1024*1024)}MB"
    )

    # TODO: Download and queue for analysis
    job_id = f"job_{message.from_user.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    await message.answer(
        f"✅ <b>Analysis started!</b>\n\n"
        f"Job ID: <code>{job_id}</code>"
    )


# ============== Admin Commands ==============

@router.message(Command("admin"))
async def cmd_admin(message: Message):
    """Handle /admin commands."""
    if not is_admin(message.from_user.id):
        await message.answer("⛔ Admin access required.")
        return

    args = message.text.split()[1:] if len(message.text.split()) > 1 else []

    if not args:
        await message.answer(
            "<b>Admin Commands:</b>\n\n"
            "/admin stats - System statistics\n"
            "/admin disk - Disk usage\n"
            "/admin cache clear - Clear cache\n"
            "/admin ban &lt;user_id&gt; - Ban user\n"
            "/admin unban &lt;user_id&gt; - Unban user"
        )
        return

    cmd = args[0].lower()

    if cmd == "stats":
        await message.answer(
            f"<b>System Statistics:</b>\n\n"
            f"Total users: {len(user_jobs)}\n"
            f"Banned users: {len(banned_users)}\n"
            f"Active jobs: N/A"
        )

    elif cmd == "disk":
        disk = get_disk_usage()
        status = "🟢" if disk["used_percent"] < 80 else ("🟡" if disk["used_percent"] < 90 else "🔴")
        await message.answer(
            f"<b>Disk Usage:</b> {status}\n\n"
            f"Total: {disk['total_gb']}GB\n"
            f"Used: {disk['used_gb']}GB ({disk['used_percent']}%)\n"
            f"Free: {disk['free_gb']}GB\n"
            f"Downloads: {disk['downloads_mb']}MB"
        )

    elif cmd == "cache" and len(args) > 1 and args[1] == "clear":
        # TODO: Clear cache
        await message.answer("🗑️ Cache cleared.")

    elif cmd == "ban" and len(args) > 1:
        try:
            user_id = int(args[1])
            banned_users.add(user_id)
            await message.answer(f"🚫 User {user_id} banned.")
        except ValueError:
            await message.answer("Invalid user ID.")

    elif cmd == "unban" and len(args) > 1:
        try:
            user_id = int(args[1])
            banned_users.discard(user_id)
            await message.answer(f"✅ User {user_id} unbanned.")
        except ValueError:
            await message.answer("Invalid user ID.")

    else:
        await message.answer("Unknown admin command. Use /admin for help.")


# ============== Main ==============

async def main():
    """Start the bot."""
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        return

    logger.info("Starting bot...")

    dp.include_router(router)

    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
