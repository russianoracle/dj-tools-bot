"""
Analyze handlers - URL and file analysis.

Uses single message architecture - all updates go to one message per user.
Uses ARQ for async task queue (compatible with Redis 8.x/Valkey).
"""

import os
from datetime import datetime

from aiogram import Router, F, Bot
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from ..keyboards.inline import (
    get_cancel_keyboard,
    get_back_keyboard,
    get_job_keyboard,
    get_jobs_keyboard,
)
from .start import (
    banned_users,
    is_admin,
    user_main_message,
    ensure_main_message,
    update_main_message,
    delete_user_message,
    get_main_text,
    get_main_keyboard,
)
from app.common.logging import get_logger
from app.common.logging.correlation import set_job_id

router = Router()
logger = get_logger(__name__)

# In-memory job storage (use Redis in production)
user_jobs: dict = {}  # user_id -> [job_ids]

# ARQ worker integration
from app.services.arq_worker import (
    get_job_status,
    enqueue_analyze_set,
    enqueue_download_and_analyze,
)

# Environment
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))


class AnalyzeStates(StatesGroup):
    """FSM states for analysis flow."""
    waiting_for_url = State()


def get_disk_usage() -> dict:
    """Get disk usage statistics."""
    import shutil
    downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")

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


def get_state_emoji(state: str) -> str:
    """Get emoji for job state."""
    return {
        "PENDING": "⏳",
        "PROGRESS": "🔄",
        "SUCCESS": "✅",
        "FAILURE": "❌",
    }.get(state, "❓")


@router.callback_query(F.data == "analyze_url")
async def cb_analyze_url(callback: CallbackQuery, state: FSMContext):
    """Start URL analysis flow."""
    if callback.from_user.id in banned_users:
        return

    await state.set_state(AnalyzeStates.waiting_for_url)

    await update_main_message(
        callback=callback,
        text=(
            "<b>🔗 Send URL to analyze:</b>\n\n"
            "Paste a link to a DJ set:\n"
            "• SoundCloud\n"
            "• Mixcloud\n"
            "• YouTube\n"
            "• Direct audio URL\n\n"
            "<i>Example: https://soundcloud.com/dj/set-name</i>"
        ),
        reply_markup=get_cancel_keyboard()
    )
    await callback.answer()


@router.message(AnalyzeStates.waiting_for_url)
async def process_url(message: Message, state: FSMContext, bot: Bot):
    """Process URL input."""
    if message.from_user.id in banned_users:
        return

    url = message.text.strip()
    user_id = message.from_user.id
    chat_id = message.chat.id

    # Delete user's message
    await delete_user_message(message)

    # Validate URL
    if not url.startswith(("http://", "https://")):
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=(
                "⚠️ <b>Invalid URL</b>\n\n"
                "Please send a valid link starting with http:// or https://\n\n"
                "<i>Try again or press Cancel</i>"
            ),
            reply_markup=get_cancel_keyboard()
        )
        return

    await state.clear()

    # Check disk space
    disk = get_disk_usage()
    if disk["used_percent"] > 90 and not is_admin(user_id):
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=(
                "⚠️ <b>Server disk space critically low</b>\n\n"
                "Please try again later."
            ),
            reply_markup=get_back_keyboard()
        )
        return

    # Queue analysis via ARQ
    try:
        job_id = await enqueue_download_and_analyze(url, user_id)
        set_job_id(job_id)
        logger.info("URL analysis queued", data={"url": url[:100], "job_id": job_id})
    except Exception as e:
        logger.error(f"Failed to queue task: {e}", data={"url": url[:100]})
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text="⚠️ <b>Failed to queue analysis</b>\n\nPlease try again.",
            reply_markup=get_back_keyboard()
        )
        return

    # Store job
    if user_id not in user_jobs:
        user_jobs[user_id] = []
    user_jobs[user_id].append(job_id)

    await ensure_main_message(
        bot=bot,
        user_id=user_id,
        chat_id=chat_id,
        text=(
            f"✅ <b>Analysis queued!</b>\n\n"
            f"🔗 URL: <code>{url[:50]}{'...' if len(url) > 50 else ''}</code>\n"
            f"📋 Job ID: <code>{job_id[:30]}...</code>\n\n"
            f"⏱ <i>This may take 2-5 minutes</i>"
        ),
        reply_markup=get_job_keyboard(job_id)
    )


@router.callback_query(F.data == "my_jobs")
async def cb_my_jobs(callback: CallbackQuery):
    """Show user's jobs."""
    if callback.from_user.id in banned_users:
        return

    user_id = callback.from_user.id
    if user_id not in user_jobs or not user_jobs[user_id]:
        await update_main_message(
            callback=callback,
            text=(
                "<b>📁 My Jobs</b>\n\n"
                "<i>No jobs found.</i>\n\n"
                "Send an audio file or use '🔗 Analyze URL' to start."
            ),
            reply_markup=get_back_keyboard()
        )
        await callback.answer()
        return

    # Build jobs list with status
    lines = []
    for job_id in user_jobs[user_id][-5:]:
        status = await get_job_status(job_id)
        emoji = get_state_emoji(status["state"])
        short_id = job_id[:15] + "..." if len(job_id) > 15 else job_id
        lines.append(f"{emoji} <code>{short_id}</code>\n   └ {status['status']}")

    jobs_list = "\n".join(lines)

    await update_main_message(
        callback=callback,
        text=f"<b>📁 My Jobs</b>\n\n{jobs_list}\n\n<i>Click a job to see details</i>",
        reply_markup=get_jobs_keyboard(user_jobs[user_id])
    )
    await callback.answer()


@router.callback_query(F.data.startswith("job:"))
async def cb_job_detail(callback: CallbackQuery):
    """Show job details with progress bar and ETA."""
    job_id = callback.data.split(":", 1)[1]
    status = await get_job_status(job_id)
    emoji = get_state_emoji(status["state"])

    # Progress bar with ETA
    progress_bar = ""
    if status["state"] == "PROGRESS":
        progress = status.get("progress", 0)
        filled = int(progress / 10)
        eta = status.get("eta", "")
        eta_text = f" • ETA: {eta}" if eta else ""
        progress_bar = f"\n\n{'█' * filled}{'░' * (10 - filled)} {progress}%{eta_text}"

    # Result summary
    result_text = ""
    if status["state"] == "SUCCESS" and status.get("result"):
        result = status["result"]
        elapsed = status.get("elapsed_sec", 0)
        if isinstance(result, dict):
            n_segments = result.get("n_segments", 0)
            total_drops = result.get("total_drops", 0)
            result_text = f"\n\n📊 <b>Results:</b>\n• Segments: {n_segments}\n• Drops: {total_drops}"
            if elapsed:
                result_text += f"\n• Time: {elapsed}s"

    timestamp = datetime.now().strftime("%H:%M:%S")

    await update_main_message(
        callback=callback,
        text=(
            f"<b>Job Details</b> {emoji}\n\n"
            f"📋 ID: <code>{job_id}</code>\n"
            f"📌 Status: {status['status']}"
            f"{progress_bar}{result_text}\n\n"
            f"<i>Updated: {timestamp}</i>"
        ),
        reply_markup=get_job_keyboard(job_id)
    )
    await callback.answer("Status refreshed")


@router.message(F.audio | F.document)
async def handle_file(message: Message, state: FSMContext, bot: Bot):
    """Handle uploaded audio files."""
    if message.from_user.id in banned_users:
        return

    await state.clear()
    user_id = message.from_user.id
    chat_id = message.chat.id

    file = message.audio or message.document
    if not file:
        return

    # Delete the file message (we'll update the main message)
    await delete_user_message(message)

    # Check file size
    if file.file_size and file.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=f"⚠️ <b>File too large</b>\n\nMaximum: {MAX_FILE_SIZE_MB}MB",
            reply_markup=get_back_keyboard()
        )
        return

    # Check extension
    file_name = file.file_name or "unknown"
    ext = os.path.splitext(file_name)[1].lower()
    allowed = {".mp3", ".wav", ".flac", ".m4a", ".opus", ".ogg"}
    if ext not in allowed:
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=(
                f"⚠️ <b>Unsupported format:</b> {ext}\n\n"
                f"Supported: {', '.join(allowed)}"
            ),
            reply_markup=get_back_keyboard()
        )
        return

    # Check disk space
    disk = get_disk_usage()
    if disk["used_percent"] > 90 and not is_admin(user_id):
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text="⚠️ <b>Server disk space critically low</b>",
            reply_markup=get_back_keyboard()
        )
        return

    # Update main message with processing status
    await ensure_main_message(
        bot=bot,
        user_id=user_id,
        chat_id=chat_id,
        text=(
            f"📥 <b>Processing...</b>\n\n"
            f"📄 {file_name}\n"
            f"📦 {file.file_size // (1024*1024) if file.file_size else 0}MB\n\n"
            f"<i>Downloading file...</i>"
        ),
        reply_markup=None
    )

    try:
        # Download file
        downloads_dir = os.getenv("DOWNLOADS_DIR", "/tmp/downloads")
        os.makedirs(downloads_dir, exist_ok=True)

        file_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        local_path = os.path.join(downloads_dir, f"{file_id}{ext}")

        tg_file = await bot.get_file(file.file_id)
        await bot.download_file(tg_file.file_path, local_path)

        # Queue analysis via ARQ
        try:
            job_id = await enqueue_analyze_set(local_path, user_id)
            set_job_id(job_id)
            logger.info("File analysis queued", data={
                "file_name": file_name,
                "file_size_mb": file.file_size // (1024*1024) if file.file_size else 0,
                "job_id": job_id,
            })
        except Exception as e:
            logger.error(f"Failed to queue ARQ task: {e}", data={"file_name": file_name})
            job_id = f"job_{file_id}"  # Fallback ID

        # Store job
        if user_id not in user_jobs:
            user_jobs[user_id] = []
        user_jobs[user_id].append(job_id)

        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=(
                f"✅ <b>Analysis queued!</b>\n\n"
                f"📄 {file_name}\n"
                f"📋 Job: <code>{job_id[:30]}...</code>\n\n"
                f"⏱ <i>This may take 2-5 minutes</i>"
            ),
            reply_markup=get_job_keyboard(job_id)
        )

    except Exception as e:
        logger.error(f"Failed to process file: {e}", exc_info=True, data={"file_name": file_name})
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=f"❌ <b>Failed to process file</b>\n\n{str(e)[:100]}",
            reply_markup=get_back_keyboard()
        )
