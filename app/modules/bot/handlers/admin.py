"""
Admin handlers - Admin panel functionality.

Uses single message architecture.
"""

import os
import logging
import sys

from aiogram import Router, F, Bot
from aiogram.types import CallbackQuery, Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from ..keyboards.inline import get_admin_keyboard, get_back_keyboard, get_cancel_keyboard
from .start import (
    is_admin,
    banned_users,
    update_main_message,
    ensure_main_message,
    delete_user_message,
)
from .analyze import user_jobs, get_disk_usage

router = Router()
logger = logging.getLogger(__name__)


class AdminStates(StatesGroup):
    """FSM states for admin actions."""
    waiting_for_ban_id = State()


@router.callback_query(F.data == "admin")
async def cb_admin(callback: CallbackQuery):
    """Admin panel."""
    if not is_admin(callback.from_user.id):
        await callback.answer("Access denied", show_alert=True)
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Admin Panel</b>\n\n"
            "Select an option:"
        ),
        reply_markup=get_admin_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "admin:stats")
async def cb_admin_stats(callback: CallbackQuery):
    """Admin statistics."""
    if not is_admin(callback.from_user.id):
        await callback.answer("Access denied", show_alert=True)
        return

    total_jobs = sum(len(jobs) for jobs in user_jobs.values())

    # Try to get ARQ/Redis status
    queue_status = "Not available"
    try:
        from app.services.arq_worker import get_redis_settings
        settings = get_redis_settings()
        queue_status = f"ARQ ({settings.host}:{settings.port})"
    except Exception:
        pass

    await update_main_message(
        callback=callback,
        text=(
            f"<b>System Statistics</b>\n\n"
            f"Total users: {len(user_jobs)}\n"
            f"Total jobs: {total_jobs}\n"
            f"Banned users: {len(banned_users)}\n"
            f"Queue: {queue_status}"
        ),
        reply_markup=get_admin_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "admin:disk")
async def cb_admin_disk(callback: CallbackQuery):
    """Admin disk usage."""
    if not is_admin(callback.from_user.id):
        await callback.answer("Access denied", show_alert=True)
        return

    disk = get_disk_usage()

    await update_main_message(
        callback=callback,
        text=(
            f"<b>Disk Usage</b>\n\n"
            f"Total: {disk['total_gb']}GB\n"
            f"Used: {disk['used_gb']}GB ({disk['used_percent']}%)\n"
            f"Free: {disk['free_gb']}GB\n"
            f"Downloads: {disk['downloads_mb']}MB"
        ),
        reply_markup=get_admin_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "admin:cache")
async def cb_admin_cache(callback: CallbackQuery):
    """Clear cache."""
    if not is_admin(callback.from_user.id):
        await callback.answer("Access denied", show_alert=True)
        return

    try:
        from app.core.connectors import SQLiteCache
        cache = SQLiteCache()
        cache.clear()
        await callback.answer("Cache cleared!", show_alert=True)
        logger.info("Cache cleared by admin")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        await callback.answer(f"Failed: {str(e)[:50]}", show_alert=True)


@router.callback_query(F.data == "admin:ban")
async def cb_admin_ban(callback: CallbackQuery, state: FSMContext):
    """Start ban user flow."""
    if not is_admin(callback.from_user.id):
        await callback.answer("Access denied", show_alert=True)
        return

    await state.set_state(AdminStates.waiting_for_ban_id)

    await update_main_message(
        callback=callback,
        text=(
            "<b>Ban User</b>\n\n"
            "Send the Telegram user ID to ban.\n\n"
            "<i>Example: 123456789</i>"
        ),
        reply_markup=get_cancel_keyboard()
    )
    await callback.answer()


@router.message(AdminStates.waiting_for_ban_id)
async def process_ban_id(message: Message, state: FSMContext, bot: Bot):
    """Process ban user ID input."""
    if not is_admin(message.from_user.id):
        return

    user_id = message.from_user.id
    chat_id = message.chat.id

    # Delete user message
    await delete_user_message(message)

    try:
        ban_user_id = int(message.text.strip())

        if ban_user_id == user_id:
            await ensure_main_message(
                bot=bot,
                user_id=user_id,
                chat_id=chat_id,
                text="You cannot ban yourself!",
                reply_markup=get_admin_keyboard()
            )
            await state.clear()
            return

        banned_users.add(ban_user_id)
        await state.clear()

        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text=f"User <code>{ban_user_id}</code> has been banned.",
            reply_markup=get_admin_keyboard()
        )
        logger.info(f"User {ban_user_id} banned by admin {user_id}")

    except ValueError:
        await ensure_main_message(
            bot=bot,
            user_id=user_id,
            chat_id=chat_id,
            text="Invalid user ID. Please send a number.",
            reply_markup=get_cancel_keyboard()
        )


@router.callback_query(F.data == "admin:restart")
async def cb_admin_restart(callback: CallbackQuery):
    """Restart services."""
    if not is_admin(callback.from_user.id):
        await callback.answer("Access denied", show_alert=True)
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Restarting...</b>\n\n"
            "The bot will restart in a few seconds."
        ),
        reply_markup=get_back_keyboard()
    )
    await callback.answer("Restarting...", show_alert=True)

    logger.info(f"Restart requested by admin {callback.from_user.id}")

    # Schedule restart
    import asyncio

    async def restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, ['python'] + sys.argv)

    asyncio.create_task(restart())
