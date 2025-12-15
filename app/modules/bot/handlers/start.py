"""
Start handler - /start command and main menu.

IMPORTANT: Single message architecture - bot maintains ONE message per user
that gets edited for all interactions. User messages are deleted.
"""

import time
from aiogram import Router, F, Bot
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext

from ..keyboards.inline import get_main_keyboard, get_back_keyboard

router = Router()

# In-memory banned users (use Redis in production)
banned_users: set = set()

# Single message ID per user (use Redis in production)
user_main_message: dict = {}  # user_id -> message_id

# Debounce: last /start time per user
_last_start_time: dict = {}  # user_id -> timestamp
START_DEBOUNCE_SEC = 1.0


def is_admin(user_id: int) -> bool:
    """Check if user is admin."""
    import os
    admin_id = int(os.getenv("ADMIN_USER_ID", "0"))
    return user_id == admin_id


def get_main_text() -> str:
    """Get main menu text."""
    return (
        "<b>🎵 DJ Set Analyzer</b>\n\n"
        "Analyze DJ sets and classify tracks by energy zones:\n\n"
        "🟨 <b>Yellow</b> - Rest zone (low energy)\n"
        "🟩 <b>Green</b> - Transitional (medium)\n"
        "🟪 <b>Purple</b> - Hits/Energy (high)\n\n"
        "<i>Send an audio file or use buttons below:</i>"
    )


async def ensure_main_message(
    bot: Bot,
    user_id: int,
    chat_id: int,
    text: str,
    reply_markup,
    force_new: bool = False
) -> int:
    """
    Ensure user has a single main message.
    Returns the message ID.

    - If force_new: delete old message and create new
    - Otherwise: try to edit, create new if fails
    """
    old_msg_id = user_main_message.get(user_id)

    if force_new and old_msg_id:
        try:
            await bot.delete_message(chat_id, old_msg_id)
        except Exception:
            pass
        old_msg_id = None

    if old_msg_id:
        try:
            await bot.edit_message_text(
                text=text,
                chat_id=chat_id,
                message_id=old_msg_id,
                reply_markup=reply_markup
            )
            return old_msg_id
        except Exception:
            # Message doesn't exist or can't be edited
            pass

    # Create new message
    msg = await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=reply_markup
    )
    user_main_message[user_id] = msg.message_id
    return msg.message_id


async def update_main_message(
    callback: CallbackQuery,
    text: str,
    reply_markup
) -> None:
    """Update the main message via callback (edit in place)."""
    user_id = callback.from_user.id

    try:
        await callback.message.edit_text(
            text=text,
            reply_markup=reply_markup
        )
        # Update stored message ID
        user_main_message[user_id] = callback.message.message_id
    except Exception:
        pass


async def delete_user_message(message: Message) -> None:
    """Delete user's text message to keep chat clean."""
    try:
        await message.delete()
    except Exception:
        pass


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, bot: Bot):
    """Handle /start command - show main menu."""
    user_id = message.from_user.id

    if user_id in banned_users:
        return

    # Debounce: ignore rapid /start commands
    now = time.time()
    last_time = _last_start_time.get(user_id, 0)
    if now - last_time < START_DEBOUNCE_SEC:
        await delete_user_message(message)
        return
    _last_start_time[user_id] = now

    await state.clear()

    # Delete the /start command message
    await delete_user_message(message)

    # Create new main message (force new on /start)
    await ensure_main_message(
        bot=bot,
        user_id=user_id,
        chat_id=message.chat.id,
        text=get_main_text(),
        reply_markup=get_main_keyboard(is_admin(user_id)),
        force_new=True
    )


@router.callback_query(F.data == "main_menu")
async def cb_main_menu(callback: CallbackQuery, state: FSMContext):
    """Back to main menu."""
    if callback.from_user.id in banned_users:
        return

    await state.clear()

    await update_main_message(
        callback=callback,
        text=get_main_text(),
        reply_markup=get_main_keyboard(is_admin(callback.from_user.id))
    )
    await callback.answer()


@router.callback_query(F.data == "help")
async def cb_help(callback: CallbackQuery):
    """Show help information."""
    await update_main_message(
        callback=callback,
        text=(
            "<b>❓ How to use:</b>\n\n"
            "<b>Option 1:</b> Send audio file directly\n"
            "• Supported: MP3, WAV, FLAC, M4A, OPUS\n"
            "• Max size: 500MB\n\n"
            "<b>Option 2:</b> Analyze by URL\n"
            "• Click '🔗 Analyze URL' button\n"
            "• Paste SoundCloud/Mixcloud link\n\n"
            "<b>After analysis:</b>\n"
            "• Track boundaries detected\n"
            "• Drops identified\n"
            "• Energy zones classified\n\n"
            "<i>Analysis takes 2-5 minutes</i>"
        ),
        reply_markup=get_back_keyboard()
    )
    await callback.answer()
