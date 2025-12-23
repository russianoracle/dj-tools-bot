"""
Inline keyboards for Telegram bot.
"""

from typing import List
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def get_main_keyboard(is_admin: bool = False) -> InlineKeyboardMarkup:
    """Main menu keyboard."""
    buttons = [
        [InlineKeyboardButton(text="🔗 Analyze URL", callback_data="analyze_url")],
        [InlineKeyboardButton(text="📁 My Jobs", callback_data="my_jobs")],
        [InlineKeyboardButton(text="👤 DJ Profile", callback_data="profile")],
        [InlineKeyboardButton(text="🎯 Generate Set", callback_data="generate_set")],
        [InlineKeyboardButton(text="❓ Help", callback_data="help")],
    ]
    if is_admin:
        buttons.append([InlineKeyboardButton(text="⚙️ Admin Panel", callback_data="admin")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_back_keyboard() -> InlineKeyboardMarkup:
    """Back to main menu keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Back to Menu", callback_data="main_menu")]
    ])


def get_jobs_keyboard(jobs: List[str]) -> InlineKeyboardMarkup:
    """Jobs list with refresh button."""
    buttons = []
    for job_id in jobs[-5:]:
        short_id = job_id[:20] + "..." if len(job_id) > 20 else job_id
        buttons.append([InlineKeyboardButton(text=f"📋 {short_id}", callback_data=f"job:{job_id}")])
    buttons.append([InlineKeyboardButton(text="🔄 Refresh", callback_data="my_jobs")])
    buttons.append([InlineKeyboardButton(text="◀️ Back", callback_data="main_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def get_job_keyboard(job_id: str) -> InlineKeyboardMarkup:
    """Single job view keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Refresh Status", callback_data=f"job:{job_id}")],
        [InlineKeyboardButton(text="◀️ Back to Jobs", callback_data="my_jobs")]
    ])


def get_result_keyboard(job_id: str) -> InlineKeyboardMarkup:
    """Navigation for analysis results."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Overview", callback_data=f"result:overview:{job_id}")],
        [InlineKeyboardButton(text="🎚️ Transitions", callback_data=f"result:trans:{job_id}")],
        [InlineKeyboardButton(text="💥 Drops", callback_data=f"result:drops:{job_id}")],
        [InlineKeyboardButton(text="🎵 Genres", callback_data=f"result:genres:{job_id}")],
        [InlineKeyboardButton(text="📈 Energy", callback_data=f"result:energy:{job_id}")],
        [InlineKeyboardButton(text="◀️ Back", callback_data="my_jobs")]
    ])


def get_admin_keyboard() -> InlineKeyboardMarkup:
    """Admin panel keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Statistics", callback_data="admin:stats")],
        [InlineKeyboardButton(text="💾 Disk Usage", callback_data="admin:disk")],
        [InlineKeyboardButton(text="🗑️ Clear Cache", callback_data="admin:cache")],
        [InlineKeyboardButton(text="🚫 Ban User", callback_data="admin:ban")],
        [InlineKeyboardButton(text="🔄 Restart", callback_data="admin:restart")],
        [InlineKeyboardButton(text="◀️ Back", callback_data="main_menu")]
    ])


def get_cancel_keyboard() -> InlineKeyboardMarkup:
    """Cancel action keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="❌ Cancel", callback_data="main_menu")]
    ])


def get_profile_keyboard() -> InlineKeyboardMarkup:
    """DJ profile navigation keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📈 Energy Arc", callback_data="profile:energy")],
        [InlineKeyboardButton(text="💥 Drop Pattern", callback_data="profile:drops")],
        [InlineKeyboardButton(text="🎹 Tempo", callback_data="profile:tempo")],
        [InlineKeyboardButton(text="🔑 Keys", callback_data="profile:keys")],
        [InlineKeyboardButton(text="◀️ Back", callback_data="main_menu")]
    ])


def get_generate_keyboard() -> InlineKeyboardMarkup:
    """Set generation options keyboard."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⏱ 30 min", callback_data="gen:30")],
        [InlineKeyboardButton(text="⏱ 60 min", callback_data="gen:60")],
        [InlineKeyboardButton(text="⏱ 90 min", callback_data="gen:90")],
        [InlineKeyboardButton(text="◀️ Back", callback_data="main_menu")]
    ])
