"""
Profile handlers - DJ profile viewing.

Uses single message architecture.
"""

from aiogram import Router, F
from aiogram.types import CallbackQuery

from ..keyboards.inline import get_profile_keyboard, get_back_keyboard
from .start import banned_users, update_main_message
from .analyze import user_jobs

router = Router()


@router.callback_query(F.data == "profile")
async def cb_profile(callback: CallbackQuery):
    """Show DJ profile overview."""
    if callback.from_user.id in banned_users:
        return

    user_id = callback.from_user.id

    # Check if user has any analyzed sets
    if user_id not in user_jobs or not user_jobs[user_id]:
        await update_main_message(
            callback=callback,
            text=(
                "<b>DJ Profile</b>\n\n"
                "<i>No profile data yet.</i>\n\n"
                "Analyze some DJ sets first to build your profile."
            ),
            reply_markup=get_back_keyboard()
        )
        await callback.answer()
        return

    # Get profile from cache
    try:
        from app.core.connectors import SQLiteCache
        cache = SQLiteCache()
        profile = cache.get_profile(str(user_id))

        if profile:
            # Format profile data
            energy_arc = profile.get('energy_arc', {})
            drop_pattern = profile.get('drop_pattern', {})
            tempo = profile.get('tempo', {})

            text = (
                f"<b>DJ Profile</b>\n\n"
                f"Sets analyzed: {len(user_jobs[user_id])}\n\n"
                f"<b>Energy Arc:</b> {energy_arc.get('shape', 'N/A')}\n"
                f"   Opening: {energy_arc.get('opening', 0):.0%}\n"
                f"   Peak: {energy_arc.get('peak', 0):.0%}\n"
                f"   Closing: {energy_arc.get('closing', 0):.0%}\n\n"
                f"<b>Drop Style:</b> {drop_pattern.get('style', 'N/A')}\n"
                f"   Drops/hour: {drop_pattern.get('per_hour', 0):.1f}\n\n"
                f"<b>Tempo:</b> {tempo.get('min', 0):.0f}-{tempo.get('max', 0):.0f} BPM\n"
                f"   Dominant: {tempo.get('dominant', 0):.0f} BPM"
            )
        else:
            text = (
                f"<b>DJ Profile</b>\n\n"
                f"Sets analyzed: {len(user_jobs[user_id])}\n\n"
                "<i>Profile data being computed...</i>\n"
                "Complete more analyses to build your profile."
            )

        await update_main_message(
            callback=callback,
            text=text,
            reply_markup=get_profile_keyboard()
        )
    except Exception as e:
        await update_main_message(
            callback=callback,
            text=(
                f"<b>DJ Profile</b>\n\n"
                f"Sets analyzed: {len(user_jobs[user_id])}\n\n"
                f"<i>Error loading profile: {str(e)[:50]}</i>"
            ),
            reply_markup=get_back_keyboard()
        )

    await callback.answer()


@router.callback_query(F.data == "profile:energy")
async def cb_profile_energy(callback: CallbackQuery):
    """Show energy arc details."""
    if callback.from_user.id in banned_users:
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Energy Arc Analysis</b>\n\n"
            "Your energy flow patterns:\n\n"
            "- <b>Crescendo</b> - Building energy throughout\n"
            "- <b>Peak & Fade</b> - Early peak, gradual decline\n"
            "- <b>Plateau</b> - Consistent energy level\n"
            "- <b>Journey</b> - Multiple peaks and valleys\n\n"
            "<i>Analyze more sets to refine your profile.</i>"
        ),
        reply_markup=get_profile_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "profile:drops")
async def cb_profile_drops(callback: CallbackQuery):
    """Show drop pattern details."""
    if callback.from_user.id in banned_users:
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Drop Pattern Analysis</b>\n\n"
            "Your drop usage patterns:\n\n"
            "- <b>Festival</b> - High density, clustered drops\n"
            "- <b>Technical</b> - Precise, spaced drops\n"
            "- <b>Minimal</b> - Sparse, impactful drops\n"
            "- <b>Progressive</b> - Building intensity\n\n"
            "<i>Analyze more sets to refine your profile.</i>"
        ),
        reply_markup=get_profile_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "profile:tempo")
async def cb_profile_tempo(callback: CallbackQuery):
    """Show tempo analysis details."""
    if callback.from_user.id in banned_users:
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Tempo Analysis</b>\n\n"
            "Your BPM preferences:\n\n"
            "- Tempo range and variance\n"
            "- Dominant BPM (most common)\n"
            "- Tempo trajectory over sets\n"
            "- BPM distribution histogram\n\n"
            "<i>Analyze more sets to refine your profile.</i>"
        ),
        reply_markup=get_profile_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data == "profile:keys")
async def cb_profile_keys(callback: CallbackQuery):
    """Show key analysis details."""
    if callback.from_user.id in banned_users:
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Key Analysis</b>\n\n"
            "Your harmonic preferences:\n\n"
            "- Dominant key (pitch center)\n"
            "- Camelot wheel positions\n"
            "- Key stability score\n"
            "- Top 5 keys used\n\n"
            "<i>Analyze more sets to refine your profile.</i>"
        ),
        reply_markup=get_profile_keyboard()
    )
    await callback.answer()
