"""
Generate handlers - Set generation functionality.

Uses single message architecture.
"""

from aiogram import Router, F
from aiogram.types import CallbackQuery

from ..keyboards.inline import get_generate_keyboard, get_back_keyboard
from .start import banned_users, update_main_message
from .analyze import user_jobs

router = Router()


@router.callback_query(F.data == "generate_set")
async def cb_generate_set(callback: CallbackQuery):
    """Show set generation options."""
    if callback.from_user.id in banned_users:
        return

    user_id = callback.from_user.id

    # Check if user has any analyzed sets
    if user_id not in user_jobs or not user_jobs[user_id]:
        await update_main_message(
            callback=callback,
            text=(
                "<b>Generate Set</b>\n\n"
                "<i>No tracks available.</i>\n\n"
                "Analyze some DJ sets first to enable set generation."
            ),
            reply_markup=get_back_keyboard()
        )
        await callback.answer()
        return

    await update_main_message(
        callback=callback,
        text=(
            "<b>Generate Set</b>\n\n"
            "Select set duration:\n\n"
            "The generator will create a set plan based on:\n"
            "- Your DJ profile energy arc\n"
            "- Track compatibility scores\n"
            "- Transition quality predictions\n\n"
            "<i>Choose duration below:</i>"
        ),
        reply_markup=get_generate_keyboard()
    )
    await callback.answer()


@router.callback_query(F.data.startswith("gen:"))
async def cb_generate_duration(callback: CallbackQuery):
    """Generate set with specified duration."""
    if callback.from_user.id in banned_users:
        return

    duration = int(callback.data.split(":")[1])

    await update_main_message(
        callback=callback,
        text=(
            f"<b>Generating {duration}-minute Set...</b>\n\n"
            "Analyzing track compatibility...\n"
            "Computing transition scores...\n"
            "Building energy arc...\n\n"
            "<i>This may take a moment.</i>"
        ),
        reply_markup=get_back_keyboard()
    )

    try:
        from app.modules.generation.pipelines.set_generator import SetGeneratorPipeline

        # Get user DJ profile
        user_id = callback.from_user.id
        dj_name = f"user_{user_id}"

        # Generate set plan
        generator = SetGeneratorPipeline(cache_dir="cache")
        plan = generator.generate_plan(
            dj_name=dj_name,
            target_duration_min=duration,
            rekordbox_xml=None,  # Could be set from user config
        )

        if plan and plan.tracks:
            # Vectorized formatting using numpy string operations
            import numpy as np

            tracks = plan.tracks
            phases = np.array([t.phase for t in tracks])
            titles = np.array([t.title or 'Unknown' for t in tracks])
            bpms = np.array([t.bpm for t in tracks])
            keys = np.array([t.key or '?' for t in tracks])

            # Find phase boundaries (where phase changes)
            phase_changes = np.concatenate([[True], phases[1:] != phases[:-1]])
            phase_headers = np.where(phase_changes, np.char.add('\n<b>', np.char.upper(phases.astype(str))), '')
            phase_headers = np.char.add(phase_headers, np.where(phase_changes, '</b>\n', ''))

            # Format track lines vectorized
            indices = np.arange(1, len(tracks) + 1).astype(str)
            track_lines = np.char.add(indices, '. ')
            track_lines = np.char.add(track_lines, titles)
            track_lines = np.char.add(track_lines, ' (')
            track_lines = np.char.add(track_lines, bpms.astype(int).astype(str))
            track_lines = np.char.add(track_lines, ' BPM, ')
            track_lines = np.char.add(track_lines, keys)
            track_lines = np.char.add(track_lines, ')')

            # Combine headers and tracks
            combined = np.char.add(phase_headers, track_lines)
            phases_text = '\n'.join(combined)

            await update_main_message(
                callback=callback,
                text=(
                    f"<b>Set Plan ({duration} min)</b>\n\n"
                    f"{''.join(phases_text)}\n\n"
                    f"<i>Confidence: {plan.confidence:.0%}</i>"
                ),
                reply_markup=get_back_keyboard()
            )
        else:
            await update_main_message(
                callback=callback,
                text=(
                    f"<b>Set Plan ({duration} min)</b>\n\n"
                    "<i>No tracks available for set generation.</i>\n"
                    "Analyze some DJ sets first to build your profile."
                ),
                reply_markup=get_back_keyboard()
            )
    except Exception as e:
        await update_main_message(
            callback=callback,
            text=f"<b>Generation Failed</b>\n\n{str(e)[:100]}",
            reply_markup=get_back_keyboard()
        )

    await callback.answer()
