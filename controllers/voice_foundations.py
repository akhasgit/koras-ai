"""
Voice Foundations controller — the authenticated variant of `/analyze`
used by korasapi's Voice Foundations background task.

Semantically identical to `handle_analyze` (same audio → features →
transcript → scores → coach feedback pipeline) but:
  1. Accepts multipart form with an `audio` file plus optional
     `activity_id` / `day` metadata for logging/routing.
  2. Sits behind `x-koras-secret` (registered on the route) so the public
     landing-page demo endpoint at `/analyze` isn't the surface used for
     signed-in programme traffic.
  3. Emits its own log stream (`/analyze-voice-foundations`) so we can tell
     public-demo volume apart from in-app programme volume.
"""
from __future__ import annotations

import asyncio
import uuid

from fastapi import HTTPException, Request

from config import MAX_UPLOAD_BYTES, MIN_TRANSCRIPT_WORDS, extract_sem
from models.analyze import AnalyzeResponse
from services.audio import extract_features, normalize_audio, transcribe
from services.llm import analyze_transcript, generate_coach_feedback
from services.scoring import compute_voice_scores, pick_archetype
from utils.logging import log


async def handle_analyze_voice_foundations(request: Request) -> AnalyzeResponse:
    req_id = uuid.uuid4().hex[:8]

    form = await request.form()
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    activity_id = str(form.get("activity_id") or "").strip() or None
    day_raw = form.get("day")
    day: int | None = None
    if day_raw is not None:
        try:
            day = int(str(day_raw))
        except (TypeError, ValueError):
            day = None

    log(
        "/analyze-voice-foundations",
        req_id,
        None,
        event="start",
        activity_id=activity_id,
        day=day,
    )

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    filename = getattr(audio_field, "filename", "audio") or "audio"

    async with extract_sem:
        try:
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, filename)
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        try:
            features, transcript_data = await asyncio.gather(
                asyncio.to_thread(extract_features, wav_bytes),
                transcribe(wav_bytes),
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(500, f"Analysis failed: {e}") from e

    if len(transcript_data["text"].split()) < MIN_TRANSCRIPT_WORDS:
        raise HTTPException(
            400,
            "Audio too short or unclear. Please record at least 10 words.",
        )

    try:
        transcript_analysis = await analyze_transcript(transcript_data["text"])
    except Exception as e:
        raise HTTPException(500, f"Transcript analysis failed: {e}") from e

    scores, metrics = compute_voice_scores(features, transcript_data, transcript_analysis)

    if metrics.duration_seconds > 0:
        transcript_analysis.filler_analysis.rate_per_minute = round(
            transcript_analysis.filler_analysis.count / (metrics.duration_seconds / 60),
            2,
        )

    try:
        coach_feedback = await generate_coach_feedback(scores, transcript_analysis)
    except Exception as e:
        log(
            "/analyze-voice-foundations",
            req_id,
            None,
            event="coach_feedback_fallback",
            error=str(e),
        )
        coach_feedback = (
            "Nice work on this take. Focus on one thing next: slow down slightly "
            "and let your pauses breathe."
        )

    archetype = pick_archetype(scores)
    log(
        "/analyze-voice-foundations",
        req_id,
        None,
        event="done",
        activity_id=activity_id,
        day=day,
        duration=features["duration"],
        wpm=metrics.words_per_minute,
        overall=scores.overall,
        archetype=archetype,
    )

    return AnalyzeResponse(
        scores=scores,
        metrics=metrics,
        transcript=transcript_data["text"],
        transcript_analysis=transcript_analysis,
        coach_feedback=coach_feedback,
        archetype=archetype,
    )
