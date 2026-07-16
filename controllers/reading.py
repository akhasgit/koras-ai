"""
Controllers for the Reading programme endpoints on koras-ai.

`/analyze-reading` mirrors the vocabulary controllers' audio pipeline —
normalise + Whisper + CPU-heavy scoring under `extract_sem` and off the
event loop. The two generation endpoints wrap the strict-JSON services;
their failures surface as 500 so koras-api can apply its template fallback.
"""

import asyncio
import json
import uuid

from fastapi import File, Form, HTTPException, UploadFile

from config import MAX_UPLOAD_BYTES, extract_sem
from models.reading import (
    GenerateReadingProgramRequest,
    GenerateReadingProgramResponse,
    GenerateReadingStageRequest,
    GenerateReadingStageResponse,
    ReadingAnalysisResponse,
)
from services.audio import normalize_audio, transcribe
from services.reading_analysis import (
    compute_reading_metrics,
    generate_reading_feedback,
)
from services.reading_generation import (
    generate_reading_program,
    generate_reading_stage,
)
from utils.logging import log

ATTEMPT_TYPES = {
    "calibration", "free_read", "guided_read", "echo",
    "cold_read", "punctuation", "speed_ladder", "vocab_context",
}


# ─── /analyze-reading ─────────────────────────────────────────────────────────


async def handle_analyze_reading(
    audio: UploadFile = File(...),
    passage_text: str = Form(...),
    attempt_type: str = Form(...),
    guide_wpm: int | None = Form(None),
    level_hint: str | None = Form(None),
    target_contour: str | None = Form(None),
) -> ReadingAnalysisResponse:
    req_id = uuid.uuid4().hex[:8]
    endpoint = "/analyze-reading"

    passage = (passage_text or "").strip()
    if not passage:
        raise HTTPException(400, "passage_text is required.")
    kind = (attempt_type or "").strip()
    if kind not in ATTEMPT_TYPES:
        raise HTTPException(400, f"Unsupported attempt_type: {kind}")

    contour: list[float] | None = None
    if target_contour:
        try:
            parsed = json.loads(target_contour)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"target_contour must be valid JSON: {e}") from e
        if not isinstance(parsed, list) or not all(isinstance(v, (int, float)) for v in parsed):
            raise HTTPException(400, "target_contour must be a JSON array of numbers.")
        contour = [float(v) for v in parsed] or None

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    log(
        endpoint, req_id, None, event="start",
        attempt_type=kind, guide_wpm=guide_wpm, level_hint=level_hint,
        has_contour=contour is not None,
    )

    async with extract_sem:
        try:
            wav_bytes = await asyncio.to_thread(
                normalize_audio, audio_bytes, audio.filename or "audio",
            )
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        try:
            transcript_data = await transcribe(wav_bytes)
            metrics = await asyncio.to_thread(
                compute_reading_metrics,
                passage, transcript_data, wav_bytes,
                kind, guide_wpm, level_hint, contour,
            )
        except Exception as e:
            log(endpoint, req_id, None, event="pipeline_error", error=str(e))
            raise HTTPException(500, f"Reading analysis failed: {e}") from e

    coach_feedback, chips = await generate_reading_feedback(metrics, kind, level_hint)

    log(
        endpoint, req_id, None, event="done",
        match=metrics["match_pct"], hesitance=metrics["hesitance_score"],
        pace=metrics["pace_score"], flow=metrics["flow_score"],
        overall=metrics["overall_score"],
    )
    return ReadingAnalysisResponse(**metrics, coach_feedback=coach_feedback, chips=chips)


# ─── /generate-reading-program ────────────────────────────────────────────────


async def handle_generate_reading_program(
    req: GenerateReadingProgramRequest,
) -> GenerateReadingProgramResponse:
    req_id = uuid.uuid4().hex[:8]
    endpoint = "/generate-reading-program"
    log(
        endpoint, req_id, None, event="start",
        persona=req.profile.persona, grade_level=req.profile.grade_level,
        has_baseline=req.baseline is not None,
        exclude_count=len(req.exclude_words),
    )

    try:
        out = await generate_reading_program(req)
    except Exception as e:
        log(endpoint, req_id, None, event="error", error=str(e))
        raise HTTPException(500, f"Reading programme generation failed: {e}") from e

    log(
        endpoint, req_id, None, event="done",
        stages=len(out.skeleton), lessons=len(out.stage_1_content.lessons),
    )
    return out


# ─── /generate-reading-stage ──────────────────────────────────────────────────


async def handle_generate_reading_stage(
    req: GenerateReadingStageRequest,
) -> GenerateReadingStageResponse:
    req_id = uuid.uuid4().hex[:8]
    endpoint = "/generate-reading-stage"
    log(
        endpoint, req_id, None, event="start",
        position=req.skeleton_entry.position,
        has_performance=req.completed_stage_performance is not None,
    )

    try:
        out = await generate_reading_stage(req)
    except Exception as e:
        log(endpoint, req_id, None, event="error", error=str(e))
        raise HTTPException(500, f"Reading stage generation failed: {e}") from e

    log(endpoint, req_id, None, event="done", lessons=len(out.stage_content.lessons))
    return out
