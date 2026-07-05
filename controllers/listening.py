"""
Controllers for the Listening Comprehension endpoints.

Two endpoints:
  - POST /synthesize-listening-audio (JSON in, MP3 bytes out)
  - POST /analyze-listening-response (multipart audio OR JSON transcript)
"""
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import HTTPException, Request
from fastapi.responses import Response

from config import MAX_UPLOAD_BYTES, extract_sem
from models.listening import (
    AnalyzeListeningResponseJSONRequest,
    ListeningVoiceAnalysis,
    SynthesizeListeningAudioRequest,
)
from services.audio import normalize_audio, transcribe
from services.listening import grade_listening_answer, synthesize_tts
from utils.logging import log


# ─── TTS ────────────────────────────────────────────────────────────────────


async def handle_synthesize_listening_audio(
    req: SynthesizeListeningAudioRequest,
) -> Response:
    req_id = uuid.uuid4().hex[:8]
    log(
        "/synthesize-listening-audio",
        req_id,
        None,
        chars=len(req.text or ""),
        voice=req.voice,
        speed=req.speed,
    )

    try:
        audio_bytes = await synthesize_tts(req.text, req.voice, req.speed)
    except Exception as e:  # noqa: BLE001
        log("/synthesize-listening-audio", req_id, None, event="error", error=str(e)[:200])
        raise HTTPException(502, f"TTS synthesis failed: {e}") from e

    if not audio_bytes:
        raise HTTPException(502, "TTS synthesis returned empty audio.")

    log(
        "/synthesize-listening-audio",
        req_id,
        None,
        event="done",
        bytes=len(audio_bytes),
    )
    return Response(content=audio_bytes, media_type="audio/mpeg")


# ─── Voice grading ──────────────────────────────────────────────────────────


def _parse_expected_points(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception:
        pass
    return [line.strip() for line in raw.splitlines() if line.strip()]


async def handle_analyze_listening_response(request: Request) -> dict:
    req_id = uuid.uuid4().hex[:8]
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = await request.json()
        req = AnalyzeListeningResponseJSONRequest(**body)
        log(
            "/analyze-listening-response",
            req_id,
            None,
            mode="json",
            skill=req.target_skill,
        )
        transcript = (req.transcript or "").strip()
        if len(transcript.split()) < 3:
            raise HTTPException(
                400,
                "Transcript too short to grade for comprehension (minimum 3 words).",
            )
        analysis = await grade_listening_answer(
            transcript=transcript,
            question_prompt=req.question_prompt,
            passage_gist=req.passage_gist or "",
            expected_points=req.expected_points or [],
            target_skill=req.target_skill or "main_idea",
        )
        return {**analysis.model_dump(), "transcript": transcript}

    form = await request.form()
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    question_prompt = str(form.get("question_prompt") or "").strip()
    if not question_prompt:
        raise HTTPException(400, "Missing 'question_prompt' field.")

    passage_gist = str(form.get("passage_gist") or "")
    target_skill = str(form.get("target_skill") or "main_idea").strip() or "main_idea"
    expected_points = _parse_expected_points(str(form.get("expected_points") or ""))

    log(
        "/analyze-listening-response",
        req_id,
        None,
        mode="audio",
        skill=target_skill,
    )

    async with extract_sem:
        try:
            filename = getattr(audio_field, "filename", "audio") or "audio"
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, filename)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        try:
            transcript_data = await transcribe(wav_bytes)
        except Exception as e:
            raise HTTPException(500, f"Transcription failed: {e}") from e

    transcript_text = (transcript_data.get("text") or "").strip()
    if len(transcript_text.split()) < 3:
        raise HTTPException(
            400,
            "We couldn't detect a meaningful spoken answer. Please try again.",
        )

    analysis = await grade_listening_answer(
        transcript=transcript_text,
        question_prompt=question_prompt,
        passage_gist=passage_gist,
        expected_points=expected_points,
        target_skill=target_skill,
    )

    log(
        "/analyze-listening-response",
        req_id,
        None,
        event="done",
        skill=target_skill,
        words=len(transcript_text.split()),
        comp=analysis.comprehension_score,
    )

    # Piggy-back the transcript so koras-api can persist it without
    # re-transcribing. The route returns dict (no strict response_model)
    # so this passes through as-is.
    return {**analysis.model_dump(), "transcript": transcript_text}
