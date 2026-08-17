import asyncio
import json
import uuid

from fastapi import File, Form, HTTPException, UploadFile

from config import MAX_UPLOAD_BYTES, extract_sem
from models.speech_clarity import (
    AnalyzeClarityReadResponse,
    GenerateClarityDrillsRequest,
    GenerateClarityDrillsResponse,
    GenerateClarityPassageRequest,
    GenerateClarityPassageResponse,
    ScoreClarityDrillResponse,
)
from services.audio import extract_features, normalize_audio, transcribe
from services.listening import synthesize_tts
from services.speech_clarity import (
    analyze_clarity_read,
    generate_clarity_drills,
    generate_clarity_passage,
    score_clarity_drill,
)
from utils.logging import log


async def handle_generate_clarity_passage(
    req: GenerateClarityPassageRequest,
) -> GenerateClarityPassageResponse:
    return await generate_clarity_passage(req)


async def handle_analyze_clarity_read(
    audio: UploadFile = File(...),
    passage: str = Form(...),
    active_features: str = Form("[]"),
) -> AnalyzeClarityReadResponse:
    req_id = uuid.uuid4().hex[:8]
    log("/analyze-clarity-read", req_id, None, event="start")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    try:
        features_list = json.loads(active_features or "[]")
        if not isinstance(features_list, list):
            features_list = []
    except json.JSONDecodeError:
        features_list = []

    async with extract_sem:
        try:
            wav_bytes = await asyncio.to_thread(
                normalize_audio, audio_bytes, audio.filename or "audio",
            )
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e
        try:
            features, transcript_data = await asyncio.gather(
                asyncio.to_thread(extract_features, wav_bytes),
                transcribe(wav_bytes),
            )
        except Exception as e:
            raise HTTPException(500, f"Analysis failed: {e}") from e

    words = len((transcript_data.get("text") or "").split())
    duration = float(features.get("duration") or 0.0)
    if words < 10 or duration < 8:
        raise HTTPException(400, "Audio too short or unclear. Please record at least 10 words.")

    out = await analyze_clarity_read(
        wav_bytes, transcript_data, features, passage, [str(f) for f in features_list],
    )
    log("/analyze-clarity-read", req_id, None, event="done", overall=out.scores.overall)
    return out


async def handle_score_clarity_drill(
    audio: UploadFile = File(...),
    prompt_text: str = Form(""),
) -> ScoreClarityDrillResponse:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    async with extract_sem:
        wav_bytes = await asyncio.to_thread(
            normalize_audio, audio_bytes, audio.filename or "audio",
        )
        features, transcript_data = await asyncio.gather(
            asyncio.to_thread(extract_features, wav_bytes),
            transcribe(wav_bytes),
        )
    return await score_clarity_drill(wav_bytes, transcript_data, features, prompt_text)


async def handle_generate_clarity_drills(
    req: GenerateClarityDrillsRequest,
) -> GenerateClarityDrillsResponse:
    return await generate_clarity_drills(req)


async def handle_synthesize_clarity_audio(text: str, voice: str | None, speed: float) -> bytes:
    return await synthesize_tts(text, voice, speed)
