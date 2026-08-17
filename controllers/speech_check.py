import asyncio
import uuid

from fastapi import File, Form, HTTPException, UploadFile

from models.speech_check import (
    AnalyzeSpeechCheckResponse,
    RescoreSpeechCheckRequest,
    RescoreSpeechCheckResponse,
)
from services.audio import extract_features, normalize_audio, transcribe
from services.speech_check import analyze_speech_check_with_coach, rescore_speech_check
from utils.logging import log
from config import MAX_UPLOAD_BYTES, extract_sem


async def handle_analyze_speech_check(
    audio: UploadFile = File(...),
    mode: str = Form("talk"),
    passage: str = Form(""),
    prompt: str = Form(""),
) -> AnalyzeSpeechCheckResponse:
    req_id = uuid.uuid4().hex[:8]
    log("/analyze-speech-check", req_id, None, event="start", mode=mode)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    picked_mode = "read" if (mode or "").strip().lower() == "read" else "talk"
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

    out = await analyze_speech_check_with_coach(
        transcript_data,
        features,
        picked_mode,
        passage or None,
        prompt or passage or "",
    )
    log(
        "/analyze-speech-check", req_id, None, event="done",
        overall=out.scores.overall, too_short=out.too_short, off_script=out.off_script,
    )
    return out


async def handle_rescore_speech_check(req: RescoreSpeechCheckRequest) -> RescoreSpeechCheckResponse:
    return rescore_speech_check(req)
