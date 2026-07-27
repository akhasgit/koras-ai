"""
Controllers for the Daily Vocabulary endpoints on koras-ai.

Wraps the pure-function services in HTTP-shaped orchestrators —
audio decode, Whisper transcription, and feature extraction happen
here in `asyncio.to_thread` / `asyncio.gather` so the CPU-heavy work
runs off the event loop, exactly like `controllers/analyze.py`.
"""

import asyncio
import json
import uuid

from fastapi import File, Form, HTTPException, UploadFile

from config import MAX_UPLOAD_BYTES, extract_sem
from models.vocabulary import (
    AnalyzePronunciationResponse,
    AnalyzeVocabularySentenceResponse,
    AnalyzeVocabularySpeechResponse,
    GenerateVocabularyRequest,
    GenerateVocabularyResponse,
)
from services.audio import extract_features, normalize_audio, transcribe
from services.vocabulary import (
    analyze_pronunciation_sync,
    analyze_vocabulary_sentence,
    analyze_vocabulary_speech,
    generate_vocabulary_words,
)
from utils.logging import log


async def handle_generate_vocabulary_words(
    req: GenerateVocabularyRequest,
) -> GenerateVocabularyResponse:
    req_id = uuid.uuid4().hex[:8]
    log(
        "/generate-vocabulary-words",
        req_id,
        req.user_id,
        event="start",
        segment=req.segment,
        cefr_target=req.cefr_target,
        grade_level=req.grade_level,
        goals_count=len(req.goals or []),
        weak_count=len(req.weak_areas or []),
        exclude_count=len(req.exclude_words or []),
        count=req.count,
    )

    if req.count < 1 or req.count > 10:
        # 5 is the production count; anything outside [1,10] is a bug
        # in the calling code rather than a user-facing problem.
        raise HTTPException(400, "count must be between 1 and 10.")

    try:
        out = await generate_vocabulary_words(req)
    except Exception as e:
        log("/generate-vocabulary-words", req_id, req.user_id, event="error", error=str(e))
        raise HTTPException(500, f"Vocabulary generation failed: {e}") from e

    log(
        "/generate-vocabulary-words",
        req_id,
        req.user_id,
        event="done",
        returned=len(out.words),
    )
    return out


# ─── Audio pipeline (shared by the three analyze handlers) ──────────────────


async def _decode_and_transcribe(
    audio: UploadFile,
    req_id: str,
    endpoint: str,
) -> tuple[dict, dict]:
    """
    Normalise → extract_features → transcribe, exactly like
    `controllers/analyze.py::handle_analyze`. Returns
    (features, transcript_data).
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    async with extract_sem:
        try:
            wav_bytes = await asyncio.to_thread(
                normalize_audio, audio_bytes, audio.filename or "audio",
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        try:
            features, transcript_data = await asyncio.gather(
                asyncio.to_thread(extract_features, wav_bytes),
                transcribe(wav_bytes),
            )
        except Exception as e:
            log(endpoint, req_id, None, event="pipeline_error", error=str(e))
            raise HTTPException(500, f"Analysis failed: {e}") from e

    return features, transcript_data


# ─── /analyze-pronunciation ─────────────────────────────────────────────────


async def handle_analyze_pronunciation(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
) -> AnalyzePronunciationResponse:
    req_id = uuid.uuid4().hex[:8]
    endpoint = "/analyze-pronunciation"
    word = (target_word or "").strip()
    if not word:
        raise HTTPException(400, "target_word is required.")

    log(endpoint, req_id, None, event="start", target_word=word)

    features, transcript_data = await _decode_and_transcribe(audio, req_id, endpoint)

    out = analyze_pronunciation_sync(word, transcript_data, features)
    log(
        endpoint, req_id, None, event="done",
        matched=out.matched, similarity=out.similarity, score=out.pronunciation_score,
    )
    return out


# ─── /analyze-vocabulary-sentence ───────────────────────────────────────────


async def handle_analyze_vocabulary_sentence(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    part_of_speech: str = Form(""),
) -> AnalyzeVocabularySentenceResponse:
    req_id = uuid.uuid4().hex[:8]
    endpoint = "/analyze-vocabulary-sentence"
    word = (target_word or "").strip()
    pos = (part_of_speech or "").strip()
    if not word:
        raise HTTPException(400, "target_word is required.")

    log(endpoint, req_id, None, event="start", target_word=word, part_of_speech=pos)

    features, transcript_data = await _decode_and_transcribe(audio, req_id, endpoint)

    out = await analyze_vocabulary_sentence(word, pos, transcript_data, features)
    log(
        endpoint, req_id, None, event="done",
        word_present=out.word_present, used_correctly=out.used_correctly,
        clarity=out.scores.clarity, pace=out.scores.pace,
    )
    return out


# ─── /analyze-vocabulary-speech ─────────────────────────────────────────────


async def handle_analyze_vocabulary_speech(
    audio: UploadFile = File(...),
    words: str = Form(...),
) -> AnalyzeVocabularySpeechResponse:
    req_id = uuid.uuid4().hex[:8]
    endpoint = "/analyze-vocabulary-speech"

    try:
        parsed = json.loads(words)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"words must be a JSON-encoded array: {e}") from e
    if not isinstance(parsed, list) or not parsed:
        raise HTTPException(400, "words must be a non-empty array.")
    word_list = [str(w).strip() for w in parsed if isinstance(w, (str, int, float)) and str(w).strip()]
    if len(word_list) < 1:
        raise HTTPException(400, "words must contain at least one non-empty entry.")

    log(endpoint, req_id, None, event="start", words=word_list)

    features, transcript_data = await _decode_and_transcribe(audio, req_id, endpoint)

    out = await analyze_vocabulary_speech(word_list, transcript_data, features)
    log(
        endpoint, req_id, None, event="done",
        detected_used=sum(1 for d in out.words_detected if d.used),
        archetype=out.archetype, overall=out.scores.overall,
    )
    return out
