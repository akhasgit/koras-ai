"""
Listening Comprehension routes.

- POST /synthesize-listening-audio  → MP3 bytes (dumb TTS; caching lives in koras-api)
- POST /analyze-listening-response  → grade a spoken answer for comprehension
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from controllers.listening import (
    handle_analyze_listening_response,
    handle_synthesize_listening_audio,
)
from models.listening import SynthesizeListeningAudioRequest
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/synthesize-listening-audio",
    dependencies=[Depends(verify_internal_secret)],
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def synthesize_listening_audio(req: SynthesizeListeningAudioRequest) -> Response:
    return await handle_synthesize_listening_audio(req)


@router.post(
    "/analyze-listening-response",
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_listening_response(request: Request) -> dict:
    return await handle_analyze_listening_response(request)
