from fastapi import APIRouter, Depends, File, Form, UploadFile

from controllers.speech_check import handle_analyze_speech_check, handle_rescore_speech_check
from models.speech_check import (
    AnalyzeSpeechCheckResponse,
    RescoreSpeechCheckRequest,
    RescoreSpeechCheckResponse,
)
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/analyze-speech-check",
    response_model=AnalyzeSpeechCheckResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_speech_check_route(
    audio: UploadFile = File(...),
    mode: str = Form("talk"),
    passage: str = Form(""),
    prompt: str = Form(""),
):
    return await handle_analyze_speech_check(
        audio=audio, mode=mode, passage=passage, prompt=prompt,
    )


@router.post(
    "/rescore-speech-check",
    response_model=RescoreSpeechCheckResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def rescore_speech_check_route(req: RescoreSpeechCheckRequest):
    return await handle_rescore_speech_check(req)
