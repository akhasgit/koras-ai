from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from controllers.speech_clarity import (
    handle_analyze_clarity_read,
    handle_generate_clarity_drills,
    handle_generate_clarity_passage,
    handle_score_clarity_drill,
    handle_synthesize_clarity_audio,
)
from models.listening import SynthesizeListeningAudioRequest
from models.speech_clarity import (
    AnalyzeClarityReadResponse,
    GenerateClarityDrillsRequest,
    GenerateClarityDrillsResponse,
    GenerateClarityPassageRequest,
    GenerateClarityPassageResponse,
    ScoreClarityDrillResponse,
)
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/generate-clarity-passage",
    response_model=GenerateClarityPassageResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def generate_clarity_passage_route(req: GenerateClarityPassageRequest):
    return await handle_generate_clarity_passage(req)


@router.post(
    "/analyze-clarity-read",
    response_model=AnalyzeClarityReadResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_clarity_read_route(
    audio: UploadFile = File(...),
    passage: str = Form(...),
    active_features: str = Form("[]"),
):
    return await handle_analyze_clarity_read(
        audio=audio, passage=passage, active_features=active_features,
    )


@router.post(
    "/score-clarity-drill",
    response_model=ScoreClarityDrillResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def score_clarity_drill_route(
    audio: UploadFile = File(...),
    prompt_text: str = Form(""),
):
    return await handle_score_clarity_drill(audio=audio, prompt_text=prompt_text)


@router.post(
    "/generate-clarity-drills",
    response_model=GenerateClarityDrillsResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def generate_clarity_drills_route(req: GenerateClarityDrillsRequest):
    return await handle_generate_clarity_drills(req)


@router.post(
    "/synthesize-clarity-audio",
    dependencies=[Depends(verify_internal_secret)],
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def synthesize_clarity_audio_route(req: SynthesizeListeningAudioRequest):
    audio = await handle_synthesize_clarity_audio(req.text, req.voice, req.speed)
    return Response(content=audio, media_type="audio/mpeg")
