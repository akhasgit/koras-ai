from fastapi import APIRouter, Depends, File, Form, UploadFile

from controllers.reading import (
    handle_analyze_reading,
    handle_generate_reading_program,
    handle_generate_reading_stage,
)
from models.reading import (
    GenerateReadingProgramRequest,
    GenerateReadingProgramResponse,
    GenerateReadingStageRequest,
    GenerateReadingStageResponse,
    ReadingAnalysisResponse,
)
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/analyze-reading",
    response_model=ReadingAnalysisResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_reading_route(
    audio: UploadFile = File(...),
    passage_text: str = Form(...),
    attempt_type: str = Form(...),
    guide_wpm: int | None = Form(None),
    level_hint: str | None = Form(None),
    target_contour: str | None = Form(None),
) -> ReadingAnalysisResponse:
    return await handle_analyze_reading(
        audio=audio,
        passage_text=passage_text,
        attempt_type=attempt_type,
        guide_wpm=guide_wpm,
        level_hint=level_hint,
        target_contour=target_contour,
    )


@router.post(
    "/generate-reading-program",
    response_model=GenerateReadingProgramResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def generate_reading_program_route(
    req: GenerateReadingProgramRequest,
) -> GenerateReadingProgramResponse:
    return await handle_generate_reading_program(req)


@router.post(
    "/generate-reading-stage",
    response_model=GenerateReadingStageResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def generate_reading_stage_route(
    req: GenerateReadingStageRequest,
) -> GenerateReadingStageResponse:
    return await handle_generate_reading_stage(req)
