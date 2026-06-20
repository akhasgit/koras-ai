from fastapi import APIRouter, File, UploadFile

from controllers.analyze import handle_analyze
from models.analyze import AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(audio: UploadFile = File(...)) -> AnalyzeResponse:
    return await handle_analyze(audio)
