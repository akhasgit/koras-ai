from fastapi import APIRouter, Depends, File, UploadFile

from controllers.analyze import handle_analyze
from models.analyze import AnalyzeResponse
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(verify_internal_secret)])
async def analyze(audio: UploadFile = File(...)) -> AnalyzeResponse:
    return await handle_analyze(audio)
