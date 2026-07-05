from fastapi import APIRouter, Depends, File, Form, UploadFile

from controllers.vocabulary import (
    handle_analyze_pronunciation,
    handle_analyze_vocabulary_sentence,
    handle_analyze_vocabulary_speech,
    handle_generate_vocabulary_words,
)
from models.vocabulary import (
    AnalyzePronunciationResponse,
    AnalyzeVocabularySentenceResponse,
    AnalyzeVocabularySpeechResponse,
    GenerateVocabularyRequest,
    GenerateVocabularyResponse,
)
from utils.auth import verify_internal_secret

router = APIRouter()


@router.post(
    "/generate-vocabulary-words",
    response_model=GenerateVocabularyResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def generate_vocabulary_words_route(
    req: GenerateVocabularyRequest,
) -> GenerateVocabularyResponse:
    return await handle_generate_vocabulary_words(req)


@router.post(
    "/analyze-pronunciation",
    response_model=AnalyzePronunciationResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_pronunciation_route(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
) -> AnalyzePronunciationResponse:
    return await handle_analyze_pronunciation(audio=audio, target_word=target_word)


@router.post(
    "/analyze-vocabulary-sentence",
    response_model=AnalyzeVocabularySentenceResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_vocabulary_sentence_route(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    part_of_speech: str = Form(""),
) -> AnalyzeVocabularySentenceResponse:
    return await handle_analyze_vocabulary_sentence(
        audio=audio,
        target_word=target_word,
        part_of_speech=part_of_speech,
    )


@router.post(
    "/analyze-vocabulary-speech",
    response_model=AnalyzeVocabularySpeechResponse,
    dependencies=[Depends(verify_internal_secret)],
)
async def analyze_vocabulary_speech_route(
    audio: UploadFile = File(...),
    words: str = Form(...),
) -> AnalyzeVocabularySpeechResponse:
    return await handle_analyze_vocabulary_speech(audio=audio, words=words)
