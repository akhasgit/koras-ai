"""
Pydantic models for the Listening Comprehension programme.

Mirrored 1:1 by `koras-web/src/lib/listening/types.ts`. If you change a
field here, update the TypeScript type in the same commit.
"""
from typing import Literal

from pydantic import BaseModel, Field


# ─── TTS ────────────────────────────────────────────────────────────────────


class SynthesizeListeningAudioRequest(BaseModel):
    """
    Turn text into MP3 audio. Cheap, on-demand, and cache-friendly:
    koras-api hashes (text, voice, speed) and caches the MP3 in R2 so
    this endpoint is only hit on cache miss.
    """
    text: str = Field(..., min_length=1, max_length=6000)
    voice: str | None = None  # provider-specific id, defaults picked by service
    speed: float = Field(1.0, ge=0.5, le=1.5)


# ─── Voice grading ──────────────────────────────────────────────────────────


VocabularyRange = Literal["limited", "adequate", "strong"]


class VocabularyNote(BaseModel):
    range: VocabularyRange = "adequate"
    notable_words: list[str] = []


class ListeningVoiceAnalysis(BaseModel):
    """
    Claude's grading of a student's spoken answer to a listening
    question. Focus is on comprehension — did they understand what
    they heard — not on delivery.
    """
    comprehension_score: int = Field(..., ge=0, le=100)
    relevance_score: int = Field(..., ge=0, le=100)
    captured_meaning: bool = False
    key_points_covered: list[str] = []
    key_points_missed: list[str] = []
    vocabulary: VocabularyNote = VocabularyNote()
    feedback: str = ""


class AnalyzeListeningResponseJSONRequest(BaseModel):
    """JSON-mode (transcript already produced client-side)."""
    transcript: str
    question_prompt: str
    passage_gist: str = ""
    expected_points: list[str] = []
    target_skill: str = "main_idea"
