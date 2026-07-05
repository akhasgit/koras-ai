"""
Pydantic models for the Daily Vocabulary feature.

These shapes are mirrored in `koras-web/src/lib/vocabulary/types.ts` —
treat the pydantic models here as the source of truth; the TS types
follow.

Covers:
  * Word generation  (Phase 2)  — GenerateVocabularyRequest / Response, WordObject
  * Pronounce/sentence/speech analysis (Phase 3) — the three Analyze* responses
    plus the shared VoiceSubScores helper and WordDetected sub-shape.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from models.analyze import VoiceScores


# ─── Word generation ────────────────────────────────────────────────────────


class GenerateVocabularyRequest(BaseModel):
    """
    Input to /generate-vocabulary-words.

    Assembled by `koras-api/routes/vocabulary.py::_assemble_signals` from
    profile / onboarding_responses / learner_insights / vocabulary_learned_words.
    """

    user_id: str
    segment: str = Field(description="'individual' | 'students' | 'professionals'")
    grade_level: Optional[int] = Field(default=None, description="Set for students; null otherwise.")
    cefr_target: str = Field(description="'A1' | 'A2' | 'B1' | 'B2' | 'C1' | 'C2'")
    profession: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)
    exclude_words: List[str] = Field(default_factory=list, description="Words to avoid (last ~60 learned).")
    count: int = 5


class WordObject(BaseModel):
    """
    A single generated word. Stored as one entry in
    `vocabulary_daily_sets.words` and (once mastered) on
    `vocabulary_learned_words.word_data`.
    """

    word: str
    ipa: str
    part_of_speech: str
    definition: str
    example_sentence: str
    register: str = Field(description="formal | neutral | informal | academic | business")
    difficulty: str = Field(description="CEFR band: A1..C2")
    why_chosen: str


class GenerateVocabularyResponse(BaseModel):
    words: List[WordObject]


# ─── Shared sub-shapes ──────────────────────────────────────────────────────


class VoiceSubScores(BaseModel):
    """Lean speaking-quality trio used by sentence analysis.

    Distinct from the full `VoiceScores` we return for the end-of-day
    speech — sentence-level takes are too short to reliably score pitch
    or resonance, so we only surface the axes that actually move.
    """

    clarity: int
    pace: int
    confidence: int


class WordDetected(BaseModel):
    """One entry in `AnalyzeVocabularySpeechResponse.words_detected`."""

    word: str
    used: bool
    correct: bool
    note: str


# ─── Pronounce analysis ─────────────────────────────────────────────────────


class AnalyzePronunciationResponse(BaseModel):
    """
    Response from `POST /analyze-pronunciation`.

    Cheap: no Claude call. Whisper transcript + phonetic/fuzzy match
    against the target word + HNR-derived clarity.
    """

    target_word: str
    transcript: str
    matched: bool
    similarity: float       # 0..1
    pronunciation_score: float  # 0..100
    clarity: float          # 0..100
    feedback: str


# ─── Sentence analysis ──────────────────────────────────────────────────────


class AnalyzeVocabularySentenceResponse(BaseModel):
    """
    Response from `POST /analyze-vocabulary-sentence`.

    Whisper transcript + Claude usage check + acoustic sub-scores.
    Non-fatal fallback: `used_correctly=false` with a friendly nudge,
    scores from acoustic features alone.
    """

    target_word: str
    transcript: str
    word_present: bool
    used_correctly: bool
    usage_feedback: str
    grammar_notes: str
    scores: VoiceSubScores
    coach_feedback: str


# ─── End-of-day speech analysis ─────────────────────────────────────────────


class AnalyzeVocabularySpeechResponse(BaseModel):
    """
    Response from `POST /analyze-vocabulary-speech`.

    Mirrors the shape of `AnalyzeResponse` (from `/analyze`) so the SPA
    can reuse the same score-render components. Adds `words_detected`
    for the "you used 4 of 5 words correctly" summary.
    """

    transcript: str
    words_detected: List[WordDetected]
    scores: VoiceScores
    coach_feedback: str
    archetype: str
