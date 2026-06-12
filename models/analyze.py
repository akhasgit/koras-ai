from typing import List

from pydantic import BaseModel


class FillerAnalysis(BaseModel):
    count: int
    rate_per_minute: float
    fillers_used: List[str]
    worst_sentences: List[str]


class PhrasingRewrite(BaseModel):
    original: str
    stronger: str
    why: str


class ConfidenceMarkers(BaseModel):
    hedging_words: List[str]
    count: int


class TranscriptAnalysis(BaseModel):
    topic_summary: str
    filler_analysis: FillerAnalysis
    phrasing_rewrites: List[PhrasingRewrite]
    clarity_issues: List[str]
    confidence_markers: ConfidenceMarkers
    suggested_revision: str


class VoiceScores(BaseModel):
    pitch: int
    pace: int
    clarity: int
    resonance: int
    confidence: int
    overall: int


class AcousticMetrics(BaseModel):
    duration_seconds: float
    words_per_minute: float
    mean_pitch_hz: float
    pitch_std_hz: float
    hnr_db: float
    pause_count: int
    long_pause_count: int


class AnalyzeResponse(BaseModel):
    scores: VoiceScores
    metrics: AcousticMetrics
    transcript: str
    transcript_analysis: TranscriptAnalysis
    coach_feedback: str
    archetype: str
