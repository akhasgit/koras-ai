from typing import Literal

from pydantic import BaseModel


class IELTSAnalyzeRequest(BaseModel):
    user_id: str | None = None
    attempt_id: str | None = None
    lesson_id: str
    part: Literal["overview", "part_1", "part_2", "part_3", "mock"]
    prompt: str
    transcript: str | None = None
    duration_seconds: int | None = None
    audio_object_key: str | None = None
    acoustic_metrics: dict | None = None


class IELTSCriteriaBandScoresModel(BaseModel):
    fluencyCoherence: float
    lexicalResource: float
    grammarRangeAccuracy: float
    pronunciation: float


class IELTSCriteriaScores100Model(BaseModel):
    fluencyCoherence: int
    lexicalResource: int
    grammarRangeAccuracy: int
    pronunciation: int


class IELTSReportModel(BaseModel):
    practiceBandEstimate: float
    overallScore: int
    criteriaBand: IELTSCriteriaBandScoresModel
    criteriaScores: IELTSCriteriaScores100Model
    korasMetrics: dict
    feedback: dict
    transcriptFeedback: list[dict]
    nextRecommendedLessonId: str | None = None
    transcript: str | None = None
