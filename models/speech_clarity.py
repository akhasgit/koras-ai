from typing import Any, Optional

from pydantic import BaseModel, Field


class GenerateClarityPassageRequest(BaseModel):
    user_id: str
    date: str
    target_accent_id: str = "accent_5"
    grade_band: Optional[str] = None
    active_feature_ids: list[str] = Field(default_factory=list)
    generator_version: str = "v1"
    passage_seed: str


class GenerateClarityPassageResponse(BaseModel):
    passage_text: str
    seeded_feature_ids: list[str]
    fallback: bool = False


class ClarityScores(BaseModel):
    intelligibility: Optional[int] = None
    rhythm: Optional[int] = None
    word_stress: Optional[int] = None
    intonation: Optional[int] = None
    fluency: int
    accent_alignment: Optional[int] = None
    overall: int


class AnalyzeClarityReadResponse(BaseModel):
    transcript: str
    scores: ClarityScores
    acoustic_metrics: dict[str, Any]
    feature_findings: list[dict[str, Any]]
    error_map: dict[str, Any]
    feedback: dict[str, Any]
    phone_sequence: Optional[str] = None
    reference_phones: Optional[str] = None
    phone_available: bool = False
    off_script: bool = False


class ScoreClarityDrillResponse(BaseModel):
    scores: ClarityScores
    acoustic_metrics: dict[str, Any]
    drill_result: dict[str, Any]
    phone_available: bool = False


class GenerateClarityDrillsRequest(BaseModel):
    error_map: dict[str, Any]
    active_features: list[dict[str, Any]] = Field(default_factory=list)
    include_transfer: bool = False


class GenerateClarityDrillsResponse(BaseModel):
    drills: list[dict[str, Any]]
