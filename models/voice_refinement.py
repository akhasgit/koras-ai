"""
Voice Refinement — pydantic models.

Source of truth for the /analyze-voice-refinement response shape. The TS types
in `koras-web/src/lib/voice-refinement/types.ts` mirror these — keep them in
lockstep.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from models.analyze import AnalyzeResponse


# ─────────────────────────────────────────────────────────────────────────
# Slider intent + classification
# ─────────────────────────────────────────────────────────────────────────

class TargetIntent(BaseModel):
    """Slider state at submit time. All fields optional at parse; the service
    clamps + defaults sensibly before classification."""

    pitch_semitones: float = 0.0
    speed_ratio: float = 1.0
    resonance: float = 0.0   # -1..+1, negative = warmer, positive = brighter
    brightness: float = 0.0  # -1..+1


class BandFeatures(BaseModel):
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float


class NaturalRangeHz(BaseModel):
    mean: float
    q1: float
    q3: float
    low: float
    high: float


PitchClassification = Literal["trainable", "perceptual_proxy", "out_of_scope"]


class TargetClassification(BaseModel):
    pitch: PitchClassification
    notes: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Plan
# ─────────────────────────────────────────────────────────────────────────

ActivityType = Literal["training", "drill", "recording", "reflection"]


class VoiceRefinementPlanActivity(BaseModel):
    id: str
    day: int  # 1..14
    type: ActivityType
    title: str
    description: str
    duration_minutes: str = Field(alias="durationMinutes")
    purpose: str
    instructions: List[str] = Field(default_factory=list)
    referenced_activity_id: Optional[str] = Field(default=None, alias="referencedActivityId")
    is_checkpoint: Optional[bool] = Field(default=None, alias="isCheckpoint")
    target_seconds: Optional[int] = Field(default=None, alias="targetSeconds")

    class Config:
        populate_by_name = True


class VoiceRefinementPlanDay(BaseModel):
    day: int
    title: str
    activities: List[VoiceRefinementPlanActivity]


class VoiceRefinementPlan(BaseModel):
    total_days: int = Field(default=14, alias="totalDays")
    days: List[VoiceRefinementPlanDay]
    clinical_note: Optional[str] = Field(default=None, alias="clinicalNote")

    class Config:
        populate_by_name = True


# ─────────────────────────────────────────────────────────────────────────
# Full response
# ─────────────────────────────────────────────────────────────────────────

PromptKind = Literal["baseline", "checkpoint_d7", "checkpoint_d14"]


class VoiceRefinementReport(BaseModel):
    baseline: AnalyzeResponse
    band_features: BandFeatures
    natural_range_hz: NaturalRangeHz
    target_intent: TargetIntent
    target_classification: TargetClassification
    recommended_focus: List[str] = Field(default_factory=list)
    clinical_note: Optional[str] = None
    plan: Optional[VoiceRefinementPlan] = None
    prompt_kind: PromptKind = "baseline"


# ─────────────────────────────────────────────────────────────────────────
# VF activity catalog item (passed in as multipart JSON string)
# ─────────────────────────────────────────────────────────────────────────

class VfCatalogItem(BaseModel):
    id: str
    day: int
    title: str
    type: ActivityType
    duration_minutes: str = Field(alias="durationMinutes")

    class Config:
        populate_by_name = True
