"""
Reading programme — pydantic models.

Source of truth for the /analyze-reading and /generate-reading-* shapes.
koras-api stores the analysis payload verbatim in `reading_attempts.analysis`
and the stage content in `reading_stages.content` — keep these stable.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ─── /analyze-reading ────────────────────────────────────────────────────────


WordStatus = Literal["correct", "substituted", "skipped", "inserted-neighbour"]
HesitationKind = Literal["mid_clause_pause", "restart", "repetition", "filler"]


class WordFeedbackEntry(BaseModel):
    """One entry per passage token, in passage order."""

    word: str
    index: int
    status: WordStatus
    heard: Optional[str] = None
    pause_before_ms: Optional[int] = None
    hesitation: bool = False


class HesitationEvent(BaseModel):
    after_word_index: int
    gap_ms: int
    kind: HesitationKind


class ReadingAnalysisResponse(BaseModel):
    transcript: str
    duration_seconds: float
    words_per_minute: int
    match_pct: int
    hesitance_score: int        # 0–100, higher = smoother
    pace_score: int
    pace_delta_pct: int         # signed, vs guide_wpm → "+4% PACE"
    flow_score: int
    contour_similarity: Optional[int] = None  # echo attempts only
    overall_score: int
    word_feedback: List[WordFeedbackEntry] = Field(default_factory=list)
    hesitation_events: List[HesitationEvent] = Field(default_factory=list)
    coach_feedback: str
    chips: List[str] = Field(default_factory=list)


# ─── Generation request blocks ───────────────────────────────────────────────


class ReadingProfileBlock(BaseModel):
    """The user's reading intake (reading_profiles row)."""

    intent: Optional[str] = None
    goals: List[str] = Field(default_factory=list)
    persona: str = "other"
    grade_level: Optional[int] = None


class OnboardingBlock(BaseModel):
    background: Optional[str] = None
    biggest_challenge: Optional[str] = None
    goals: List[str] = Field(default_factory=list)


class BaselineScores(BaseModel):
    overall: Optional[int] = None
    pace: Optional[int] = None
    clarity: Optional[int] = None
    confidence: Optional[int] = None


class BaselineBlock(BaseModel):
    """Snapshot of the onboarding voice assessment; null when the user has none."""

    scores: BaselineScores = Field(default_factory=BaselineScores)
    wpm: Optional[int] = None
    pause_count: Optional[int] = None
    long_pause_count: Optional[int] = None
    transcript_analysis_summary: Optional[str] = None


class CalibrationBlock(BaseModel):
    """Derived by koras-api from the calibration attempt's stored analysis."""

    match_pct: Optional[int] = None
    hesitance_score: Optional[int] = None
    pace_score: Optional[int] = None
    flow_score: Optional[int] = None
    wpm: Optional[int] = None
    problem_words: List[str] = Field(default_factory=list)
    hesitation_kinds: Dict[str, int] = Field(default_factory=dict)


class GenerateReadingProgramRequest(BaseModel):
    profile: ReadingProfileBlock
    onboarding: Optional[OnboardingBlock] = None
    baseline: Optional[BaselineBlock] = None
    calibration: CalibrationBlock
    exclude_words: List[str] = Field(default_factory=list)


# ─── Programme skeleton + stage content ──────────────────────────────────────


StepType = Literal[
    "free_read", "guided_read", "echo", "cold_read",
    "punctuation", "speed_ladder", "vocab_context",
]


class SkeletonStage(BaseModel):
    position: int
    title: str
    theme: Optional[str] = None
    focus_areas: List[str] = Field(default_factory=list)


class ReadingStep(BaseModel):
    step_id: str
    type: StepType
    title: Optional[str] = None
    xp: int = 6
    passage: Optional[str] = None       # read steps
    sentence: Optional[str] = None      # echo steps
    guide_wpm: Optional[int] = None
    target_notes: Optional[str] = None  # echo steps


class ReadingLesson(BaseModel):
    lesson_id: str
    title: str
    focus: Optional[str] = None
    estimated_min: int = 6
    xp: int = 0
    steps: List[ReadingStep]

    @model_validator(mode="after")
    def _lesson_xp_is_step_sum(self) -> "ReadingLesson":
        # XP lives on steps; the lesson badge is always their exact sum.
        self.xp = sum(step.xp for step in self.steps)
        return self


class TargetVocabWord(BaseModel):
    word: str
    definition: str
    in_lesson: Optional[str] = None


class StageContent(BaseModel):
    lessons: List[ReadingLesson]
    target_vocab: List[TargetVocabWord] = Field(default_factory=list)


class GenerateReadingProgramResponse(BaseModel):
    skeleton: List[SkeletonStage]
    stage_1_content: StageContent


# ─── Stage N+1 generation ────────────────────────────────────────────────────


class StageAverages(BaseModel):
    match_pct: Optional[float] = None
    hesitance: Optional[float] = None
    pace: Optional[float] = None
    flow: Optional[float] = None


class CompletedStagePerformance(BaseModel):
    avg: StageAverages = Field(default_factory=StageAverages)
    trend: Optional[str] = None
    persistent_problem_words: List[str] = Field(default_factory=list)
    dominant_hesitation_kind: Optional[str] = None
    pace_trend_pct: List[int] = Field(default_factory=list)


class GenerateReadingStageRequest(GenerateReadingProgramRequest):
    skeleton_entry: SkeletonStage
    completed_stage_performance: Optional[CompletedStagePerformance] = None


class GenerateReadingStageResponse(BaseModel):
    stage_content: StageContent
