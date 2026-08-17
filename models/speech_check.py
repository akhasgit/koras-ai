from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SpeechCheckScores(BaseModel):
    intelligibility: int
    fluency: int
    pace: int
    clarity_acoustic: int
    overall: int


class FlaggedWord(BaseModel):
    id: str
    word: str
    ipa: str = ""
    start_s: float
    end_s: float
    start_ms: int
    end_ms: int
    segment_logprob: Optional[float] = None
    intelligibility: int
    issue: Literal["unclear", "uncertain", "rushed"]


class WordToken(BaseModel):
    id: Optional[str] = None
    word: str
    start_s: float = 0.0
    end_s: float = 0.0
    segment_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    deleted: bool = False


class AnalyzeSpeechCheckResponse(BaseModel):
    transcript: str
    words: list[dict[str, Any]]
    scores: SpeechCheckScores
    flagged_words: list[FlaggedWord]
    also_noticed: list[FlaggedWord] = Field(default_factory=list)
    acoustic_metrics: dict[str, Any]
    coach_note: str = ""
    off_script: bool = False
    too_short: bool = False


class RescoreSpeechCheckRequest(BaseModel):
    words: list[WordToken]
    acoustic_metrics: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["talk", "read"] = "talk"
    passage_text: Optional[str] = None
    existing_flagged: list[FlaggedWord] = Field(default_factory=list)


class RescoreSpeechCheckResponse(BaseModel):
    transcript: str
    scores: SpeechCheckScores
    flagged_words: list[FlaggedWord]
    also_noticed: list[FlaggedWord] = Field(default_factory=list)
    off_script: bool = False
