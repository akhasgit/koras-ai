from pydantic import BaseModel


class AITutorTurnInput(BaseModel):
    role: str
    transcript: str
    turn_index: int | None = None
    started_at: str | None = None
    ended_at: str | None = None


class AITutorAnalyzeRequest(BaseModel):
    session_id: str
    user_id: str | None = None
    mode: str = "speaking_foundations"
    duration_seconds: int | None = None
    acoustic_metrics: dict | None = None
    audio_object_key: str | None = None
    turns: list[AITutorTurnInput]


class AITutorScores(BaseModel):
    relevance: int
    eloquence: int
    fluency: int
    grammar: int
    tense: int
    fillerControl: int
    clarity: int
    confidence: int
    vocabulary: int
    listening: int


class AITutorFeedback(BaseModel):
    summary: str
    strengths: list[str]
    improvements: list[str]
    repeatedMistakes: list[str]
    bestAnswer: str | None = None
    rewrittenAnswer: str | None = None
    nextRecommendedLesson: str | None = None


class AITutorTurnFeedback(BaseModel):
    turnIndex: int
    relevanceScore: int | None = None
    grammarNotes: list[str] = []
    strongerVersion: str | None = None


class AITutorReportResponse(BaseModel):
    overall: int
    scores: AITutorScores
    metrics: dict
    feedback: AITutorFeedback
    turnFeedback: list[AITutorTurnFeedback] = []
