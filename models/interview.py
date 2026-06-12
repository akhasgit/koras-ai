from pydantic import BaseModel


class GenerateInterviewQuestionsRequest(BaseModel):
    title: str | None = None
    jobRole: str | None = None
    company: str | None = None
    interviewType: str = "general"
    experienceLevel: str | None = None
    description: str | None = None
    notes: str | None = None


class GeneratedInterviewQuestion(BaseModel):
    id: str
    question: str
    type: str
    skillTags: list[str] = []
    suggestedDurationSeconds: int = 90
    answerFramework: str | None = None
    whatGoodLooksLike: list[str] = []


class GenerateInterviewQuestionsResponse(BaseModel):
    questions: list[GeneratedInterviewQuestion]
    extractedContext: dict = {}
    warning: str | None = None


class InterviewScores(BaseModel):
    overall: int
    delivery: int
    relevance: int
    structure: int
    specificity: int
    confidence: int
    fluency: int
    grammar: int
    conciseness: int
    professionalism: int
    star: int | None = None


class InterviewFeedback(BaseModel):
    summary: str
    strengths: list[str] = []
    improvements: list[str] = []
    repeatedMistakes: list[str] = []
    bestLine: str | None = None
    weakerLine: str | None = None
    strongerVersion: str | None = None
    nextPracticeFocus: str | None = None


class InterviewAnswerReport(BaseModel):
    transcript: str
    scores: InterviewScores
    metrics: dict
    feedback: InterviewFeedback
    frameworkAnalysis: dict = {}
    question: str
    questionType: str | None = None


class InterviewAnalyzeJSONRequest(BaseModel):
    transcript: str
    question: str
    questionType: str | None = None
    scenarioTitle: str | None = None
    jobRole: str | None = None
    company: str | None = None
    description: str | None = None
    durationSeconds: int | None = None
