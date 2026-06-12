"""
Koras Backend API — pure FastAPI, Docker/GCP target

Run locally:
    uvicorn local_app:app --reload --port 8000

Run in Docker (via gunicorn):
    gunicorn local_app:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000

Environment variables required:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    KORAS_INTERNAL_SECRET   — shared secret, must match Vercel env var
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import io
import json
import os
import re
import tempfile
import time
import uuid
from typing import List, Literal

import ffmpeg
import librosa
import numpy as np
import parselmouth
from anthropic import AsyncAnthropic
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from parselmouth.praat import call
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
#  Singletons — one connection pool per process                               #
# --------------------------------------------------------------------------- #

_openai = AsyncOpenAI()
_anthropic = AsyncAnthropic()

# Semaphore: cap concurrent CPU-heavy extractions to (cores × 1) so the event
# loop isn't starved when many requests land simultaneously.
_extract_sem = asyncio.Semaphore(os.cpu_count() or 2)

# --------------------------------------------------------------------------- #
#  Auth — shared secret (Next.js server → Docker, never browser)             #
# --------------------------------------------------------------------------- #

KORAS_INTERNAL_SECRET = os.environ.get("KORAS_INTERNAL_SECRET", "")


def verify_internal_secret(x_koras_secret: str = Header(...)):
    if not KORAS_INTERNAL_SECRET or x_koras_secret != KORAS_INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --------------------------------------------------------------------------- #
#  Tunables                                                                   #
# --------------------------------------------------------------------------- #

CLAUDE_MODEL = "claude-sonnet-4-5"
WHISPER_MODEL = "whisper-1"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_TRANSCRIPT_WORDS = 10

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://koras.com",
    "https://www.koras.com",
    "https://koras.vercel.app",
]

ALLOWED_ORIGIN_REGEX: str | None = None

# --------------------------------------------------------------------------- #
#  Logging helpers                                                             #
# --------------------------------------------------------------------------- #

def _log(endpoint: str, req_id: str, user_id: str | None, **extra) -> None:
    print(json.dumps({
        "ts": time.time(),
        "req_id": req_id,
        "endpoint": endpoint,
        "user_id": user_id,
        **extra,
    }))

# --------------------------------------------------------------------------- #
#  Prompts                                                                    #
# --------------------------------------------------------------------------- #

TRANSCRIPT_ANALYSIS_PROMPT = """You are analyzing a short voice sample where someone introduces themselves. You receive only the transcript.

Return STRICT JSON with this exact schema. Do not include preamble, markdown fences, or commentary — only the JSON object.

{{
  "topic_summary": "1-2 sentences describing what they were trying to communicate",
  "filler_analysis": {{
    "count": <integer, total filler instances>,
    "rate_per_minute": 0,
    "fillers_used": ["um", "like", ...],
    "worst_sentences": ["full sentence with many fillers", ...]
  }},
  "phrasing_rewrites": [
    {{"original": "weak sentence verbatim from transcript", "stronger": "improved version", "why": "one-line reason"}},
    ... exactly 3 entries
  ],
  "clarity_issues": ["short description of issue", ... up to 5],
  "confidence_markers": {{
    "hedging_words": ["maybe", "kind of", ... in order of appearance],
    "count": <integer>
  }},
  "suggested_revision": "a cleaner version of what they said — same length and intent, but tighter, more confident, no fillers, fewer hedges"
}}

FILLERS to count (case-insensitive):
um, uh, er, ah, like (when used as filler not comparison), you know, sort of, kind of, basically, literally, actually (when hedging), I mean, right? (rhetorical), so (when starting a sentence)

HEDGING WORDS to flag:
maybe, perhaps, probably, kinda, sorta, I think, I guess, somewhat, fairly, pretty (as in "pretty good"), just (as a softener: "I just wanted to..."), only, a little, a bit

PHRASING REWRITES — pick 3 sentences that are weakest in: vague language, hedging, run-on, passive when active would land harder, or buried lede.

CLARITY ISSUES — flag specific things like: "Run-on sentence in section about X", "Unclear pronoun reference", "Buried key point".

If transcript is clean, return shorter lists or empty arrays — but ALWAYS return exactly 3 phrasing_rewrites by picking the comparatively weakest sentences.

Set rate_per_minute to 0 — the backend computes the real value.

Transcript to analyze:
\"\"\"
{transcript}
\"\"\"

Return only the JSON object."""


COACH_FEEDBACK_PROMPT = """You are a warm, expert voice coach giving immediate feedback on a short voice sample someone just recorded. They're nervous. They want to know what's working and what to focus on.

Their scores (0–100):
- Pitch variation: {pitch}
- Pace: {pace}
- Clarity: {clarity}
- Resonance: {resonance}
- Confidence: {confidence}
- Overall: {overall}

Transcript signals:
- Filler words: {filler_count}
- Hedging language: {hedging_count}
- Topic communicated: {topic_summary}

Write 2 to 3 sentences that:
1. Open by acknowledging their strongest dimension — describe what it sounds like, don't just say "your X is good"
2. Name ONE concrete thing to work on with one specific action they can try
3. Sound like a real coach, not a report

Hard rules:
- DO NOT mention numerical scores
- DO NOT say "your X score" or "you scored"
- DO NOT use jargon like "F0 variation" or "spectral centroid"
- DO use natural language: "your pacing", "the warmth in your voice", "those little 'um' moments"
- Tone: warm, specific, encouraging but honest
- Length: 2 to 3 sentences total, max 70 words

Return ONLY the feedback text. No headers, no JSON, no quotation marks."""

# --------------------------------------------------------------------------- #
#  Pydantic models                                                            #
# --------------------------------------------------------------------------- #

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

class DailyPlanItemModel(BaseModel):
    item_id: str | None = None
    type: str
    program_id: str
    route: str
    title: str
    reason: str
    estimated_minutes: int
    priority: int
    status: str = "pending"
    completed_at: str | None = None

class GenerateDailyPlanRequest(BaseModel):
    user_id: str
    signals: dict
    rules_plan: dict

class GenerateDailyPlanResponse(BaseModel):
    summary: str
    focus_area: str
    advice: str
    items: List[DailyPlanItemModel]

# --------------------------------------------------------------------------- #
#  FastAPI app                                                                 #
# --------------------------------------------------------------------------- #

app = FastAPI(title="Koras Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
#  Audio helpers                                                               #
# --------------------------------------------------------------------------- #

def normalize_audio(audio_bytes: bytes, filename: str) -> bytes:
    suffix = os.path.splitext(filename)[1] or ".bin"
    in_path: str | None = None
    out_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        out_path = in_path + ".wav"
        (
            ffmpeg
            .input(in_path)
            .output(out_path, ac=1, ar=16000, format="wav", loglevel="error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        with open(out_path, "rb") as f:
            return f.read()
    except ffmpeg.Error as e:  # type: ignore[attr-defined]
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed: {stderr[:400]}") from e
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


def extract_features(wav_bytes: bytes) -> dict:
    wav_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        snd = parselmouth.Sound(wav_path)
        duration = snd.duration

        pitch_obj = snd.to_pitch()
        pitch_values = pitch_obj.selected_array["frequency"]
        voiced = pitch_values[pitch_values > 0]
        mean_f0 = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
        std_f0 = float(np.std(voiced)) if len(voiced) > 0 else 0.0

        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        hnr = float(hnr) if hnr == hnr else 0.0  # nan guard

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        intervals = librosa.effects.split(y, top_db=30)
        pause_count = max(0, len(intervals) - 1)
        long_pause_count = 0
        if len(intervals) > 1:
            gap_starts = intervals[:-1, 1]
            gap_ends = intervals[1:, 0]
            gaps = (gap_ends - gap_starts) / sr
            long_pause_count = int(np.sum(gaps > 0.5))

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        return {
            "duration": duration,
            "mean_f0": mean_f0,
            "std_f0": std_f0,
            "hnr": hnr,
            "pause_count": int(pause_count),
            "long_pause_count": int(long_pause_count),
            "spectral_centroid": centroid,
        }
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


async def transcribe(wav_bytes: bytes) -> dict:
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"

    result = await _openai.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )

    words_raw = getattr(result, "words", None) or []
    words = []
    for w in words_raw:
        if hasattr(w, "word"):
            words.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
        else:
            words.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})

    return {
        "text": (result.text or "").strip(),
        "words": words,
        "duration": float(getattr(result, "duration", 0.0) or 0.0),
    }

# --------------------------------------------------------------------------- #
#  Shared helpers                                                              #
# --------------------------------------------------------------------------- #

_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END = re.compile(r"\s*```\s*$")

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = _FENCE_START.sub("", raw)
    raw = _FENCE_END.sub("", raw)
    if not raw.startswith("{"):
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            raw = raw[first : last + 1]
    return raw.strip()

def clamp(v: int | float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))

FILLER_PATTERNS = [
    r"\bum\b", r"\buh\b", r"\berm\b", r"\blike\b",
    r"\byou know\b", r"\bbasically\b", r"\bactually\b",
    r"\bi mean\b", r"\bsort of\b", r"\bkind of\b", r"\bkinda\b",
]
_filler_regex = re.compile("|".join(FILLER_PATTERNS), re.IGNORECASE)

def count_fillers(text: str) -> int:
    return len(_filler_regex.findall(text))

# --------------------------------------------------------------------------- #
#  /health                                                                    #
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# --------------------------------------------------------------------------- #
#  /analyze  (voice coaching)                                                 #
# --------------------------------------------------------------------------- #

async def analyze_transcript(transcript: str) -> TranscriptAnalysis:
    msg = await _anthropic.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": TRANSCRIPT_ANALYSIS_PROMPT.format(transcript=transcript)}],
    )
    text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    raw = "\n".join(text_blocks).strip()
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude returned invalid JSON: {cleaned[:300]}") from e

    fa = data.get("filler_analysis", {})
    cm = data.get("confidence_markers", {})
    return TranscriptAnalysis(
        topic_summary=data.get("topic_summary", ""),
        filler_analysis=FillerAnalysis(
            count=int(fa.get("count", 0)),
            rate_per_minute=float(fa.get("rate_per_minute", 0)),
            fillers_used=fa.get("fillers_used", []),
            worst_sentences=fa.get("worst_sentences", []),
        ),
        phrasing_rewrites=[
            PhrasingRewrite(**r) for r in data.get("phrasing_rewrites", [])
        ],
        clarity_issues=data.get("clarity_issues", []),
        confidence_markers=ConfidenceMarkers(
            hedging_words=cm.get("hedging_words", []),
            count=int(cm.get("count", 0)),
        ),
        suggested_revision=data.get("suggested_revision", ""),
    )


async def generate_coach_feedback(scores: VoiceScores, transcript_analysis: TranscriptAnalysis) -> str:
    msg = await _anthropic.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": COACH_FEEDBACK_PROMPT.format(
                pitch=scores.pitch,
                pace=scores.pace,
                clarity=scores.clarity,
                resonance=scores.resonance,
                confidence=scores.confidence,
                overall=scores.overall,
                filler_count=transcript_analysis.filler_analysis.count,
                hedging_count=transcript_analysis.confidence_markers.count,
                topic_summary=transcript_analysis.topic_summary,
            ),
        }],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts).strip().strip('"').strip()


def pick_archetype(s: VoiceScores) -> str:
    dims = {"pitch": s.pitch, "pace": s.pace, "clarity": s.clarity,
            "resonance": s.resonance, "confidence": s.confidence}
    top = max(dims, key=dims.get)
    if top == "resonance" and s.resonance >= 80: return "The Warm Communicator"
    if top == "clarity" and s.clarity >= 80: return "The Precise Speaker"
    if top == "pitch" and s.pitch >= 80: return "The Natural Storyteller"
    if top == "confidence" and s.confidence >= 80: return "The Grounded Voice"
    if s.clarity >= 75 and s.pace >= 75: return "The Clear Thinker"
    if s.pitch >= 70 and s.pace >= 70: return "The Energetic Presenter"
    return "The Developing Voice"


def compute_voice_scores(features: dict, transcript_data: dict, transcript_analysis: TranscriptAnalysis) -> tuple[VoiceScores, AcousticMetrics]:
    duration = features["duration"]
    word_count = len(transcript_data["text"].split())
    wpm = (word_count / duration * 60) if duration > 0 else 0

    std_f0 = features["std_f0"]
    pitch_score = clamp(int(min(100, (std_f0 / 50) * 100)))

    pace_score = clamp(int(100 - abs(wpm - 130) / 1.3)) if wpm > 0 else 50

    filler_rate = transcript_analysis.filler_analysis.count / max(duration / 60, 0.1)
    clarity_score = clamp(int(100 - filler_rate * 15))

    hnr = features["hnr"]
    resonance_score = clamp(int(min(100, (hnr / 20) * 100))) if hnr > 0 else 50

    hedge_count = transcript_analysis.confidence_markers.count
    confidence_score = clamp(int(100 - hedge_count * 8))

    overall = clamp(int(
        pitch_score * 0.20 +
        pace_score * 0.20 +
        clarity_score * 0.25 +
        resonance_score * 0.15 +
        confidence_score * 0.20
    ))

    scores = VoiceScores(
        pitch=pitch_score,
        pace=pace_score,
        clarity=clarity_score,
        resonance=resonance_score,
        confidence=confidence_score,
        overall=overall,
    )
    metrics = AcousticMetrics(
        duration_seconds=round(duration, 2),
        words_per_minute=round(wpm, 1),
        mean_pitch_hz=round(features["mean_f0"], 1),
        pitch_std_hz=round(std_f0, 1),
        hnr_db=round(hnr, 1),
        pause_count=features["pause_count"],
        long_pause_count=features["long_pause_count"],
    )
    return scores, metrics


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(verify_internal_secret)])
async def analyze(audio: UploadFile = File(...)) -> AnalyzeResponse:
    req_id = uuid.uuid4().hex[:8]
    user_id = None  # /analyze has no user context — voice coaching demo endpoint
    _log("/analyze", req_id, user_id, event="start")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    async with _extract_sem:
        try:
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, audio.filename or "audio")
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        features_task = asyncio.to_thread(extract_features, wav_bytes)
        transcript_task = transcribe(wav_bytes)
        try:
            features, transcript_data = await asyncio.gather(features_task, transcript_task)
        except Exception as e:
            raise HTTPException(500, f"Analysis failed: {e}") from e

    word_count = len(transcript_data["text"].split())
    if word_count < MIN_TRANSCRIPT_WORDS:
        raise HTTPException(400, "Audio too short or unclear. Please record at least 10 words.")

    try:
        transcript_analysis = await analyze_transcript(transcript_data["text"])
    except Exception as e:
        raise HTTPException(500, f"Transcript analysis failed: {e}") from e

    scores, metrics = compute_voice_scores(features, transcript_data, transcript_analysis)

    # Update filler rate now that we have duration
    if metrics.duration_seconds > 0:
        rate = transcript_analysis.filler_analysis.count / (metrics.duration_seconds / 60)
        transcript_analysis.filler_analysis.rate_per_minute = round(rate, 2)

    try:
        coach_feedback = await generate_coach_feedback(scores, transcript_analysis)
    except Exception as e:
        _log("/analyze", req_id, user_id, event="coach_feedback_fallback", error=str(e))
        coach_feedback = "Nice work on this take. Focus on one thing next: slow down slightly and let your pauses breathe."

    archetype = pick_archetype(scores)
    _log("/analyze", req_id, user_id, event="done",
         duration=features["duration"], wpm=metrics.words_per_minute,
         overall=scores.overall, archetype=archetype)

    return AnalyzeResponse(
        scores=scores,
        metrics=metrics,
        transcript=transcript_data["text"],
        transcript_analysis=transcript_analysis,
        coach_feedback=coach_feedback,
        archetype=archetype,
    )

# --------------------------------------------------------------------------- #
#  /analyze-ai-tutor                                                          #
# --------------------------------------------------------------------------- #

SCORE_WEIGHTS = {
    "relevance": 0.15, "eloquence": 0.15, "fluency": 0.15,
    "grammar": 0.10, "tense": 0.10, "fillerControl": 0.10,
    "clarity": 0.10, "confidence": 0.05, "vocabulary": 0.05, "listening": 0.05,
}

def weighted_overall(scores: dict) -> int:
    return clamp(sum(scores.get(k, 50) * w for k, w in SCORE_WEIGHTS.items()))

def build_fallback_report(req: AITutorAnalyzeRequest, user_text: str, filler_count: int, wpm: float | None) -> AITutorReportResponse:
    word_count = len(user_text.split())
    user_turn_count = sum(1 for t in req.turns if t.role == "user")
    base = 50
    length_bonus = min(15, word_count // 10)
    turn_bonus = min(10, user_turn_count * 2)
    filler_penalty = min(20, filler_count * 3)
    raw = base + length_bonus + turn_bonus - filler_penalty
    filler_score = clamp(100 - filler_count * 8)
    scores_dict = {
        "relevance": clamp(raw + 5), "eloquence": clamp(raw - 5), "fluency": clamp(raw),
        "grammar": clamp(raw), "tense": clamp(raw), "fillerControl": filler_score,
        "clarity": clamp(raw), "confidence": clamp(raw - 3), "vocabulary": clamp(raw - 2),
        "listening": clamp(raw + 3),
    }
    return AITutorReportResponse(
        overall=weighted_overall(scores_dict),
        scores=AITutorScores(**scores_dict),
        metrics={
            "durationSeconds": req.duration_seconds,
            "wordsPerMinute": round(wpm, 1) if wpm else None,
            "fillerCount": filler_count,
            "fillerRatePerMinute": (
                round(filler_count / (req.duration_seconds / 60), 1)
                if req.duration_seconds and req.duration_seconds > 0 else None
            ),
            "longPauseCount": None, "grammarIssueCount": None, "tenseIssueCount": None,
        },
        feedback=AITutorFeedback(
            summary="Analysis completed with limited data. Practice more for a detailed breakdown.",
            strengths=["Completed a conversation session"],
            improvements=["Try giving longer, more detailed answers"],
            repeatedMistakes=[],
        ),
        turnFeedback=[],
    )


AI_TUTOR_GRADING_PROMPT = """You are Koras, an AI speaking coach for students.

Analyze the student's side of this spoken conversation.

You will receive:
1. The AI tutor's questions/responses
2. The student's answers
3. Optional duration/acoustic metadata

Grade the student on:
- relevance
- eloquence
- fluency
- grammar
- tense control
- filler control
- clarity
- confidence
- vocabulary range
- listening and follow-up quality

Rules:
- Do not overpenalize accent.
- Focus on intelligibility, structure, and communication effectiveness.
- Be specific, fair, and student-friendly.
- Give actionable feedback.
- Return strict JSON only.
- Do not include markdown.

Output schema:
{{
  "overall": number,
  "scores": {{
    "relevance": number,
    "eloquence": number,
    "fluency": number,
    "grammar": number,
    "tense": number,
    "fillerControl": number,
    "clarity": number,
    "confidence": number,
    "vocabulary": number,
    "listening": number
  }},
  "metrics": {{
    "durationSeconds": number|null,
    "wordsPerMinute": number|null,
    "fillerCount": number,
    "fillerRatePerMinute": number|null,
    "longPauseCount": number|null,
    "grammarIssueCount": number|null,
    "tenseIssueCount": number|null
  }},
  "feedback": {{
    "summary": string,
    "strengths": [string],
    "improvements": [string],
    "repeatedMistakes": [string],
    "bestAnswer": string,
    "rewrittenAnswer": string,
    "nextRecommendedLesson": string
  }},
  "turnFeedback": [
    {{
      "turnIndex": number,
      "relevanceScore": number,
      "grammarNotes": [string],
      "strongerVersion": string
    }}
  ]
}}

All scores must be integers 0-100.

Conversation transcript:
\"\"\"
{transcript}
\"\"\"

{metadata_section}

Return only the JSON object."""


@app.post("/analyze-ai-tutor", response_model=AITutorReportResponse, dependencies=[Depends(verify_internal_secret)])
async def analyze_ai_tutor(req: AITutorAnalyzeRequest) -> AITutorReportResponse:
    req_id = uuid.uuid4().hex[:8]
    _log("/analyze-ai-tutor", req_id, req.user_id, session_id=req.session_id, turns=len(req.turns))

    if not req.turns:
        raise HTTPException(400, "No conversation turns provided.")

    user_turns = [t for t in req.turns if t.role == "user"]
    if not user_turns:
        raise HTTPException(400, "No user turns found in conversation.")

    user_text = " ".join(t.transcript for t in user_turns)
    if len(user_text.strip()) < 20:
        raise HTTPException(400, "User responses too short for meaningful analysis.")

    filler_count = count_fillers(user_text)
    word_count = len(user_text.split())
    wpm: float | None = None
    if req.duration_seconds and req.duration_seconds > 0:
        wpm = (word_count / req.duration_seconds) * 60

    lines: list[str] = []
    for t in sorted(req.turns, key=lambda x: x.turn_index or 0):
        speaker = "Tutor" if t.role == "assistant" else "Student"
        lines.append(f"[{speaker}]: {t.transcript}")
    transcript_text = "\n".join(lines)

    metadata_parts: list[str] = []
    if req.duration_seconds:
        metadata_parts.append(f"Duration: {req.duration_seconds}s")
    if wpm:
        metadata_parts.append(f"Approximate WPM: {round(wpm, 1)}")
    metadata_parts.append(f"Filler words detected: {filler_count}")
    metadata_parts.append(f"Total user words: {word_count}")
    metadata_section = "Metadata:\n" + "\n".join(metadata_parts) if metadata_parts else ""

    prompt = AI_TUTOR_GRADING_PROMPT.format(
        transcript=transcript_text,
        metadata_section=metadata_section,
    )

    try:
        msg = await _anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_blocks).strip()
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)

        scores_raw = data.get("scores", {})
        for k in scores_raw:
            scores_raw[k] = clamp(scores_raw[k])
        data["scores"] = scores_raw
        data["overall"] = weighted_overall(scores_raw)

        llm_metrics = data.get("metrics", {})
        llm_metrics["durationSeconds"] = req.duration_seconds
        llm_metrics["wordsPerMinute"] = round(wpm, 1) if wpm else None
        llm_metrics["fillerCount"] = max(filler_count, llm_metrics.get("fillerCount", 0))
        if req.duration_seconds and req.duration_seconds > 0:
            llm_metrics["fillerRatePerMinute"] = round(
                llm_metrics["fillerCount"] / (req.duration_seconds / 60), 1
            )
        data["metrics"] = llm_metrics

        report = AITutorReportResponse(**data)
    except Exception as e:
        _log("/analyze-ai-tutor", req_id, req.user_id, event="llm_fallback", error=str(e))
        report = build_fallback_report(req, user_text, filler_count, wpm)

    _log("/analyze-ai-tutor", req_id, req.user_id, event="done",
         overall=report.overall, user_turns=len(user_turns))
    return report

# --------------------------------------------------------------------------- #
#  /analyze-ielts-speaking                                                    #
# --------------------------------------------------------------------------- #

def score_to_band(score_100: int) -> float:
    if score_100 >= 92: return 9.0
    if score_100 >= 87: return 8.5
    if score_100 >= 82: return 8.0
    if score_100 >= 77: return 7.5
    if score_100 >= 72: return 7.0
    if score_100 >= 65: return 6.5
    if score_100 >= 58: return 6.0
    if score_100 >= 52: return 5.5
    if score_100 >= 45: return 5.0
    if score_100 >= 38: return 4.5
    return 4.0

def round_to_nearest_half_band(value: float) -> float:
    return round(value * 2) / 2

def compute_practice_band(criteria: dict) -> tuple[float, dict]:
    per = {
        "fluencyCoherence": score_to_band(int(criteria.get("fluencyCoherence", 50))),
        "lexicalResource": score_to_band(int(criteria.get("lexicalResource", 50))),
        "grammarRangeAccuracy": score_to_band(int(criteria.get("grammarRangeAccuracy", 50))),
        "pronunciation": score_to_band(int(criteria.get("pronunciation", 50))),
    }
    avg = sum(per.values()) / 4.0
    return round_to_nearest_half_band(avg), per


IELTS_NORMALIZE_PROMPT = """You will receive a speech transcript that may contain a mix of English and other
languages (commonly Tamil, Malay, Hindi, Mandarin, Hokkien, written phonetically
in English script or in native script).

Your task:
1. Identify any non-English words or phrases
2. Translate them into natural English equivalents in context — preserve the
   speaker's meaning and conversational tone
3. Return a single clean English transcript
4. Identify the languages detected

IMPORTANT: Singapore English particles (lah, lor, leh, meh, hor, sia, can, cannot)
are grammatical particles, NOT filler words. Preserve their grammatical effect
when translating but do not flag them as errors.

Do NOT correct grammar, remove filler words, or rewrite the answer to be better.
Preserve natural speech patterns. The goal is normalization, not improvement.

Return STRICT JSON only:
{{
  "clean_transcript": "...",
  "detected_languages": ["English", "Tamil"],
  "code_switching_detected": true,
  "non_english_fragments": ["naan enna pannanum-nu theriyala"]
}}

Transcript:
\"\"\"
{transcript}
\"\"\""""


async def normalize_ielts_transcript(raw_transcript: str) -> dict:
    try:
        msg = await _anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": IELTS_NORMALIZE_PROMPT.format(transcript=raw_transcript)}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_blocks).strip()
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
        return {
            "clean_transcript": str(data.get("clean_transcript", raw_transcript)),
            "detected_languages": data.get("detected_languages", ["English"]),
            "code_switching_detected": bool(data.get("code_switching_detected", False)),
            "non_english_fragments": data.get("non_english_fragments", []),
        }
    except Exception as e:
        print(json.dumps({"event": "ielts_normalization_fallback", "error": str(e)}))
        return {
            "clean_transcript": raw_transcript,
            "detected_languages": ["English"],
            "code_switching_detected": False,
            "non_english_fragments": [],
        }


IELTS_GRADING_PROMPT = """You are Koras IELTS Speaking Coach.

Analyze this IELTS Speaking practice answer.

IMPORTANT: This is NOT an official IELTS score. Return a practice estimate only.

Assess using IELTS-style criteria:
1. Fluency and Coherence — flow, hesitation, organization, linking
2. Lexical Resource — vocabulary range, accuracy, naturalness
3. Grammatical Range and Accuracy — sentence variety, tense control, errors
4. Pronunciation — intelligibility, stress, rhythm, NOT accent

You will receive:
- IELTS part: {part}
- Prompt: {prompt}
- Raw transcript: {raw_transcript}
- Normalized transcript: {normalized_transcript}
- Code-switching detected: {code_switching_detected}
- Duration seconds: {duration_seconds}
- Acoustic metrics: {acoustic_metrics}

If code-switching is true, the speaker mixed English with another language naturally.
Do NOT heavily penalize this. Focus grading on the English content but gently advise
keeping IELTS answers fully in English.

Part-specific expectations:
- part_1: short familiar-topic answer. 2-4 sentences is enough.
- part_2: long turn. ~90-120 seconds, organized around cue card bullets.
- part_3: abstract discussion. Developed opinions with reasoning.

Be strict but encouraging.
Do not overpenalize accent — focus on intelligibility.

Return STRICT JSON only. All scores are integers 0-100.

{{
  "criteriaScores": {{
    "fluencyCoherence": <int>,
    "lexicalResource": <int>,
    "grammarRangeAccuracy": <int>,
    "pronunciation": <int>
  }},
  "korasMetrics": {{
    "wordsPerMinute": <number|null>,
    "fillerCount": <int>,
    "fillerRatePerMinute": <number|null>,
    "longPauseCount": <int|null>,
    "answerRelevance": <int>,
    "structureScore": <int>,
    "specificExampleScore": <int>,
    "vocabularyRangeScore": <int>,
    "grammarIssueCount": <int>,
    "tenseIssueCount": <int>,
    "clarityScore": <int>,
    "pronunciationIntelligibility": <int>
  }},
  "feedback": {{
    "summary": "<2-3 sentence summary>",
    "strengths": ["..."],
    "improvements": ["..."],
    "ieltsAdvice": ["..."],
    "bestSentence": "<verbatim sentence>",
    "weakerSentence": "<verbatim sentence>",
    "strongerVersion": "<rewrite>",
    "nextPracticeFocus": "<one-line focus>"
  }},
  "transcriptFeedback": [
    {{"text": "<verbatim phrase>", "issue": "<one-line>", "suggestion": "<one-line>"}}
  ],
  "nextRecommendedLessonId": "<lesson_id or null>"
}}

Return only the JSON object."""


def ielts_fallback_report(req_part: str, transcript: str, duration: int | None, filler_count: int, normalization: dict) -> dict:
    words = transcript.split()
    word_count = len(words)
    wpm = (word_count / duration) * 60 if duration and duration > 0 else None
    base = 55
    length_bonus = min(15, word_count // 12)
    filler_penalty = min(20, filler_count * 3)
    raw_score = max(40, min(80, base + length_bonus - filler_penalty))
    criteria = {
        "fluencyCoherence": raw_score,
        "lexicalResource": max(40, raw_score - 3),
        "grammarRangeAccuracy": max(40, raw_score - 2),
        "pronunciation": max(45, raw_score),
    }
    overall_band, per_band = compute_practice_band(criteria)
    overall_100 = int(sum(criteria.values()) / 4)
    return {
        "criteriaScores": criteria,
        "criteriaBand": per_band,
        "practiceBandEstimate": overall_band,
        "overallScore": overall_100,
        "korasMetrics": {
            "wordsPerMinute": round(wpm, 1) if wpm else None,
            "fillerCount": filler_count,
            "fillerRatePerMinute": (round(filler_count / (duration / 60), 1) if duration and duration > 0 else None),
            "longPauseCount": None,
            "answerRelevance": 60, "structureScore": 55, "specificExampleScore": 50,
            "vocabularyRangeScore": 55, "grammarIssueCount": 0, "tenseIssueCount": 0,
            "clarityScore": 60, "pronunciationIntelligibility": 65,
            "codeSwitchingDetected": normalization["code_switching_detected"],
            "normalizedTranscript": normalization["clean_transcript"],
            "detectedLanguages": normalization["detected_languages"],
            "durationSeconds": duration,
        },
        "feedback": {
            "summary": "Analysis ran with limited grading data — these scores are a rough estimate. Record again to get more accurate feedback.",
            "strengths": ["You completed the recording."],
            "improvements": ["Try giving a longer answer next time.", "Aim to use the structure taught in the lesson."],
            "ieltsAdvice": [
                "Keep your answer in English throughout." if normalization["code_switching_detected"]
                else "Focus on staying organized — answer, reason, example."
            ],
            "bestSentence": "", "weakerSentence": "", "strongerVersion": "",
            "nextPracticeFocus": "Try the prompt again with one more example.",
        },
        "transcriptFeedback": [],
        "nextRecommendedLessonId": None,
        "transcript": transcript,
    }


async def analyze_ielts_core(part: str, prompt_text: str, transcript: str, duration: int | None, acoustic_metrics: dict | None) -> dict:
    normalization = await normalize_ielts_transcript(transcript)
    filler_count = count_fillers(normalization["clean_transcript"])

    try:
        msg = await _anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": IELTS_GRADING_PROMPT.format(
                    part=part,
                    prompt=prompt_text,
                    raw_transcript=transcript,
                    normalized_transcript=normalization["clean_transcript"],
                    code_switching_detected=normalization["code_switching_detected"],
                    duration_seconds=duration if duration else "unknown",
                    acoustic_metrics=json.dumps(acoustic_metrics or {}),
                ),
            }],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_blocks).strip()
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
    except Exception as e:
        print(json.dumps({"event": "ielts_grading_fallback", "error": str(e)}))
        return ielts_fallback_report(part, transcript, duration, filler_count, normalization)

    criteria = data.get("criteriaScores", {})
    for k in ("fluencyCoherence", "lexicalResource", "grammarRangeAccuracy", "pronunciation"):
        criteria[k] = clamp(int(criteria.get(k, 50)))

    overall_band, per_band = compute_practice_band(criteria)
    overall_100 = int(sum(criteria.values()) / 4)

    koras_metrics = data.get("korasMetrics", {}) or {}
    koras_metrics["fillerCount"] = max(filler_count, int(koras_metrics.get("fillerCount", 0)))
    if duration and duration > 0:
        koras_metrics["fillerRatePerMinute"] = round(koras_metrics["fillerCount"] / (duration / 60), 1)
        wpm = (len(normalization["clean_transcript"].split()) / duration) * 60
        koras_metrics["wordsPerMinute"] = round(wpm, 1)
    koras_metrics["codeSwitchingDetected"] = normalization["code_switching_detected"]
    koras_metrics["normalizedTranscript"] = normalization["clean_transcript"]
    koras_metrics["detectedLanguages"] = normalization["detected_languages"]
    koras_metrics["durationSeconds"] = duration

    feedback = data.get("feedback", {}) or {}
    if normalization["code_switching_detected"]:
        advice = list(feedback.get("ieltsAdvice") or [])
        advice.insert(0, "In IELTS practice, try to keep your full answer in English.")
        feedback["ieltsAdvice"] = advice

    return {
        "criteriaScores": criteria,
        "criteriaBand": per_band,
        "practiceBandEstimate": overall_band,
        "overallScore": overall_100,
        "korasMetrics": koras_metrics,
        "feedback": feedback,
        "transcriptFeedback": data.get("transcriptFeedback") or [],
        "nextRecommendedLessonId": data.get("nextRecommendedLessonId"),
        "transcript": transcript,
    }


@app.post("/analyze-ielts-speaking", response_model=IELTSReportModel, dependencies=[Depends(verify_internal_secret)])
async def analyze_ielts_speaking(request: Request) -> IELTSReportModel:
    req_id = uuid.uuid4().hex[:8]
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = await request.json()
        req = IELTSAnalyzeRequest(**body)
        _log("/analyze-ielts-speaking", req_id, req.user_id, mode="json", part=req.part)
        if not req.transcript or len(req.transcript.split()) < 5:
            raise HTTPException(400, "Transcript too short for analysis (minimum 5 words).")
        report = await analyze_ielts_core(
            part=req.part, prompt_text=req.prompt, transcript=req.transcript,
            duration=req.duration_seconds, acoustic_metrics=req.acoustic_metrics,
        )
        return IELTSReportModel(**report)

    form = await request.form()
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    part = str(form.get("part") or "part_1")
    prompt_text = str(form.get("prompt") or "")
    lesson_id = str(form.get("lesson_id") or "ielts")
    user_id = str(form.get("user_id") or "") or None
    _log("/analyze-ielts-speaking", req_id, user_id, mode="audio", part=part, lesson_id=lesson_id)

    async with _extract_sem:
        try:
            filename = getattr(audio_field, "filename", "audio") or "audio"
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, filename)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        features_task = asyncio.to_thread(extract_features, wav_bytes)
        transcript_task = transcribe(wav_bytes)
        try:
            features, transcript_data = await asyncio.gather(features_task, transcript_task)
        except Exception as e:
            raise HTTPException(500, f"Audio analysis failed: {e}") from e

    transcript_text = (transcript_data.get("text") or "").strip()
    if len(transcript_text.split()) < 5:
        raise HTTPException(400, "Audio too short or unclear. Please speak for longer (at least 5 words).")

    duration = int(features["duration"])
    acoustic_metrics = {
        "duration": features["duration"], "mean_f0": features["mean_f0"],
        "std_f0": features["std_f0"], "hnr": features["hnr"],
        "pause_count": features["pause_count"], "long_pause_count": features["long_pause_count"],
        "spectral_centroid": features["spectral_centroid"],
    }

    report = await analyze_ielts_core(
        part=part, prompt_text=prompt_text, transcript=transcript_text,
        duration=duration, acoustic_metrics=acoustic_metrics,
    )

    if part == "part_2" and duration < 60:
        advice = list(report["feedback"].get("ieltsAdvice") or [])
        advice.append(f"Part 2 long turns should run close to 2 minutes — you stopped at {duration}s.")
        report["feedback"]["ieltsAdvice"] = advice

    if report["korasMetrics"].get("longPauseCount") is None:
        report["korasMetrics"]["longPauseCount"] = int(features["long_pause_count"])

    _log("/analyze-ielts-speaking", req_id, user_id, event="done",
         lesson_id=lesson_id, part=part, duration=duration,
         band=report["practiceBandEstimate"])

    return IELTSReportModel(**report)

# --------------------------------------------------------------------------- #
#  /generate-interview-questions                                              #
# --------------------------------------------------------------------------- #

INTERVIEW_QUESTION_TYPES = {
    "general", "behavioral", "role_specific", "motivation",
    "strengths", "weakness", "teamwork", "leadership",
    "scenario", "technical_project", "company_fit",
}
INTERVIEW_FRAMEWORKS = {"STAR", "present_past_proof_future", "point_reason_example"}

def _slugify_question(q: str, idx: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:48]
    return f"gen-{idx}-{base}" if base else f"gen-{idx}"

INTERVIEW_FALLBACK_QUESTIONS: list[dict] = [
    {"question": "Tell me about yourself.", "type": "general", "framework": "present_past_proof_future", "duration": 90,
     "skills": ["structure", "confidence", "relevance"],
     "what_good": ["Starts with who you are now", "Connects past experience to the opportunity", "Includes one proof point", "Ends with why this opportunity makes sense"]},
    {"question": "Why are you interested in this opportunity?", "type": "motivation", "framework": "point_reason_example",
     "duration": 75, "skills": ["motivation", "company_fit"],
     "what_good": ["Specific reason", "Connection to role", "Avoids generic praise"]},
    {"question": "What are your strengths?", "type": "strengths", "framework": "point_reason_example",
     "duration": 75, "skills": ["specificity", "confidence"],
     "what_good": ["Names a real strength", "Gives evidence", "Connects to role"]},
    {"question": "What is one weakness you are working on?", "type": "weakness", "framework": "point_reason_example",
     "duration": 75, "skills": ["self_awareness", "growth"],
     "what_good": ["Honest but not damaging", "Shows action", "Shows improvement"]},
    {"question": "Tell me about a challenge you faced and how you handled it.", "type": "behavioral", "framework": "STAR",
     "duration": 120, "skills": ["STAR", "resilience"],
     "what_good": ["Clear situation", "Specific action", "Measurable result"]},
    {"question": "Describe a time you worked in a team.", "type": "teamwork", "framework": "STAR",
     "duration": 120, "skills": ["collaboration", "STAR"],
     "what_good": ["Explains team goal", "Shows your contribution", "Includes result"]},
    {"question": "Tell me about a time you showed leadership.", "type": "leadership", "framework": "STAR",
     "duration": 120, "skills": ["leadership", "STAR"],
     "what_good": ["Shows initiative", "Explains action", "Shows impact"]},
    {"question": "Why should we choose you?", "type": "company_fit", "framework": "point_reason_example",
     "duration": 90, "skills": ["confidence", "relevance"],
     "what_good": ["Clear value proposition", "Role fit", "Evidence"]},
    {"question": "Where do you see yourself in five years?", "type": "general", "framework": "point_reason_example",
     "duration": 75, "skills": ["career_goals", "clarity"],
     "what_good": ["Realistic goal", "Connects to role", "Shows ambition"]},
    {"question": "Do you have any questions for us?", "type": "company_fit", "framework": None,
     "duration": 60, "skills": ["curiosity", "professionalism"],
     "what_good": ["Asks thoughtful question", "Shows preparation", "Avoids salary-only focus"]},
]

def interview_fallback_question_list() -> list[GeneratedInterviewQuestion]:
    out: list[GeneratedInterviewQuestion] = []
    for i, q in enumerate(INTERVIEW_FALLBACK_QUESTIONS):
        out.append(GeneratedInterviewQuestion(
            id=_slugify_question(q["question"], i),
            question=q["question"],
            type=q["type"],
            skillTags=list(q["skills"]),
            suggestedDurationSeconds=int(q["duration"]),
            answerFramework=q["framework"],
            whatGoodLooksLike=list(q["what_good"]),
        ))
    return out

def normalize_generated_question(raw: dict, idx: int) -> GeneratedInterviewQuestion:
    q_text = str(raw.get("question") or "").strip()
    if not q_text:
        raise ValueError("missing question text")
    q_type = str(raw.get("type") or "general").strip()
    if q_type not in INTERVIEW_QUESTION_TYPES:
        q_type = "general"
    framework = raw.get("answerFramework")
    framework_str: str | None = framework.strip() if isinstance(framework, str) else None
    if framework_str not in INTERVIEW_FRAMEWORKS:
        framework_str = None
    skills = [str(s) for s in (raw.get("skillTags") or []) if isinstance(s, str)]
    what_good = [str(s) for s in (raw.get("whatGoodLooksLike") or []) if isinstance(s, str)]
    try:
        duration = max(30, min(180, int(raw.get("suggestedDurationSeconds") or 90)))
    except (TypeError, ValueError):
        duration = 90
    provided_id = str(raw.get("id") or "").strip()
    return GeneratedInterviewQuestion(
        id=provided_id or _slugify_question(q_text, idx),
        question=q_text,
        type=q_type,
        skillTags=skills,
        suggestedDurationSeconds=duration,
        answerFramework=framework_str,
        whatGoodLooksLike=what_good,
    )

INTERVIEW_GENERATE_PROMPT = """You are Koras Interview Prep Coach.

Generate spoken-interview practice questions for the user's scenario.

Inputs (any may be missing):
- Title: {title}
- Job role: {job_role}
- Company: {company}
- Interview type: {interview_type}
- Experience level: {experience_level}
- Pasted job description: \"\"\"{description}\"\"\"
- Notes: \"\"\"{notes}\"\"\"

Goals:
- Help the candidate practice spoken interview answers, not technical drills.
- Generate 10 to 14 questions.
- Mix question types so the candidate trains different skills.

Required mix (when context allows):
- Always include "Tell me about yourself" unless the scenario is highly specialized.
- At least 2 behavioral questions (use STAR framework).
- At least 2 role-specific questions if the job context is available.
- At least 1 motivation question (why this role / company).
- At least 1 strengths and 1 weakness question.
- At least 1 teamwork or leadership question.
- End with "Do you have any questions for us?" or similar closing prompt.

If JD and context are thin, generate sensible general interview questions.

Frameworks:
- Use "STAR" for behavioral questions.
- Use "present_past_proof_future" for "Tell me about yourself".
- Use "point_reason_example" for opinions, strengths, weaknesses, motivation.
- Use null when no framework applies (e.g. closing "do you have questions").

Suggested durations:
- general / motivation / strengths / weakness: 75 seconds
- behavioral / teamwork / leadership / scenario: 120 seconds
- closing / "do you have any questions": 60 seconds

Question type must be one of:
general | behavioral | role_specific | motivation | strengths | weakness |
teamwork | leadership | scenario | technical_project | company_fit

Return STRICT JSON only — no markdown, no commentary:
{{
  "questions": [
    {{
      "id": "kebab-case-id",
      "question": "string",
      "type": "general|behavioral|role_specific|motivation|strengths|weakness|teamwork|leadership|scenario|technical_project|company_fit",
      "skillTags": ["string"],
      "suggestedDurationSeconds": 75,
      "answerFramework": "STAR|present_past_proof_future|point_reason_example|null",
      "whatGoodLooksLike": ["short bullet", "short bullet"]
    }}
  ],
  "extractedContext": {{
    "role": "string or null",
    "seniority": "string or null",
    "skills": ["string"],
    "responsibilities": ["string"]
  }}
}}

Return only the JSON object."""


@app.post("/generate-interview-questions", response_model=GenerateInterviewQuestionsResponse, dependencies=[Depends(verify_internal_secret)])
async def generate_interview_questions(req: GenerateInterviewQuestionsRequest) -> GenerateInterviewQuestionsResponse:
    req_id = uuid.uuid4().hex[:8]
    _log("/generate-interview-questions", req_id, None, role=req.jobRole, type=req.interviewType)

    if not any([(req.description or "").strip(), (req.jobRole or "").strip(),
                (req.notes or "").strip(), (req.title or "").strip()]):
        return GenerateInterviewQuestionsResponse(
            questions=interview_fallback_question_list(),
            extractedContext={},
            warning="No scenario context provided — returning default questions.",
        )

    prompt = INTERVIEW_GENERATE_PROMPT.format(
        title=(req.title or "unspecified"),
        job_role=(req.jobRole or "unspecified"),
        company=(req.company or "unspecified"),
        interview_type=(req.interviewType or "general"),
        experience_level=(req.experienceLevel or "unspecified"),
        description=(req.description or "")[:6000],
        notes=(req.notes or "")[:2000],
    )

    try:
        msg = await _anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_blocks).strip()
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
    except Exception as e:
        _log("/generate-interview-questions", req_id, None, event="llm_fallback", error=str(e))
        return GenerateInterviewQuestionsResponse(
            questions=interview_fallback_question_list(),
            extractedContext={},
            warning="Question generation fell back to defaults. Try again or edit the questions below.",
        )

    raw_questions = data.get("questions") or []
    normalized: list[GeneratedInterviewQuestion] = []
    for i, q in enumerate(raw_questions):
        if not isinstance(q, dict):
            continue
        try:
            normalized.append(normalize_generated_question(q, i))
        except Exception as e:
            _log("/generate-interview-questions", req_id, None, event="skip_bad_question", error=str(e))
    if not normalized:
        return GenerateInterviewQuestionsResponse(
            questions=interview_fallback_question_list(),
            extractedContext={},
            warning="Question generation returned no usable items. Showing defaults.",
        )

    extracted = data.get("extractedContext") or {}
    if not isinstance(extracted, dict):
        extracted = {}

    _log("/generate-interview-questions", req_id, None, event="done", questions=len(normalized))
    return GenerateInterviewQuestionsResponse(questions=normalized, extractedContext=extracted, warning=None)

# --------------------------------------------------------------------------- #
#  /analyze-interview-answer                                                  #
# --------------------------------------------------------------------------- #

INTERVIEW_SCORE_WEIGHTS = {
    "relevance": 0.20, "structure": 0.15, "specificity": 0.15,
    "delivery": 0.10, "confidence": 0.10, "fluency": 0.10,
    "grammar": 0.10, "conciseness": 0.05, "professionalism": 0.05,
}

def interview_weighted_overall(scores: dict) -> int:
    return clamp(sum(int(scores.get(k, 50)) * w for k, w in INTERVIEW_SCORE_WEIGHTS.items()))

def _interview_default_metrics(transcript: str, duration: int | None, filler_count: int) -> dict:
    word_count = len(transcript.split())
    wpm: float | None = None
    filler_rate: float | None = None
    if duration and duration > 0:
        wpm = round((word_count / duration) * 60, 1)
        filler_rate = round(filler_count / (duration / 60), 1)
    return {
        "wordsPerMinute": wpm, "fillerCount": filler_count, "fillerRatePerMinute": filler_rate,
        "longPauseCount": None, "durationSeconds": duration, "grammarIssueCount": 0,
        "tenseIssueCount": 0, "specificityMarkers": 0,
        "ramblingDetected": word_count > 240, "transcriptWordCount": word_count,
    }

def interview_answer_fallback(question: str, question_type: str | None, transcript: str, duration: int | None, filler_count: int) -> dict:
    word_count = len(transcript.split())
    base = 55
    length_bonus = min(15, word_count // 14)
    filler_penalty = min(20, filler_count * 3)
    raw_score = max(40, min(78, base + length_bonus - filler_penalty))
    scores = {
        "delivery": raw_score, "relevance": max(40, raw_score - 2),
        "structure": max(40, raw_score - 5), "specificity": max(40, raw_score - 8),
        "confidence": max(40, raw_score - 3), "fluency": clamp(100 - filler_count * 6),
        "grammar": max(45, raw_score), "conciseness": max(45, raw_score - 2),
        "professionalism": max(50, raw_score),
    }
    scores["overall"] = interview_weighted_overall(scores)
    star_score: int | None = None
    if (question_type or "") in {"behavioral", "teamwork", "leadership", "scenario"}:
        star_score = max(40, raw_score - 8)
    scores["star"] = star_score
    return {
        "transcript": transcript, "scores": scores,
        "metrics": _interview_default_metrics(transcript, duration, filler_count),
        "feedback": {
            "summary": "Analysis ran with limited grading data — these scores are a rough estimate. Try recording again with a fuller answer for a more accurate report.",
            "strengths": ["You completed the recording."],
            "improvements": ["Aim for a longer, more detailed answer.", "Use a clear structure (e.g. STAR for behavioural questions)."],
            "repeatedMistakes": [], "bestLine": None, "weakerLine": None,
            "strongerVersion": None, "nextPracticeFocus": "Try the same question again with one concrete example.",
        },
        "frameworkAnalysis": {
            "framework": "STAR" if star_score is not None else "none",
            "present": None, "past": None, "proof": None, "future": None,
            "situation": None, "task": None, "action": None, "result": None, "missingParts": [],
        },
        "question": question, "questionType": question_type,
    }


INTERVIEW_ANSWER_PROMPT = """You are Koras Interview Prep Coach.

Grade the candidate's spoken interview answer.

Context:
- Question: \"\"\"{question}\"\"\"
- Question type: {question_type}
- Scenario title: {scenario_title}
- Role: {job_role}
- Company: {company}
- Job/scenario description: \"\"\"{description}\"\"\"
- Duration seconds: {duration_seconds}
- Transcript: \"\"\"{transcript}\"\"\"
- Acoustic / delivery metrics: {acoustic_metrics}

Score the answer on:
- relevance (how well it answers THIS question)
- structure (is there a clear arc / framework)
- specificity (concrete examples, names, numbers, outcomes)
- confidence (assertive language, low hedging)
- fluency (smooth, low filler)
- grammar (sentence-level correctness)
- conciseness (no rambling)
- professionalism (tone, register)
- delivery (pacing, energy — infer from metrics + word choice)

For behavioral questions (or any STAR-tagged question), also produce a STAR
analysis identifying Situation / Task / Action / Result and listing any
missing parts.

For "Tell me about yourself" or any present_past_proof_future tagged question,
identify Present / Past / Proof / Future and missing parts.

Be specific, fair, and practical.
Do not be harsh. Do not over-penalize accent. Do not invent facts that
aren't in the transcript. If the candidate didn't answer the question,
say so plainly in the summary and give them a constructive next step.

Return STRICT JSON only — no markdown:
{{
  "scores": {{
    "overall": <int 0-100>,
    "delivery": <int 0-100>,
    "relevance": <int 0-100>,
    "structure": <int 0-100>,
    "specificity": <int 0-100>,
    "confidence": <int 0-100>,
    "fluency": <int 0-100>,
    "grammar": <int 0-100>,
    "conciseness": <int 0-100>,
    "professionalism": <int 0-100>,
    "star": <int 0-100 or null>
  }},
  "metrics": {{
    "wordsPerMinute": <number or null>,
    "fillerCount": <int>,
    "fillerRatePerMinute": <number or null>,
    "longPauseCount": <int or null>,
    "durationSeconds": <number or null>,
    "grammarIssueCount": <int>,
    "tenseIssueCount": <int>,
    "specificityMarkers": <int>,
    "ramblingDetected": <true|false>,
    "transcriptWordCount": <int>
  }},
  "feedback": {{
    "summary": "<2-3 sentences>",
    "strengths": ["..."],
    "improvements": ["..."],
    "repeatedMistakes": ["..."],
    "bestLine": "<verbatim sentence or null>",
    "weakerLine": "<verbatim sentence or null>",
    "strongerVersion": "<rewrite of weaker line or null>",
    "nextPracticeFocus": "<one-line focus or null>"
  }},
  "frameworkAnalysis": {{
    "framework": "STAR | present_past_proof_future | point_reason_example | none",
    "present": "<excerpt or null>",
    "past": "<excerpt or null>",
    "proof": "<excerpt or null>",
    "future": "<excerpt or null>",
    "situation": "<excerpt or null>",
    "task": "<excerpt or null>",
    "action": "<excerpt or null>",
    "result": "<excerpt or null>",
    "missingParts": ["..."]
  }}
}}

Return only the JSON object."""


async def grade_interview_answer(
    question: str, question_type: str | None, scenario_title: str | None,
    job_role: str | None, company: str | None, description: str | None,
    transcript: str, duration: int | None, acoustic_metrics: dict | None,
) -> dict:
    filler_count = count_fillers(transcript)
    prompt = INTERVIEW_ANSWER_PROMPT.format(
        question=question,
        question_type=question_type or "general",
        scenario_title=(scenario_title or "unspecified"),
        job_role=(job_role or "unspecified"),
        company=(company or "unspecified"),
        description=(description or "")[:4000],
        duration_seconds=duration if duration else "unknown",
        transcript=transcript[:8000],
        acoustic_metrics=json.dumps(acoustic_metrics or {}),
    )

    try:
        msg = await _anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_blocks).strip()
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
    except Exception as e:
        print(json.dumps({"event": "interview_answer_llm_fallback", "error": str(e)}))
        return interview_answer_fallback(question, question_type, transcript, duration, filler_count)

    scores_raw = data.get("scores") or {}
    clamped: dict = {}
    for k in INTERVIEW_SCORE_WEIGHTS:
        clamped[k] = clamp(int(scores_raw.get(k, 50)))
    star_val = scores_raw.get("star")
    clamped["star"] = clamp(int(star_val)) if star_val is not None else None
    clamped["overall"] = interview_weighted_overall(clamped)

    metrics = data.get("metrics") or {}
    defaults = _interview_default_metrics(transcript, duration, filler_count)
    for key, default_value in defaults.items():
        if metrics.get(key) is None:
            metrics[key] = default_value
    metrics["fillerCount"] = max(int(metrics.get("fillerCount") or 0), filler_count)
    if duration and duration > 0 and metrics.get("fillerRatePerMinute") is None:
        metrics["fillerRatePerMinute"] = round(metrics["fillerCount"] / (duration / 60), 1)

    feedback = data.get("feedback") or {}
    for k in ("strengths", "improvements", "repeatedMistakes"):
        if not isinstance(feedback.get(k), list):
            feedback[k] = []
    if not isinstance(feedback.get("summary"), str) or not feedback["summary"].strip():
        feedback["summary"] = "Your answer was scored — see the strengths and improvements below."

    framework_analysis = data.get("frameworkAnalysis") or {}
    if not isinstance(framework_analysis.get("missingParts"), list):
        framework_analysis["missingParts"] = []

    return {
        "transcript": transcript, "scores": clamped, "metrics": metrics,
        "feedback": feedback, "frameworkAnalysis": framework_analysis,
        "question": question, "questionType": question_type,
    }


@app.post("/analyze-interview-answer", response_model=InterviewAnswerReport, dependencies=[Depends(verify_internal_secret)])
async def analyze_interview_answer(request: Request) -> InterviewAnswerReport:
    req_id = uuid.uuid4().hex[:8]
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = await request.json()
        req = InterviewAnalyzeJSONRequest(**body)
        _log("/analyze-interview-answer", req_id, None, mode="json", question_type=req.questionType)
        transcript = (req.transcript or "").strip()
        if len(transcript.split()) < 5:
            raise HTTPException(400, "Transcript too short for analysis (minimum 5 words).")
        report = await grade_interview_answer(
            question=req.question, question_type=req.questionType,
            scenario_title=req.scenarioTitle, job_role=req.jobRole,
            company=req.company, description=req.description,
            transcript=transcript, duration=req.durationSeconds, acoustic_metrics=None,
        )
        return InterviewAnswerReport(**report)

    form = await request.form()
    audio_field = form.get("audio")
    if audio_field is None or not hasattr(audio_field, "read"):
        raise HTTPException(400, "Missing 'audio' file in multipart payload.")

    audio_bytes = await audio_field.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload.")
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 10MB).")

    question = str(form.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "Missing 'question' field.")
    question_type_raw = str(form.get("questionType") or "").strip()
    question_type = question_type_raw if question_type_raw else None
    scenario_title = str(form.get("scenarioTitle") or "") or None
    job_role = str(form.get("jobRole") or "") or None
    company = str(form.get("company") or "") or None
    description = str(form.get("description") or "") or None
    user_id = str(form.get("user_id") or "") or None
    duration_field = form.get("durationSeconds")
    duration: int | None = None
    if duration_field is not None:
        try:
            duration = int(str(duration_field))
        except (TypeError, ValueError):
            duration = None

    _log("/analyze-interview-answer", req_id, user_id, mode="audio", question_type=question_type)

    async with _extract_sem:
        try:
            filename = getattr(audio_field, "filename", "audio") or "audio"
            wav_bytes = await asyncio.to_thread(normalize_audio, audio_bytes, filename)
        except Exception as e:
            raise HTTPException(415, f"Could not decode audio: {e}") from e

        features_task = asyncio.to_thread(extract_features, wav_bytes)
        transcript_task = transcribe(wav_bytes)
        try:
            features, transcript_data = await asyncio.gather(features_task, transcript_task)
        except Exception as e:
            raise HTTPException(500, f"Audio analysis failed: {e}") from e

    transcript_text = (transcript_data.get("text") or "").strip()
    if len(transcript_text.split()) < 5:
        raise HTTPException(400, "We couldn't detect a meaningful answer. Please try again and speak for at least a few sentences.")

    measured_duration = duration if duration and duration > 0 else int(features["duration"])
    acoustic_metrics = {
        "duration": features["duration"], "mean_f0": features["mean_f0"],
        "std_f0": features["std_f0"], "hnr": features["hnr"],
        "pause_count": features["pause_count"], "long_pause_count": features["long_pause_count"],
        "spectral_centroid": features["spectral_centroid"],
    }

    report = await grade_interview_answer(
        question=question, question_type=question_type, scenario_title=scenario_title,
        job_role=job_role, company=company, description=description,
        transcript=transcript_text, duration=measured_duration, acoustic_metrics=acoustic_metrics,
    )

    if report["metrics"].get("longPauseCount") is None:
        report["metrics"]["longPauseCount"] = int(features["long_pause_count"])

    _log("/analyze-interview-answer", req_id, user_id, event="done",
         question_type=question_type, duration=measured_duration,
         words=len(transcript_text.split()), overall=report["scores"]["overall"])

    return InterviewAnswerReport(**report)

# --------------------------------------------------------------------------- #
#  /generate-daily-plan                                                       #
# --------------------------------------------------------------------------- #

DAILY_PLAN_ALLOWED_PROGRAMS = {"ai-tutor", "ielts-speaking", "interview-prep"}
DAILY_PLAN_ROUTE_BY_PROGRAM = {
    "ai-tutor": "/ai-tutor",
    "ielts-speaking": "/ielts",
    "interview-prep": "/interview-prep",
}
DAILY_PLAN_ALLOWED_TYPES = {"program_session", "review", "reflection", "streak_save"}

DAILY_PLAN_PROMPT = """You are Koras, a daily speaking coach.

You will receive structured learner signals and a rules-generated daily plan. Your job is to rewrite the plan into clear, encouraging, student-friendly copy.

Important rules:
- Do NOT invent new programs.
- Do NOT invent new routes.
- Do NOT add more than 3 items.
- Keep each recommendation practical and short.
- The plan should feel personalized, not generic.
- Focus on ONE clear focus area for the next 24 hours.
- Do not mention internal table names or raw scores unless useful.
- Do not expose sensitive transcripts or recordings.
- If the learner has little history, encourage a baseline AI Tutor session.

Allowed program_id values (must match exactly): "ai-tutor", "ielts-speaking", "interview-prep".
Allowed type values: "program_session", "review", "reflection", "streak_save".
Allowed routes per program: "ai-tutor" → "/ai-tutor", "ielts-speaking" → "/ielts", "interview-prep" → "/interview-prep".

Return STRICT JSON only — no markdown, no commentary — matching this schema:
{{
  "summary": "string",
  "focus_area": "string",
  "advice": "string",
  "items": [
    {{
      "item_id": "string",
      "type": "program_session | review | reflection | streak_save",
      "program_id": "ai-tutor | ielts-speaking | interview-prep",
      "route": "string",
      "title": "string",
      "reason": "string",
      "estimated_minutes": 5,
      "priority": 1,
      "status": "pending",
      "completed_at": null
    }}
  ]
}}

LEARNER SIGNALS (JSON):
{signals_json}

RULES-GENERATED PLAN (JSON):
{rules_plan_json}

Return only the JSON object."""


def _coerce_int(raw, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _daily_plan_fallback_items(rules_plan: dict) -> list[DailyPlanItemModel]:
    items_raw = rules_plan.get("items") or []
    out: list[DailyPlanItemModel] = []
    for raw in items_raw:
        if not isinstance(raw, dict):
            continue
        program_id = raw.get("program_id")
        if program_id not in DAILY_PLAN_ALLOWED_PROGRAMS:
            continue
        route = raw.get("route") or DAILY_PLAN_ROUTE_BY_PROGRAM.get(program_id, "")
        if not route:
            continue
        type_ = raw.get("type", "program_session")
        if type_ not in DAILY_PLAN_ALLOWED_TYPES:
            type_ = "program_session"
        priority = max(1, min(3, _coerce_int(raw.get("priority"), 2)))
        minutes = max(2, min(45, _coerce_int(raw.get("estimated_minutes"), 10)))
        out.append(DailyPlanItemModel(
            item_id=raw.get("item_id"),
            type=type_,
            program_id=program_id,
            route=route,
            title=str(raw.get("title") or "")[:120],
            reason=str(raw.get("reason") or "")[:280],
            estimated_minutes=minutes,
            priority=priority,
            status="pending",
            completed_at=None,
        ))
        if len(out) >= 3:
            break
    return out


def _daily_plan_fallback(rules_plan: dict) -> GenerateDailyPlanResponse:
    return GenerateDailyPlanResponse(
        summary=str(rules_plan.get("summary") or "")[:500],
        focus_area=str(rules_plan.get("focus_area") or "")[:80],
        advice=str(rules_plan.get("advice") or "")[:800],
        items=_daily_plan_fallback_items(rules_plan),
    )


@app.post("/generate-daily-plan", response_model=GenerateDailyPlanResponse, dependencies=[Depends(verify_internal_secret)])
async def generate_daily_plan(req: GenerateDailyPlanRequest) -> GenerateDailyPlanResponse:
    req_id = uuid.uuid4().hex[:8]
    _log("/generate-daily-plan", req_id, req.user_id)

    rules_plan = req.rules_plan or {}
    fallback = _daily_plan_fallback(rules_plan)
    if not fallback.items:
        return fallback

    try:
        signals_json = json.dumps(req.signals, ensure_ascii=False)[:6000]
        rules_plan_json = json.dumps(rules_plan, ensure_ascii=False)[:3000]
    except (TypeError, ValueError):
        return fallback

    prompt = DAILY_PLAN_PROMPT.format(signals_json=signals_json, rules_plan_json=rules_plan_json)

    try:
        msg = await _anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        raw = "\n".join(text_blocks).strip()
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
    except Exception as e:
        _log("/generate-daily-plan", req_id, req.user_id, event="llm_fallback", error=str(e))
        return fallback

    if not isinstance(data, dict):
        return fallback

    rules_items_by_program: dict[str, dict] = {}
    for raw in (rules_plan.get("items") or []):
        if isinstance(raw, dict) and raw.get("program_id") in DAILY_PLAN_ALLOWED_PROGRAMS:
            rules_items_by_program[str(raw["program_id"])] = raw

    out_items: list[DailyPlanItemModel] = []
    for raw in (data.get("items") or []):
        if not isinstance(raw, dict):
            continue
        program_id = raw.get("program_id")
        if program_id not in DAILY_PLAN_ALLOWED_PROGRAMS:
            continue
        route = raw.get("route") or DAILY_PLAN_ROUTE_BY_PROGRAM.get(program_id, "")
        if not route:
            continue
        type_ = raw.get("type", "program_session")
        if type_ not in DAILY_PLAN_ALLOWED_TYPES:
            type_ = "program_session"
        rules_item = rules_items_by_program.get(program_id, {})
        priority = max(1, min(3, _coerce_int(raw.get("priority"), _coerce_int(rules_item.get("priority"), 2))))
        minutes_raw = _coerce_int(raw.get("estimated_minutes"), _coerce_int(rules_item.get("estimated_minutes"), 10))
        minutes = max(2, min(45, minutes_raw))
        title = (str(raw.get("title") or rules_item.get("title") or "")).strip()[:120]
        reason = (str(raw.get("reason") or rules_item.get("reason") or "")).strip()[:280]
        if not title or not reason:
            continue
        item_id = (
            str(raw.get("item_id")) if isinstance(raw.get("item_id"), str) and raw.get("item_id")
            else rules_item.get("item_id")
        )
        out_items.append(DailyPlanItemModel(
            item_id=item_id, type=type_, program_id=str(program_id), route=str(route),
            title=title, reason=reason, estimated_minutes=minutes, priority=priority,
            status="pending", completed_at=None,
        ))
        if len(out_items) >= 3:
            break

    if not out_items:
        return fallback

    _log("/generate-daily-plan", req_id, req.user_id, event="done", items=len(out_items))
    return GenerateDailyPlanResponse(
        summary=(str(data.get("summary") or fallback.summary)).strip()[:500],
        focus_area=(str(data.get("focus_area") or fallback.focus_area)).strip()[:80],
        advice=(str(data.get("advice") or fallback.advice)).strip()[:800],
        items=out_items,
    )
