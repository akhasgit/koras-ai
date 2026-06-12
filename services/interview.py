import json

from config import anthropic_client, CLAUDE_MODEL
from models.interview import GeneratedInterviewQuestion, GenerateInterviewQuestionsResponse
from prompts.interview import INTERVIEW_ANSWER_PROMPT, INTERVIEW_GENERATE_PROMPT
from services.scoring import INTERVIEW_SCORE_WEIGHTS, interview_weighted_overall
from utils.text import clamp, count_fillers, slugify_question, strip_fences

INTERVIEW_QUESTION_TYPES = {
    "general", "behavioral", "role_specific", "motivation",
    "strengths", "weakness", "teamwork", "leadership",
    "scenario", "technical_project", "company_fit",
}
INTERVIEW_FRAMEWORKS = {"STAR", "present_past_proof_future", "point_reason_example"}

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
    return [
        GeneratedInterviewQuestion(
            id=slugify_question(q["question"], i),
            question=q["question"], type=q["type"],
            skillTags=list(q["skills"]),
            suggestedDurationSeconds=int(q["duration"]),
            answerFramework=q["framework"],
            whatGoodLooksLike=list(q["what_good"]),
        )
        for i, q in enumerate(INTERVIEW_FALLBACK_QUESTIONS)
    ]


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
    try:
        duration = max(30, min(180, int(raw.get("suggestedDurationSeconds") or 90)))
    except (TypeError, ValueError):
        duration = 90
    return GeneratedInterviewQuestion(
        id=str(raw.get("id") or "").strip() or slugify_question(q_text, idx),
        question=q_text, type=q_type,
        skillTags=[str(s) for s in (raw.get("skillTags") or []) if isinstance(s, str)],
        suggestedDurationSeconds=duration,
        answerFramework=framework_str,
        whatGoodLooksLike=[str(s) for s in (raw.get("whatGoodLooksLike") or []) if isinstance(s, str)],
    )


async def generate_interview_questions(
    title: str | None, job_role: str | None, company: str | None,
    interview_type: str, experience_level: str | None,
    description: str | None, notes: str | None,
) -> GenerateInterviewQuestionsResponse:
    prompt = INTERVIEW_GENERATE_PROMPT.format(
        title=(title or "unspecified"), job_role=(job_role or "unspecified"),
        company=(company or "unspecified"), interview_type=interview_type,
        experience_level=(experience_level or "unspecified"),
        description=(description or "")[:6000], notes=(notes or "")[:2000],
    )
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))
    except Exception:
        return GenerateInterviewQuestionsResponse(
            questions=interview_fallback_question_list(), extractedContext={},
            warning="Question generation fell back to defaults. Try again or edit the questions below.",
        )

    normalized = []
    for i, q in enumerate(data.get("questions") or []):
        if isinstance(q, dict):
            try:
                normalized.append(normalize_generated_question(q, i))
            except Exception:
                pass

    if not normalized:
        return GenerateInterviewQuestionsResponse(
            questions=interview_fallback_question_list(), extractedContext={},
            warning="Question generation returned no usable items. Showing defaults.",
        )

    extracted = data.get("extractedContext") or {}
    return GenerateInterviewQuestionsResponse(
        questions=normalized,
        extractedContext=extracted if isinstance(extracted, dict) else {},
        warning=None,
    )


def _interview_default_metrics(transcript: str, duration: int | None, filler_count: int) -> dict:
    word_count = len(transcript.split())
    wpm = round((word_count / duration) * 60, 1) if duration and duration > 0 else None
    filler_rate = round(filler_count / (duration / 60), 1) if duration and duration > 0 else None
    return {
        "wordsPerMinute": wpm, "fillerCount": filler_count, "fillerRatePerMinute": filler_rate,
        "longPauseCount": None, "durationSeconds": duration, "grammarIssueCount": 0,
        "tenseIssueCount": 0, "specificityMarkers": 0,
        "ramblingDetected": word_count > 240, "transcriptWordCount": word_count,
    }


def interview_answer_fallback(
    question: str, question_type: str | None, transcript: str, duration: int | None, filler_count: int
) -> dict:
    word_count = len(transcript.split())
    raw_score = max(40, min(78, 55 + min(15, word_count // 14) - min(20, filler_count * 3)))
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
            "summary": "Analysis ran with limited grading data. Try recording again for a more accurate report.",
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


async def grade_interview_answer(
    question: str, question_type: str | None, scenario_title: str | None,
    job_role: str | None, company: str | None, description: str | None,
    transcript: str, duration: int | None, acoustic_metrics: dict | None,
) -> dict:
    filler_count = count_fillers(transcript)
    try:
        msg = await anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2500,
            messages=[{"role": "user", "content": INTERVIEW_ANSWER_PROMPT.format(
                question=question, question_type=question_type or "general",
                scenario_title=(scenario_title or "unspecified"),
                job_role=(job_role or "unspecified"), company=(company or "unspecified"),
                description=(description or "")[:4000],
                duration_seconds=duration if duration else "unknown",
                transcript=transcript[:8000],
                acoustic_metrics=json.dumps(acoustic_metrics or {}),
            )}],
        )
        text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        data = json.loads(strip_fences("\n".join(text_blocks).strip()))
    except Exception:
        return interview_answer_fallback(question, question_type, transcript, duration, filler_count)

    scores_raw = data.get("scores") or {}
    clamped: dict = {k: clamp(int(scores_raw.get(k, 50))) for k in INTERVIEW_SCORE_WEIGHTS}
    star_val = scores_raw.get("star")
    clamped["star"] = clamp(int(star_val)) if star_val is not None else None
    clamped["overall"] = interview_weighted_overall(clamped)

    metrics = data.get("metrics") or {}
    for key, val in _interview_default_metrics(transcript, duration, filler_count).items():
        if metrics.get(key) is None:
            metrics[key] = val
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
