"""
Koras Backend API — FastAPI, Docker/GCP

Run locally:
    uvicorn main:app --reload --port 8000

Run in Docker (via gunicorn):
    gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 --bind 0.0.0.0:8000
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGIN_REGEX, ALLOWED_ORIGINS
from routes import ai_tutor, analyze, daily_plan, health, ielts, interview, listening, voice_foundations, vocabulary

app = FastAPI(title="Koras Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(ai_tutor.router)
app.include_router(ielts.router)
app.include_router(interview.router)
app.include_router(daily_plan.router)
app.include_router(listening.router)
app.include_router(voice_foundations.router)
app.include_router(vocabulary.router)
