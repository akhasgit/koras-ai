import asyncio
import os

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
WHISPER_MODEL = "whisper-1"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_TRANSCRIPT_WORDS = 10

_default_origins = ",".join([
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://koras.com",
    "https://www.koras.com",
    "https://koras.vercel.app",
    "https://koras-site.vercel.app",
])
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")

ALLOWED_ORIGIN_REGEX: str | None = None

KORAS_INTERNAL_SECRET = os.environ.get("KORAS_INTERNAL_SECRET", "")

# One connection pool per process
openai_client = AsyncOpenAI()
anthropic_client = AsyncAnthropic()

# Cap concurrent CPU-heavy work to available cores
extract_sem = asyncio.Semaphore(os.cpu_count() or 2)
