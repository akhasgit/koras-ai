import re


_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END = re.compile(r"\s*```\s*$")

_FILLER_PATTERNS = [
    r"\bum\b", r"\buh\b", r"\berm\b", r"\blike\b",
    r"\byou know\b", r"\bbasically\b", r"\bactually\b",
    r"\bi mean\b", r"\bsort of\b", r"\bkind of\b", r"\bkinda\b",
]
_filler_regex = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = _FENCE_START.sub("", raw)
    raw = _FENCE_END.sub("", raw)
    if not raw.startswith("{"):
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            raw = raw[first : last + 1]
    return raw.strip()


def count_fillers(text: str) -> int:
    return len(_filler_regex.findall(text))


def clamp(v: int | float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))


def slugify_question(q: str, idx: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:48]
    return f"gen-{idx}-{base}" if base else f"gen-{idx}"
