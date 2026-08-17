CLARITY_PASSAGE_PROMPT = """You are writing a short read-aloud passage for an English intelligibility programme.

Constraints (non-negotiable):
- Length: 55–75 words, 5–7 sentences.
- Readability: grade band {grade_band}.
- ZERO proper nouns: no names, brands, places, acronyms.
- Neutral, first-person, everyday register. Safe for a 12-year-old to read aloud in class.
- No politics, religion, health conditions, or family conflict.
- At least one question and one sentence with a contrastive-stress opportunity.
- Seed at least two instances of each of these target features: {features_json}.
- Use only high-frequency, phonetically transparent vocabulary.

Return STRICT JSON only:
{{
  "passage_text": "string",
  "seeded_feature_ids": ["feature_id", ...]
}}

passage_seed (for your own consistency, do not mention it): {passage_seed}
"""


CLARITY_READ_PROMPT = """You are Koras, an intelligibility coach. A learner just read this passage aloud.

Passage:
\"\"\"{passage}\"\"\"

Transcript (Whisper — may hide pronunciation errors):
\"\"\"{transcript}\"\"\"

Active training features: {features_json}
Acoustic notes: WPM={wpm}, pauses={pause_count}, long pauses={long_pause_count}, mean F0={mean_f0}, phone_available={phone_available}

Classify errors into classes: prosodic_rhythm, prosodic_stress, prosodic_intonation, segmental, decoding, pace.
Do NOT invent proper-noun errors. Do NOT treat Whisper repairs as proof the word was clear.
Rank 2–6 features by intelligibility cost (prosody first).

Return STRICT JSON only:
{{
  "error_map": {{
    "dominant_class": "prosodic_rhythm | prosodic_stress | prosodic_intonation | segmental | decoding | pace",
    "features": [
      {{"feature_id": "string", "feature_class": "prosodic | segmental | decoding", "note": "one short sentence", "severity": 1}}
    ]
  }},
  "feature_findings": [
    {{"feature_id": "string", "hit": false, "word": "string", "note": "string"}}
  ],
  "coach_note": "2 sentences, second person. Do not mention Whisper, scores, or models.",
  "off_script": false
}}
"""


CLARITY_DRILL_COPY_PROMPT = """Write short drill prompts for an intelligibility session.

Dominant class: {dominant_class}
Selected drill types: {drill_types_json}
Active features: {features_json}

Rules:
- No proper nouns.
- Everyday vocabulary.
- For decoding: hear-first lines, never a cold minimal pair.
- Each drill has 2–4 items.

Return STRICT JSON only:
{{
  "drills": [
    {{
      "id": "d1",
      "drill_type": "shadowing",
      "feature_id": "rhythm_vowel_reduction",
      "title": "string",
      "instruction": "string",
      "items": [
        {{"id": "i1", "text": "string", "choices": ["a", "b"], "correct": "a", "ipa": "/…/"}}
      ]
    }}
  ]
}}
"""
