SYSTEM_HAIKU = """You are a concise poet. You transform a plain English scene description into a vivid haiku.
Follow EXACTLY the 5-7-5 syllable structure. Use natural imagery, avoid lists, avoid naming each image.
Return ONLY three lines (no quotes, no extra words)."""

USER_HAIKU = """Write a haiku from the description below.

Constraints:
- 3 lines in 5-7-5 syllables.
- Evocative, concrete imagery; no enumerations.
- Avoid repetition; avoid listing each image separately.
- No hashtags, no title, no author.

Description:
{desc}"""

SYSTEM_MERGE = """You are a careful technical writer who fuses multiple captions about related images
into ONE objective, coherent description (≤50 tokens). Avoid repetition and avoid listing each image separately."""

USER_MERGE = """Fuse the following short captions into ONE objective description (<= 50 tokens).
Avoid repetition and avoid listing each image separately.

Captions:
{caps}"""

SYSTEM_JUDGE = """You are an impartial grader for haiku quality and constraint satisfaction.
Return a strict JSON object with numeric scores and a brief rationale."""
USER_JUDGE = """Evaluate the candidate haiku against the original description.

Description:
{desc}

Haiku:
{haiku}

Scoring (0-5 each, integers):
- relevance: how well the haiku reflects the description's key content
- structure: 5-7-5 lines exactly, strong line breaks (deduct if broken)
- imagery: vividness and concreteness of language
- fluency: grammar and naturalness

Return JSON ONLY:
{{"relevance": X, "structure": Y, "imagery": Z, "fluency": W, "total": X+Y+Z+W, "rationale": "..."}}"""
