import json
import os
import time
from typing import Dict, Any, List

from openai import OpenAI
from prompts import SYSTEM_JUDGE, USER_JUDGE

API_KEY = "api key"
BASE_URL = os.environ.get("OPENAI_BASE_URL")

def _get_client() -> OpenAI:
    """
    Creates an OpenAI client:
    - If MOCK_LLM is set, a dummy key is used and no network calls are made.
    - If OPENAI_BASE_URL is set, sends requests to that server (e.g., local vLLM/Ollama).
    - Otherwise uses the API_KEY constant above.
    """
    if os.environ.get("MOCK_LLM"):
        return OpenAI(api_key="mock-key", base_url=BASE_URL)
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

_client = _get_client()

def _chat(messages, model: str = "gpt-4o-mini", retries: int = 5) -> str:
    if os.environ.get("MOCK_LLM"):
        # optimistic fixed JSON
        return '{"relevance":4,"structure":4,"imagery":4,"fluency":4,"total":16,"rationale":"mock"}'
    last_err = None
    for i in range(retries):
        try:
            r = _client.chat.completions.create(model=model, messages=messages)
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                raise RuntimeError("Judge LLM failed: insufficient quota/billing or wrong config.")
            time.sleep(min(60, 2 ** i))  # backoff
            last_err = e
    raise last_err

def grade_one(desc: str, haiku: List[str], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    msg = [
        {"role": "system", "content": SYSTEM_JUDGE},
        {"role": "user",   "content": USER_JUDGE.format(desc=desc, haiku="\n".join(haiku))},
    ]
    raw = _chat(msg, model=model)
    try:
        obj = json.loads(raw)
    except Exception:
        obj = {
            "relevance": 0, "structure": 0, "imagery": 0, "fluency": 0, "total": 0,
            "rationale": "parse_error", "_raw": raw
        }
    obj["model"] = model
    return obj

def rerank(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Pick highest total; tie-break by structure then imagery
    def key(c):
        j = c["judge"]
        return (j.get("total", 0), j.get("structure", 0), j.get("imagery", 0))
    return sorted(candidates, key=key, reverse=True)[0]
