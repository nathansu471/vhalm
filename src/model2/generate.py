import os
import json
import time
import pathlib
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm
import pandas as pd
import numpy as np

from prompts import SYSTEM_HAIKU, USER_HAIKU, SYSTEM_MERGE, USER_MERGE
from structure import ensure_575
from judge import grade_one, rerank

# ================== OpenAI clients (sync + async) ==================
from openai import OpenAI, AsyncOpenAI

API_KEY = "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"
BASE_URL = os.environ.get("OPENAI_BASE_URL")


def _get_client() -> OpenAI:
    """
    Creates an OpenAI client:
    - If MOCK_LLM is set, a dummy key is used and no network calls are made.
    - If OPENAI_BASE_URL is set, sends requests to that server (e.g., local vLLM/Ollama).
    - Otherwise uses OPENAI_API_KEY from the environment.
    """
    if os.environ.get("MOCK_LLM"):
        return OpenAI(api_key="mock-key", base_url=BASE_URL)
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (or set MOCK_LLM=1 for a dry run).")
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


_client = _get_client()

# Async client (for any concurrent tooling you might use elsewhere)
client = AsyncOpenAI(
    api_key=("mock-key" if os.environ.get("MOCK_LLM") else API_KEY),
    base_url=BASE_URL,
)

MODEL_NAME = "gpt-4o"
CONCURRENCY = 10
SAVE_INTERVAL = 50
TMP_FILE = "merged_captions_tmp.json"
OUT_FILE = "merged_captions_async3.json"


def _chat(messages: List[dict], model: str = "gpt-4o-mini", retries: int = 5) -> str:
    """
    Chat wrapper with:
    - MOCK_LLM (no network)
    - retry/backoff
    - friendly error on insufficient quota
    """
    if os.environ.get("MOCK_LLM"):
        user = messages[-1]["content"].lower() if messages else ""
        if "haiku" in user:
            return "frost on empty rails\na stray cat counts slow footfalls\nnight trains breathe and fade"
        return "A single concise description of the shared scene."

    last_err = None
    for i in range(retries):
        try:
            resp = _client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                raise RuntimeError(
                    "LLM call failed: insufficient quota/billing. "
                    "Rotate your key if leaked, attach credits to the correct Project, or set OPENAI_BASE_URL to a local endpoint."
                )
            time.sleep(min(60, 2 ** i))  # backoff for transient rate limits/timeouts
            last_err = e
    raise last_err


# ================== Input loading ==================
LIKELY_SINGLE_DESC_KEYS = [
    "caption", "merged_caption", "merged", "description", "desc",
    "final_caption", "one_caption", "prompt", "y"
]

LIKELY_MULTI_CAPTIONS_KEYS = [
    "captions", "all_captions", "retained_captions", "caption_list",
    "top_captions", "topk_captions", "group_captions", "image_captions",
    "captions_kept", "captions_sorted"
]


def _first_existing(paths: List[pathlib.Path]) -> Optional[pathlib.Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _maybe_image_ids(it: Dict[str, Any]) -> Optional[List[str]]:
    # Explicit list
    if isinstance(it.get("image_ids"), list) and all(isinstance(x, (str, int)) for x in it["image_ids"]):
        return [str(x) for x in it["image_ids"]]
    # Triplet keys
    if all(k in it for k in ("img_idA", "img_idB", "img_idC")):
        return [str(it["img_idA"]), str(it["img_idB"]), str(it["img_idC"])]
    # Generic: any list of 3 under a key that looks like images/ids
    for k, v in it.items():
        if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (str, int)) for x in v):
            lk = k.lower()
            if "image" in lk or "img" in lk or "id" in lk:
                return [str(x) for x in v]
    return None


def _find_single_description(it: Dict[str, Any]) -> Optional[str]:
    for k in LIKELY_SINGLE_DESC_KEYS:
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback: any single string that looks like a caption/desc
    for k, v in it.items():
        if isinstance(v, str) and v.strip():
            lk = k.lower()
            if any(s in lk for s in ["caption", "merged", "desc", "description", "text", "prompt"]):
                return v.strip()
    return None


def _is_list_of_strings(x: Any) -> bool:
    return isinstance(x, list) and len(x) > 0 and all(isinstance(t, str) and t.strip() for t in x)


def _find_multi_captions(it: Dict[str, Any]) -> Optional[List[str]]:
    for k in LIKELY_MULTI_CAPTIONS_KEYS:
        v = it.get(k)
        if _is_list_of_strings(v):
            return [s.strip() for s in v]
    # generic scan
    for k, v in it.items():
        if _is_list_of_strings(v):
            lk = k.lower()
            if any(s in lk for s in ["caption", "captions", "desc", "description", "text"]):
                return [s.strip() for s in v]
    return None


def _canon_item(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "group_id": it.get("group_id", it.get("id", -1)),
        "image_ids": _maybe_image_ids(it),
        "merged_caption": _find_single_description(it),
        "captions": _find_multi_captions(it),
        "_orig_keys": list(it.keys()),
    }


def load_items(input_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Loads items for Model 2.

    If input_path is provided, use that file (absolute or relative to repo root).
    Otherwise, searches common repo paths (stage-1 outputs, etc).

    Supports:
      - A raw list[dict] (legacy)
      - {"data": [...]} (legacy)
      - {"samples": [...]} (Model-1 test outputs with prediction/reference)
    """
    base = pathlib.Path(__file__).parent.resolve()    # .../src/model2
    repo_root = base.parent.parent                    # .../

    if input_path:
        p = pathlib.Path(input_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
    else:
        candidates = [
            repo_root / "data_combine" / "merged_captions_async.json",
            repo_root / "data_combine" / "result" / "merged_captions_async_augmented.json",
            repo_root / "image_groups_curriulum.json",
            repo_root / "image_groups_curriculum.json",
            repo_root / "clip_453_ysy" / "image_groups_with_captions.json",
            repo_root / "clip_453_ysy" / "image_groups_fixed3.json",
        ]
        p = _first_existing(candidates)
        if not p:
            raise FileNotFoundError(
                "No input JSON found. Looked for:\n  " + "\n  ".join(map(str, candidates))
            )

    print(f"[Model2] Using input file: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))

    # Handle Model-1 schema: {"samples": [...]}
    if isinstance(raw, dict) and isinstance(raw.get("samples"), list):
        items = []
        for s in raw["samples"]:
            desc = (s.get("prediction") or s.get("reference") or "").strip()
            image_ids = s.get("image_ids")
            if isinstance(image_ids, list):
                image_ids = [str(x) for x in image_ids]
            else:
                image_ids = None
            items.append({
                "group_id": s.get("group_id", s.get("id", -1)),
                "image_ids": image_ids,
                "merged_caption": desc,  # what Model-2 will prompt from
                "captions": None,
                "_orig_keys": list(s.keys()),
            })
        return items

    # Legacy formats:
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        raw = raw["data"]
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of items in {p}, got type {type(raw)}")
    return [_canon_item(it) for it in raw]


# ================== LLM helpers ==================
def llm_merge(caps: List[str], model: str = "gpt-4o-mini") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MERGE},
        {"role": "user", "content": USER_MERGE.format(caps="\n".join(f"- {c}" for c in caps))},
    ]
    return _chat(messages, model=model).strip()


def llm_haiku(desc: str, model: str = "gpt-4o-mini") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_HAIKU},
        {"role": "user", "content": USER_HAIKU.format(desc=desc)},
    ]
    return _chat(messages, model=model).strip()


def _ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main(
    model_gen: str = "gpt-4o-mini",
    model_judge: str = "gpt-4o-mini",
    n_samples: int = 4,
    limit: Optional[int] = None,
    seed: int = 42,
    input_path: Optional[str] = None,
    random_sample: bool = False,
) -> None:
    """
    Stage-2 pipeline (Prompt + Judge):
      - load Stage-1 unified descriptions, or merge 3→1 by prompt
      - generate k haiku candidates per group
      - enforce/check 5-7-5
      - judge each candidate, keep the best
      - write artifacts to src/model2/outputs/<timestamp>/
    """
    random.seed(seed)
    items = load_items(input_path)

    if limit is not None:
        if random_sample:
            k = min(limit, len(items))
            items = random.sample(items, k)
        else:
            items = items[:limit]

    base_dir = pathlib.Path(__file__).parent
    run_dir = base_dir / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(run_dir)

    samples_fp = (run_dir / "samples.jsonl").open("w", encoding="utf-8", newline="\n")
    judge_fp   = (run_dir / "judge_logs.jsonl").open("w", encoding="utf-8", newline="\n")

    # bookkeeping
    n_total = len(items)
    n_used_merged = 0
    n_merged_from_list = 0
    n_skipped_no_text = 0
    key_hist: Dict[str, int] = {}

    rows = []
    for it in tqdm(items, desc="Model2 prompt+judge"):
        for k in it.get("_orig_keys", []):
            key_hist[k] = key_hist.get(k, 0) + 1

        # Get unified description
        desc = it.get("merged_caption")
        if desc and desc.strip():
            n_used_merged += 1
        else:
            caps = it.get("captions") or []
            if not caps:
                # last-ditch: pick any meaningful string field
                candidates = []
                for k in it.get("_orig_keys", []):
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        lk = k.lower()
                        if any(s in lk for s in ["caption", "desc", "description", "text", "prompt", "y"]):
                            candidates.append(v.strip())
                if candidates:
                    desc = max(candidates, key=len)
                    n_used_merged += 1
                else:
                    n_skipped_no_text += 1
                    continue
            else:
                desc = llm_merge(caps, model=model_gen)
                n_merged_from_list += 1

        # Generate candidates and judge
        candidates = []
        for _ in range(n_samples):
            raw = llm_haiku(desc, model=model_gen)
            lines, syl, ok = ensure_575(raw)
            cand = {"haiku": lines, "syllables": syl, "structure_ok": ok}
            j = grade_one(desc, lines, model=model_judge)
            cand["judge"] = j
            judge_fp.write(json.dumps({
                "group_id": it["group_id"],
                "desc": desc,
                "haiku": lines,
                "syllables": syl,
                "judge": j
            }) + "\n")
            candidates.append(cand)

        best = rerank(candidates)
        rec = {
            "group_id": it["group_id"],
            "image_ids": it.get("image_ids"),
            "desc": desc,
            "best_haiku": best["haiku"],
            "best_syllables": best["syllables"],
            "best_judge": best["judge"],
            "all": candidates,
        }
        samples_fp.write(json.dumps(rec) + "\n")
        rows.append(rec)

    samples_fp.close()
    judge_fp.close()

    # Tabular summary (one row per group, winner only)
    if rows:
        df = pd.DataFrame([{
            "group_id": r["group_id"],
            "image_ids": "|".join(r["image_ids"]) if r.get("image_ids") else "",
            "desc": r["desc"],
            "haiku_l1": r["best_haiku"][0],
            "haiku_l2": r["best_haiku"][1],
            "haiku_l3": r["best_haiku"][2],
            "syllables": "/".join(map(str, r["best_syllables"])),
            "total": r["best_judge"].get("total", 0),
            "relevance": r["best_judge"].get("relevance", 0),
            "structure": r["best_judge"].get("structure", 0),
            "imagery": r["best_judge"].get("imagery", 0),
            "fluency": r["best_judge"].get("fluency", 0),
        } for r in rows])
        df.to_csv(run_dir / "best.csv", index=False)
    else:
        pd.DataFrame(columns=[
            "group_id","image_ids","desc","haiku_l1","haiku_l2","haiku_l3",
            "syllables","total","relevance","structure","imagery","fluency"
        ]).to_csv(run_dir / "best.csv", index=False)

    # JSONL with just the winner per group
    with (run_dir / "best_haiku.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "group_id": r["group_id"],
                "image_ids": r.get("image_ids"),
                "desc": r["desc"],
                "haiku": r["best_haiku"],          # [l1, l2, l3]
                "syllables": r["best_syllables"],
                "judge": r["best_judge"],
            }) + "\n")

    # Human-friendly text file
    with (run_dir / "best_haiku.txt").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(f"[group_id: {r['group_id']}]\n")
            fh.write("\n".join(r["best_haiku"]) + "\n\n")

    # Metrics + run info (for debugging schema/coverage)
    metrics = {
        "num_items_loaded": n_total,
        "num_results_written": len(rows),
        "used_merged_caption": n_used_merged,
        "merged_from_captions_list": n_merged_from_list,
        "skipped_no_text": n_skipped_no_text,
        "structure_ok_>=4": int(np.sum([r["best_judge"].get("structure", 0) >= 4 for r in rows])),
        "avg_total": float(np.mean([r["best_judge"].get("total", 0) for r in rows] or [0])),
        "avg_relevance": float(np.mean([r["best_judge"].get("relevance", 0) for r in rows] or [0])),
        "mock_mode": bool(os.environ.get("MOCK_LLM")),
        "model_gen": model_gen,
        "model_judge": model_judge,
    }
    _ensure_dir(run_dir)  # defensive
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # key histogram (top 40)
    sorted_keys = sorted(key_hist.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
    _ensure_dir(run_dir)  # defensive
    (run_dir / "run_info.json").write_text(json.dumps({
        "input_key_hist_top": sorted_keys,
        "metrics_path": "metrics.json",
        "best_csv_path": "best.csv",
        "notes": "If results are empty, check skipped_no_text and the key histogram to see which fields to map."
    }, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to: {run_dir}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Model2: prompt + judge")
    ap.add_argument("--model-gen", default="gpt-4o-mini")
    ap.add_argument("--model-judge", default="gpt-4o-mini")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--desc", type=str, default=None,
                    help="If set, bypass data files and generate for this one description only.")
    ap.add_argument("--input", type=str, default=None,
                    help="Path to a JSON file (e.g., Model 1 results). Relative paths are resolved from repo root.")
    ap.add_argument("--random-sample", action="store_true",
                    help="Randomly sample --limit items instead of taking the first N.")
    args = ap.parse_args()

    if args.desc:
        candidates = []
        for _ in range(args.n_samples):
            raw = llm_haiku(args.desc, model=args.model_gen)
            lines, syl, ok = ensure_575(raw)
            j = grade_one(args.desc, lines, model=args.model_judge)
            candidates.append({"haiku": lines, "syllables": syl, "structure_ok": ok, "judge": j})
        best = rerank(candidates)
        print("\n".join(best["haiku"]))
        print(f"\nSyllables per line: {best['syllables']} | total: {best['judge'].get('total', 0)}")
    else:
        main(
            model_gen=args.model_gen,
            model_judge=args.model_judge,
            n_samples=args.n_samples,
            limit=args.limit,
            seed=args.seed,
            input_path=args.input,
            random_sample=args.random_sample,
        )
