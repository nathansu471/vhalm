# src/model2/run_all.py
"""
Usage (PowerShell):
  $env:OPENAI_API_KEY = "<your_key_here>"
  python ./src/model2/run_all.py

Customize by editing the constants below or via env vars:
  RUN_INPUT, RUN_LIMIT, RUN_N_SAMPLES, RUN_MODEL_GEN, RUN_MODEL_JUDGE, RUN_RANDOM, RUN_SEED
"""

import os
import pathlib
from generate import main as gen_main

# --- Easy knobs (edit these or set env vars of the same names) ---
INPUT = os.getenv(
    "RUN_INPUT",
    r"src/model1/outputs/final_best/exp2_att2_tag6_l.3r1e-4_ws150_20251124_134150/test_results/final_model1_sample120.json",
)
LIMIT = int(os.getenv("RUN_LIMIT", "10"))          # how many groups to use
N_SAMPLES = int(os.getenv("RUN_N_SAMPLES", "1"))   # haiku candidates per group
MODEL_GEN = os.getenv("RUN_MODEL_GEN", "gpt-4o-mini")
MODEL_JUDGE = os.getenv("RUN_MODEL_JUDGE", "gpt-4o-mini")
RANDOM_SAMPLE = os.getenv("RUN_RANDOM", "1").lower() not in ("0", "false", "no")
SEED = int(os.getenv("RUN_SEED", "42"))

def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent.resolve()

def _resolve_input(p: str) -> pathlib.Path:
    path = pathlib.Path(p)
    if not path.is_absolute():
        path = (_repo_root() / path).resolve()
    return path

def main():
    input_path = _resolve_input(INPUT)
    if not input_path.exists():
        raise FileNotFoundError(f"[run_all] Input not found:\n{input_path}")

    effective_limit = None if LIMIT < 0 else LIMIT

    print("[run_all] Using input:", input_path)
    print(f"[run_all] limit={LIMIT}  n_samples={N_SAMPLES}  random_sample={RANDOM_SAMPLE}")
    print(f"[run_all] models -> gen: {MODEL_GEN} | judge: {MODEL_JUDGE}")
    print(f"[run_all] seed={SEED}")

    gen_main(
        model_gen=MODEL_GEN,
        model_judge=MODEL_JUDGE,
        n_samples=N_SAMPLES,
        limit=effective_limit,
        seed=SEED,
        input_path=str(input_path),
        random_sample=RANDOM_SAMPLE,
    )

if __name__ == "__main__":
    main()
