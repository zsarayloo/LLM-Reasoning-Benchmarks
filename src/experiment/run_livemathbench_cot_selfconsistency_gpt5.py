import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from data_loader_livemathbench import load_livemathbench


# ============================================================
# 1. GPT-5.1 caller with basic rate-limit retry
# ============================================================

class GPT5Caller:
    def __init__(
        self,
        model_name: str = "gpt-5.1",
        max_retries: int = 5,
        base_sleep: float = 1.0,
    ):
        load_dotenv()
        self.client = OpenAI()
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_sleep = base_sleep

    def call(self, prompt: str, temperature: float = 0.6) -> str:
        """
        Simple wrapper with exponential backoff on 429 (rate limit).
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    temperature=temperature,
                )
                return resp.output[0].content[0].text
            except Exception as e:
                msg = str(e)
                last_error = e
                if "429" in msg or "Rate limit" in msg or "insufficient_quota" in msg:
                    sleep_t = self.base_sleep * (2 ** (attempt - 1))
                    print(
                        f"[GPT5Caller] Rate/Quota issue on model={self.model_name}, "
                        f"attempt {attempt}/{self.max_retries}. Sleeping {sleep_t:.1f} seconds..."
                    )
                    time.sleep(sleep_t)
                    continue
                else:
                    print("[GPT5Caller] Non-retryable error:", repr(e))
                    raise

        print("[GPT5Caller] Exhausted retries; raising last error.")
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unknown error in GPT5Caller.call")


# ============================================================
# 2. CoT prompt for general math
# ============================================================

def build_cot_prompt_math(question: str) -> str:
    """
    Standard Chain-of-Thought prompt for math problems.
    We will ask for a final answer explicitly marked as 'ANSWER: ...'.
    """
    return (
        "You are an expert competition mathematician.\n"
        "Solve the following math problem step by step.\n"
        "Show your reasoning clearly.\n"
        "At the END, on a separate last line, output the final answer in the form:\n"
        "ANSWER: <final_answer>\n\n"
        "Math problem:\n"
        f"{question}\n"
    )


# ============================================================
# 3. Answer normalization & verification (same style as PoT script)
# ============================================================

def _try_parse_float(s: str) -> Optional[float]:
    """
    Try to parse a string as float. Return None on failure.
    """
    try:
        s_norm = (
            s.replace("$", "")
             .replace("\\(", "")
             .replace("\\)", "")
             .strip()
        )
        return float(s_norm)
    except Exception:
        return None


def normalize_answer_str(ans: Any) -> str:
    """
    Normalize an answer (string / number) into a canonical string form.
    - If numeric: format with fixed precision.
    - If string: strip LaTeX delimiters and whitespace, lowercased.
    """
    from fractions import Fraction

    if isinstance(ans, Fraction):
        return f"{ans.numerator}/{ans.denominator}"

    if isinstance(ans, (int, float)):
        return ("{:.12g}".format(float(ans))).strip()

    s = str(ans)
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def verify_math_answer(pred: Any, gold: str, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Compare model prediction (pred) to gold string (from LiveMathBench).
    We check:
      1) numeric closeness if both parse as floats.
      2) string equality of normalized forms.
    """
    if pred is None:
        return {
            "correct": False,
            "numeric_match": False,
            "string_match": False,
            "pred_norm": None,
            "gold_norm": normalize_answer_str(gold),
        }

    gold_float = _try_parse_float(gold)
    if isinstance(pred, (int, float)):
        pred_float = float(pred)
    else:
        pred_float = _try_parse_float(str(pred))

    numeric_match = False
    if gold_float is not None and pred_float is not None:
        if abs(pred_float - gold_float) <= tol * max(1.0, abs(gold_float)):
            numeric_match = True

    pred_norm = normalize_answer_str(pred)
    gold_norm = normalize_answer_str(gold)
    string_match = (pred_norm == gold_norm)

    return {
        "correct": numeric_match or string_match,
        "numeric_match": numeric_match,
        "string_match": string_match,
        "pred_norm": pred_norm,
        "gold_norm": gold_norm,
    }


# ============================================================
# 4. Extract final answer from CoT text
# ============================================================

def extract_answer_from_cot(text: str) -> Optional[str]:
    """
    Look for a line starting with 'ANSWER:' and return what's after it.
    If not found, fall back to last non-empty line.
    """
    if not text:
        return None

    lines = text.strip().splitlines()
    # First try explicit 'ANSWER:' token
    for line in reversed(lines):
        m = re.search(r"ANSWER\s*:\s*(.+)", line, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: last non-empty line
    for line in reversed(lines):
        ls = line.strip()
        if ls:
            return ls
    return None


# ============================================================
# 5. Self-Consistency voting
# ============================================================

def majority_vote_normalized(answers_norm: List[str]) -> Tuple[Optional[str], int, bool]:
    """
    Compute majority vote over normalized answers.

    Returns:
        - winning_answer_norm: normalized answer with highest count (or None if list empty)
        - count: vote count for winning answer
        - tie: True if there was a tie for top count
    """
    if not answers_norm:
        return None, 0, False

    counts: Dict[str, int] = {}
    for a in answers_norm:
        if a is None:
            continue
        counts[a] = counts.get(a, 0) + 1

    if not counts:
        return None, 0, False

    # Find max count
    max_count = max(counts.values())
    winners = [a for a, c in counts.items() if c == max_count]
    tie = len(winners) > 1
    # Break ties arbitrarily by choosing first
    winning_answer = winners[0]
    return winning_answer, max_count, tie


# ============================================================
# 6. Main evaluation: LiveMathBench + CoT Self-Consistency
# ============================================================

def run_livemathbench_cot_selfconsistency(
    split: str = "all",
    n_examples: Optional[int] = None,
    random_state: int = 0,
    n_samples: int = 10,
    tol: float = 1e-6,
    local_dir: str = "data/LiveMathBench",
) -> None:
    """
    Evaluate GPT-5.1 on LiveMathBench with:
      - Chain-of-Thought
      - Self-Consistency (n_samples independent CoTs per question)

    The data loader is assumed to be:
      load_livemathbench(split="all", local_dir="data/LiveMathBench", ...)

    It should return a DataFrame with columns at least:
      - "question"
      - "answer"
      - optionally "split" (AMC, CCEE, CNMO, WLPMC, hard)
    """
    df = load_livemathbench(
        split=split,
        local_dir=local_dir,
        n_examples=n_examples,
        random_state=random_state,
    )

    assert "question" in df.columns and "answer" in df.columns, \
        f"DataFrame must contain 'question' and 'answer' columns, but has: {df.columns.tolist()}"

    caller = GPT5Caller(model_name="gpt-5.1")

    results: List[Dict[str, Any]] = []
    total = len(df)

    print(
        f"\n=== Evaluating LiveMathBench (split='{split}') with "
        f"CoT + Self-Consistency (k={n_samples}), n={total} ===\n"
    )

    for i, row in df.iterrows():
        q = row["question"]
        gold = row["answer"]
        split_name = row.get("split", split)

        print(f"\n=== Example {i+1}/{total} (split={split_name}) ===")
        print(q[:200] + ("..." if len(q) > 200 else ""))
        print("Gold answer (raw):", gold)

        sample_raw_texts: List[str] = []
        sample_answers_raw: List[Any] = []
        sample_answers_norm: List[Optional[str]] = []
        sample_correct_flags: List[bool] = []
        sample_numeric_match: List[bool] = []
        sample_string_match: List[bool] = []
        sample_latencies: List[float] = []
        sample_errors: List[Optional[str]] = []

        for s in range(n_samples):
            prompt = build_cot_prompt_math(q)
            t0 = time.time()
            try:
                raw_text = caller.call(prompt, temperature=0.7)
                latency = time.time() - t0

                ans_str = extract_answer_from_cot(raw_text)
                # For now we treat the answer as a string; verification handles numeric parsing.
                pred_val: Any = ans_str

                verif = verify_math_answer(pred_val, gold, tol=tol)
                sample_raw_texts.append(raw_text)
                sample_answers_raw.append(pred_val)
                sample_answers_norm.append(verif["pred_norm"])
                sample_correct_flags.append(verif["correct"])
                sample_numeric_match.append(verif["numeric_match"])
                sample_string_match.append(verif["string_match"])
                sample_latencies.append(latency)
                sample_errors.append(None)

                print(
                    f"  [sample {s+1}/{n_samples}] "
                    f"pred={pred_val!r}, correct={verif['correct']}, "
                    f"numeric={verif['numeric_match']}, string={verif['string_match']}, "
                    f"latency={latency:.2f}s"
                )
            except Exception as e:
                latency = time.time() - t0
                print(f"  [sample {s+1}/{n_samples}] ERROR:", repr(e))
                sample_raw_texts.append(None)
                sample_answers_raw.append(None)
                sample_answers_norm.append(None)
                sample_correct_flags.append(False)
                sample_numeric_match.append(False)
                sample_string_match.append(False)
                sample_latencies.append(latency)
                sample_errors.append(repr(e))

        # Self-consistency: majority vote on normalized answers
        winning_norm, vote_count, tie = majority_vote_normalized(sample_answers_norm)
        if winning_norm is None:
            sc_correct = False
            sc_numeric = False
            sc_string = False
        else:
            # Evaluate winning answer vs gold
            # We can treat winning_norm as a string and run verify again
            sc_verif = verify_math_answer(winning_norm, gold, tol=tol)
            sc_correct = sc_verif["correct"]
            sc_numeric = sc_verif["numeric_match"]
            sc_string = sc_verif["string_match"]

        # Stats for this question
        any_correct = any(sample_correct_flags)
        mean_latency = sum(sample_latencies) / max(1, len(sample_latencies))

        print(
            f"  -> Self-Consistency winner={winning_norm!r}, votes={vote_count}, tie={tie}, "
            f"sc_correct={sc_correct}, any_sample_correct={any_correct}, "
            f"mean_latency={mean_latency:.2f}s"
        )

        results.append({
            "idx": i,
            "split": split_name,
            "question": q,
            "gold_answer": gold,
            "n_samples": n_samples,
            "sample_raw_texts_json": json.dumps(sample_raw_texts),
            "sample_answers_raw_json": json.dumps(sample_answers_raw),
            "sample_answers_norm_json": json.dumps(sample_answers_norm),
            "sample_correct_flags_json": json.dumps(sample_correct_flags),
            "sample_numeric_match_json": json.dumps(sample_numeric_match),
            "sample_string_match_json": json.dumps(sample_string_match),
            "sample_latencies_json": json.dumps(sample_latencies),
            "sample_errors_json": json.dumps(sample_errors),
            "any_sample_correct": any_correct,
            "sc_winning_norm": winning_norm,
            "sc_vote_count": vote_count,
            "sc_tie": tie,
            "sc_correct": sc_correct,
            "sc_numeric_match": sc_numeric,
            "sc_string_match": sc_string,
            "mean_latency_sec": mean_latency,
        })

    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/livemathbench_{split}_gpt5_cot_selfconsistency.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary (official metric = self-consistency correctness)
    n = len(res_df)
    n_ok_sc = int(res_df["sc_correct"].sum())
    acc_sc = n_ok_sc / n if n > 0 else 0.0

    n_ok_any = int(res_df["any_sample_correct"].sum())
    acc_any = n_ok_any / n if n > 0 else 0.0

    mean_latency = res_df["mean_latency_sec"].mean()

    print("\n=== Summary (LiveMathBench, CoT + Self-Consistency) ===")
    print(f"split             = {split}")
    print(f"n                 = {n}")
    print(f"n_ok_sc           = {n_ok_sc}   (self-consistency)")
    print(f"acc_sc            = {acc_sc}")
    print(f"n_ok_any_sample   = {n_ok_any}  (at least one correct sample)")
    print(f"acc_any_sample    = {acc_any}")
    print(f"mean_latency_sec  = {mean_latency}")


if __name__ == "__main__":
    # You can start with split="all" and n_examples=20–50 to test.
    # Then set n_examples=None for full evaluation.
    run_livemathbench_cot_selfconsistency(
        split="all",           # "all", "AMC_en", "CCEE_en", "CNMO_en", "WLPMC_en", "hard_en"
        n_examples=None,       # None -> full; or an int for a sample
        random_state=0,
        n_samples=10,          # k in Self-Consistency
        tol=1e-6,
        local_dir="data/LiveMathBench",
    )
