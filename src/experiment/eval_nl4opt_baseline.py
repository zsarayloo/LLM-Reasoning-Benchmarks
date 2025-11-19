import os
import re
import time
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client as OllamaClient


# ---------------------------
# 1. Utility functions
# ---------------------------

def load_nl4opt_sample(n_examples: int = 10, random_state: int = 0) -> pd.DataFrame:
    """
    Load a small, reproducible sample of NL4OPT.
    Uses the hf:// path via pandas, with no token required (public dataset).
    """
    path = "hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json"
    print(f"Loading NL4OPT subset from: {path}")

    df = pd.read_json(path, lines=True)
    # Ensure expected columns exist
    assert "en_question" in df.columns and "en_answer" in df.columns, \
        "Expected columns 'en_question' and 'en_answer' not found."

    df = df.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
    print(f"Loaded {len(df)} examples.")
    return df


def extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last numeric value from a model's text response.
    Returns None if nothing looks like a number.
    """
    if not text:
        return None
    # Capture integers, floats, and scientific notation
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_answer(pred: Optional[float], gt: float, tol: float = 1e-3) -> Dict[str, Any]:
    """
    Simple numeric verifier: compares predicted numeric answer to ground truth
    within a tolerance.
    """
    if pred is None:
        return {"correct": False, "abs_error": None, "rel_error": None}

    abs_err = abs(pred - gt)
    rel_err = abs_err / (abs(gt) + 1e-9)
    return {
        "correct": abs_err <= tol,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }


# ---------------------------
# 2. Prompt templates
# ---------------------------

def build_prompt(question: str, mode: str) -> str:
    """
    mode ∈ {"direct", "cot"}
    """
    base_instruction = (
        "You are an expert operations researcher and optimization assistant. "
        "You must solve the following word problem that describes a linear program "
        "and return the optimal objective value.\n\n"
    )

    if mode == "direct":
        suffix = (
            "Problem:\n"
            f"{question}\n\n"
            "Return only the optimal objective value as a single number. "
            "Do not include any explanation or extra text."
        )
    elif mode == "cot":
        suffix = (
            "Problem:\n"
            f"{question}\n\n"
            "First, think step by step and explain your reasoning briefly. "
            "Then, on the last line, output the final optimal objective value in the form:\n"
            "ANSWER: <number>\n"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return base_instruction + suffix


# ---------------------------
# 3. Model caller wrappers
# ---------------------------

class ModelCaller:
    def __init__(self):
        load_dotenv()
        # OpenAI client
        self.openai_client = OpenAI()
        # Ollama client
        self.ollama_client = OllamaClient()

    def call_openai(self, model: str, prompt: str) -> str:
        response = self.openai_client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output[0].content[0].text

    def call_ollama(self, model: str, prompt: str) -> str:
        response = self.ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# ---------------------------
# 4. Evaluation loop
# ---------------------------

def evaluate_models_on_nl4opt(
    n_examples: int = 10,
    openai_models: Optional[List[str]] = None,
    ollama_models: Optional[List[str]] = None,
    modes: Optional[List[str]] = None,
    tol: float = 1e-3,
):
    if openai_models is None:
        openai_models = ["gpt-4.1-mini", "gpt-4.1", "gpt-5.1"]
    if ollama_models is None:
        ollama_models = ["llama3.2:3b"]
    if modes is None:
        modes = ["direct", "cot"]

    df = load_nl4opt_sample(n_examples=n_examples)
    caller = ModelCaller()

    results: List[Dict[str, Any]] = []

    total_jobs = len(df) * (len(openai_models) + len(ollama_models)) * len(modes)
    print(f"\nTotal evaluations to run: {total_jobs}\n")

    job_idx = 0

    for i, row in df.iterrows():
        question = row["en_question"]
        gt = float(row["en_answer"])

        print(f"\n=== Example {i+1}/{len(df)} ===")
        print("Question (truncated to 200 chars):")
        print(question[:200] + ("..." if len(question) > 200 else ""))
        print("Ground truth objective value:", gt)

        for mode in modes:
            prompt = build_prompt(question, mode)

            # 4.1. OpenAI models
            for m in openai_models:
                job_idx += 1
                print(f"\n[{job_idx}/{total_jobs}] OpenAI model={m}, mode={mode}")
                start = time.time()
                try:
                    response_text = caller.call_openai(m, prompt)
                except Exception as e:
                    print("Error calling OpenAI model:", repr(e))
                    results.append({
                        "example_idx": i,
                        "model": m,
                        "backend": "openai",
                        "mode": mode,
                        "gt": gt,
                        "raw_response": None,
                        "parsed_answer": None,
                        "correct": False,
                        "abs_error": None,
                        "rel_error": None,
                        "latency_sec": None,
                        "error": repr(e),
                    })
                    continue

                latency = time.time() - start
                pred_num = extract_last_number(response_text)
                verif = verify_answer(pred_num, gt, tol=tol)

                print("Response (truncated to 200 chars):")
                print(response_text[:200].replace("\n", " ") + ("..." if len(response_text) > 200 else ""))
                print(f"Parsed answer: {pred_num}, correct={verif['correct']}, abs_err={verif['abs_error']}")

                results.append({
                    "example_idx": i,
                    "model": m,
                    "backend": "openai",
                    "mode": mode,
                    "gt": gt,
                    "raw_response": response_text,
                    "parsed_answer": pred_num,
                    "correct": verif["correct"],
                    "abs_error": verif["abs_error"],
                    "rel_error": verif["rel_error"],
                    "latency_sec": latency,
                    "error": None,
                })

            # 4.2. Ollama models
            for m in ollama_models:
                job_idx += 1
                print(f"\n[{job_idx}/{total_jobs}] Ollama model={m}, mode={mode}")
                start = time.time()
                try:
                    response_text = caller.call_ollama(m, prompt)
                except Exception as e:
                    print("Error calling Ollama model:", repr(e))
                    results.append({
                        "example_idx": i,
                        "model": m,
                        "backend": "ollama",
                        "mode": mode,
                        "gt": gt,
                        "raw_response": None,
                        "parsed_answer": None,
                        "correct": False,
                        "abs_error": None,
                        "rel_error": None,
                        "latency_sec": None,
                        "error": repr(e),
                    })
                    continue

                latency = time.time() - start
                pred_num = extract_last_number(response_text)
                verif = verify_answer(pred_num, gt, tol=tol)

                print("Response (truncated to 200 chars):")
                print(response_text[:200].replace("\n", " ") + ("..." if len(response_text) > 200 else ""))
                print(f"Parsed answer: {pred_num}, correct={verif['correct']}, abs_err={verif['abs_error']}")

                results.append({
                    "example_idx": i,
                    "model": m,
                    "backend": "ollama",
                    "mode": mode,
                    "gt": gt,
                    "raw_response": response_text,
                    "parsed_answer": pred_num,
                    "correct": verif["correct"],
                    "abs_error": verif["abs_error"],
                    "rel_error": verif["rel_error"],
                    "latency_sec": latency,
                    "error": None,
                })

    # Convert to DataFrame for summary
    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    out_path = "results/nl4opt_baseline_results.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary table
    summary = (
        res_df.groupby(["backend", "model", "mode"])
        .agg(
            n=("correct", "size"),
            n_ok=("correct", "sum"),
            acc=("correct", "mean"),
            mean_abs_err=("abs_error", "mean"),
            mean_rel_err=("rel_error", "mean"),
            mean_latency=("latency_sec", "mean"),
        )
        .reset_index()
    )

    print("\n=== Summary (per backend, model, mode) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    # You can tweak n_examples if you want fewer/more
    evaluate_models_on_nl4opt(n_examples=10)
