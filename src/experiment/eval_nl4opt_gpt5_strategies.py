import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ==========================
# 1. Dataset utilities
# ==========================
print(">>> eval_nl4opt_gpt5_strategies.py imported, __name__ =", __name__)


def load_nl4opt(n_examples: Optional[int] = None, random_state: int = 0) -> pd.DataFrame:
    """
    Load NL4OPT_with_optimal_solution.json from Hugging Face (via hf://).
    If n_examples is None → use FULL dataset.
    """
    path = "hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json"
    print(f"Loading NL4OPT from: {path}")
    df = pd.read_json(path, lines=True)

    assert "en_question" in df.columns and "en_answer" in df.columns, \
        "Expected columns 'en_question' and 'en_answer' not found."

    if n_examples is not None:
        df = df.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
        print(f"Loaded {len(df)} examples (sample).")
    else:
        df = df.reset_index(drop=True)
        print(f"Loaded FULL dataset: {len(df)} examples.")

    return df


def extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last numeric value from text.
    Used for parsing the model's final answer.
    """
    if not text:
        return None
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def verify_answer_numeric(pred: Optional[float], gt: float, tol: float = 1e-3) -> Dict[str, Any]:
    """
    Simple numeric verifier: compares a predicted number to the ground-truth
    objective value (for evaluation only; NOT leaked back into the model).
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


# ==========================
# 2. GPT-5.1 caller
# ==========================

class GPT5Caller:
    def __init__(self, model_name: str = "gpt-5.1"):
        load_dotenv()
        self.client = OpenAI()
        self.model_name = model_name

    def call(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Low-level wrapper to call gpt-5.1 with a plain text prompt.
        """
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output[0].content[0].text


# ==========================
# 3. Prompt builders
# ==========================

def build_direct_prompt(question: str) -> str:
    return (
        "You are an expert operations researcher.\n"
        "Solve the following optimization word problem.\n"
        "Return ONLY the optimal objective value as a single number.\n"
        "Do NOT include units, explanation, or extra text.\n\n"
        "Problem:\n"
        f"{question}\n"
    )


def build_cot_prompt(question: str) -> str:
    return (
        "You are an expert operations researcher.\n"
        "Solve the following optimization word problem that corresponds to a linear program.\n"
        "First, think step by step and explain your reasoning briefly.\n"
        "Then, on the LAST line, output the final optimal objective value in the form:\n"
        "ANSWER: <number>\n\n"
        "Problem:\n"
        f"{question}\n"
    )


def build_reflection_prompt(question: str, prev_answer: Optional[float], prev_reasoning: str) -> str:
    """
    Reflection / verifier style prompt: ask GPT-5.1 to re-check its own reasoning.
    NOTE: we DO NOT mention the ground-truth here; this is a realistic verifier loop.
    """
    prev_answer_str = "unknown"
    if prev_answer is not None:
        prev_answer_str = str(prev_answer)

    return (
        "You previously attempted to solve the following optimization problem and might have made mistakes.\n"
        "Carefully re-check the solution and correct any errors.\n"
        "You must output a NEW final answer if the previous one is incorrect.\n"
        "At the end, output the final answer in the form:\n"
        "ANSWER: <number>\n\n"
        "Original problem:\n"
        f"{question}\n\n"
        "Your previous reasoning and answer were:\n"
        f"{prev_reasoning}\n\n"
        f"Previous final numeric answer (possibly wrong): {prev_answer_str}\n\n"
        "Now provide a corrected, carefully verified solution."
    )


def build_pot_prompt(question: str) -> str:
    """
    Program-of-Thoughts: ask GPT-5.1 to write a small Python function solve() that returns the objective value.
    We will execute this function locally.
    """
    return (
        "You are an expert in mathematical optimization and Python programming.\n"
        "You will solve the following linear programming word problem by writing a Python function.\n\n"
        "Requirements:\n"
        "1. Write a Python function with signature `def solve():` that returns the optimal objective value as a float.\n"
        "2. Do NOT use any external libraries (no pulp, no cvxpy, etc.).\n"
        "3. You may use basic Python (loops, arithmetic, if, etc.) to compute the answer.\n"
        "4. Do NOT print anything. Just compute and return the value.\n"
        "5. Only output the Python code, enclosed in a single code block starting with ```python and ending with ```.\n\n"
        "Problem:\n"
        f"{question}\n"
    )


def build_tot_prompt(question: str, num_branches: int) -> str:
    """
    Tree-of-Thought style: ask GPT-5.1 to explore multiple candidate solutions (branches) before committing.
    We still just do one call, but ask for several distinct lines of reasoning.
    """
    return (
        "You are an expert operations researcher.\n"
        "You will solve the following optimization word problem that corresponds to a linear program.\n\n"
        f"Use a Tree-of-Thought approach: explore {num_branches} different plausible solution strategies.\n"
        "For each strategy, you should:\n"
        "  - Label it as 'Branch 1', 'Branch 2', etc.\n"
        "  - Show the reasoning and the candidate objective value.\n"
        "After exploring all branches, compare them and decide which one is correct.\n"
        "On the LAST line of your answer, output the final optimal objective value in the form:\n"
        "ANSWER: <number>\n\n"
        "Problem:\n"
        f"{question}\n"
    )


# ==========================
# 4. Strategy implementations
# ==========================

def strategy_direct(caller: GPT5Caller, question: str) -> Tuple[str, Optional[float], int]:
    """
    Direct answer: one call, no reasoning prompt.
    Returns: (raw_text, parsed_number, num_calls)
    """
    prompt = build_direct_prompt(question)
    text = caller.call(prompt, temperature=0.0)
    pred = extract_last_number(text)
    return text, pred, 1


def strategy_cot(caller: GPT5Caller, question: str) -> Tuple[str, Optional[float], int]:
    """
    Chain-of-Thought: one call with CoT prompt.
    We parse from the last number in the answer.
    """
    prompt = build_cot_prompt(question)
    text = caller.call(prompt, temperature=0.0)
    pred = extract_last_number(text)
    return text, pred, 1


def run_reflection_loop(
    caller: GPT5Caller,
    question: str,
    base_text: str,
    base_pred: Optional[float],
    max_rounds: int = 3,
    temperature: float = 0.0,
) -> Tuple[str, Optional[float], int]:
    """
    Generic reflection / verifier loop on top of a base attempt.
    We re-ask GPT-5.1 to re-check its solution up to max_rounds times.
    Returns:
        final_text,
        final_pred,
        total_calls (base + reflection calls)
    """
    history = [(base_text, base_pred)]
    total_calls = 1
    last_text, last_pred = base_text, base_pred

    for _ in range(max_rounds - 1):
        refl_prompt = build_reflection_prompt(question, last_pred, last_text)
        refl_text = caller.call(refl_prompt, temperature=temperature)
        refl_pred = extract_last_number(refl_text)

        history.append((refl_text, refl_pred))
        total_calls += 1

        # simple convergence test: if numeric answer stabilizes, stop
        if last_pred is not None and refl_pred is not None:
            if abs(refl_pred - last_pred) < 1e-9:
                last_text, last_pred = refl_text, refl_pred
                break

        last_text, last_pred = refl_text, refl_pred

    final_text, final_pred = history[-1]
    return final_text, final_pred, total_calls


def strategy_cot_reflect(
    caller: GPT5Caller,
    question: str,
    max_rounds: int = 3,
) -> Tuple[str, Optional[float], int]:
    """
    CoT + verifier loop: first do CoT, then run reflection up to max_rounds times.
    """
    base_text, base_pred, _ = strategy_cot(caller, question)
    final_text, final_pred, total_calls = run_reflection_loop(
        caller, question, base_text, base_pred, max_rounds=max_rounds
    )
    return final_text, final_pred, total_calls


def extract_code_block(text: str) -> Optional[str]:
    """
    Extract the first ```python ... ``` code block. If not found, try bare ``` ... ``` block.
    """
    m = re.search(r"```python(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"```(.*?)```", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def execute_pot_code(code: str) -> Optional[float]:
    """
    Execute the model-generated Python code in a minimal namespace and call solve().
    NOTE: This executes arbitrary code from the model. In a real system, sandbox this.
    """
    local_ns: Dict[str, Any] = {}
    try:
        exec(code, {}, local_ns)  # no globals, only locals
        if "solve" not in local_ns:
            return None
        result = local_ns["solve"]()
        return float(result)
    except Exception as e:
        print("Error executing PoT code:", repr(e))
        return None


def strategy_pot(caller: GPT5Caller, question: str) -> Tuple[str, Optional[float], int]:
    """
    Program-of-Thoughts: GPT-5.1 generates Python code, we execute solve().
    """
    prompt = build_pot_prompt(question)
    text = caller.call(prompt, temperature=0.0)
    code = extract_code_block(text)
    if code is None:
        pred = None
    else:
        pred = execute_pot_code(code)
    return text, pred, 1


def strategy_pot_reflect(
    caller: GPT5Caller,
    question: str,
    max_rounds: int = 3,
) -> Tuple[str, Optional[float], int]:
    """
    PoT + verifier loop: run PoT once, then use reflection loop on the textual reasoning.
    For now, we reflect on the TEXT (not the code) and parse again numerically.
    """
    base_text, base_pred, _ = strategy_pot(caller, question)
    final_text, final_pred, total_calls = run_reflection_loop(
        caller, question, base_text, base_pred, max_rounds=max_rounds
    )
    return final_text, final_pred, total_calls


def strategy_tot(
    caller: GPT5Caller,
    question: str,
    num_branches: int = 3,
    temperature: float = 0.7,
) -> Tuple[str, Optional[float], int]:
    """
    Simple Tree-of-Thought style: one call where GPT-5.1 explores several branches,
    then picks a final ANSWER: <number>. We parse the last number as usual.
    """
    prompt = build_tot_prompt(question, num_branches=num_branches)
    text = caller.call(prompt, temperature=temperature)
    pred = extract_last_number(text)
    return text, pred, 1


def strategy_tot_reflect(
    caller: GPT5Caller,
    question: str,
    num_branches: int = 3,
    temperature: float = 0.7,
    max_rounds: int = 3,
) -> Tuple[str, Optional[float], int]:
    """
    ToT + verifier loop: ToT once, then reflection.
    """
    base_text, base_pred, _ = strategy_tot(
        caller, question, num_branches=num_branches, temperature=temperature
    )
    final_text, final_pred, total_calls = run_reflection_loop(
        caller, question, base_text, base_pred, max_rounds=max_rounds
    )
    return final_text, final_pred, total_calls


# ==========================
# 5. Evaluation loop for GPT-5.1
# ==========================

def evaluate_gpt5_strategies_on_nl4opt(
    n_examples: Optional[int] = 20,
    random_state: int = 0,
    tol: float = 1e-3,
):
    """
    Run GPT-5.1 on NL4OPT with different reasoning strategies.
    If n_examples is None → use full dataset (245 problems).
    """
    df = load_nl4opt(n_examples=n_examples, random_state=random_state)
    caller = GPT5Caller(model_name="gpt-5.1")

    # Configure which strategies to run
    # name -> function
    strategies = {
        "direct": lambda q: strategy_direct(caller, q),
        "cot": lambda q: strategy_cot(caller, q),
        "cot_reflect": lambda q: strategy_cot_reflect(caller, q, max_rounds=3),
        "pot": lambda q: strategy_pot(caller, q),
        "pot_reflect": lambda q: strategy_pot_reflect(caller, q, max_rounds=3),
        "tot": lambda q: strategy_tot(caller, q, num_branches=3, temperature=0.7),
        "tot_reflect": lambda q: strategy_tot_reflect(
            caller, q, num_branches=3, temperature=0.7, max_rounds=3
        ),
    }

    all_results: List[Dict[str, Any]] = []

    total_jobs = len(df) * len(strategies)
    print(f"\nTotal evaluations to run with GPT-5.1: {total_jobs}\n")

    job_idx = 0
    for i, row in df.iterrows():
        question = row["en_question"]
        gt = float(row["en_answer"])

        print(f"\n=== Example {i+1}/{len(df)} ===")
        print("Question (truncated to 200 chars):")
        print(question[:200] + ("..." if len(question) > 200 else ""))
        print("Ground truth objective value:", gt)

        for strat_name, strat_fn in strategies.items():
            job_idx += 1
            print(f"\n[{job_idx}/{total_jobs}] Strategy={strat_name}")

            t0 = time.time()
            try:
                raw_text, pred_num, num_calls = strat_fn(question)
                latency = time.time() - t0
                verif = verify_answer_numeric(pred_num, gt, tol=tol)

                print("Response (truncated to 200 chars):")
                preview = (raw_text or "")[:200].replace("\n", " ")
                print(preview + ("..." if raw_text and len(raw_text) > 200 else ""))
                print(
                    f"Parsed={pred_num}, correct={verif['correct']}, "
                    f"abs_err={verif['abs_error']}, calls={num_calls}"
                )

                all_results.append({
                    "example_idx": i,
                    "strategy": strat_name,
                    "gt": gt,
                    "raw_response": raw_text,
                    "parsed_answer": pred_num,
                    "correct": verif["correct"],
                    "abs_error": verif["abs_error"],
                    "rel_error": verif["rel_error"],
                    "latency_sec": latency,
                    "num_calls": num_calls,
                    "error": None,
                })
            except Exception as e:
                latency = time.time() - t0
                print("Error in strategy:", strat_name, " -> ", repr(e))
                all_results.append({
                    "example_idx": i,
                    "strategy": strat_name,
                    "gt": gt,
                    "raw_response": None,
                    "parsed_answer": None,
                    "correct": False,
                    "abs_error": None,
                    "rel_error": None,
                    "latency_sec": latency,
                    "num_calls": None,
                    "error": repr(e),
                })

    res_df = pd.DataFrame(all_results)
    os.makedirs("results", exist_ok=True)
    out_path = "results/nl4opt_gpt5_strategies.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary: accuracy and errors per strategy
    summary = (
        res_df.groupby("strategy")
        .agg(
            n=("correct", "size"),
            n_ok=("correct", "sum"),
            acc=("correct", "mean"),
            mean_abs_err=("abs_error", "mean"),
            mean_rel_err=("rel_error", "mean"),
            mean_latency=("latency_sec", "mean"),
            mean_calls=("num_calls", "mean"),
        )
        .reset_index()
    )

    print("\n=== Summary per strategy (GPT-5.1) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    # For debugging: use n_examples=10 or 20
    # For full dataset: set n_examples=None
    evaluate_gpt5_strategies_on_nl4opt(
        n_examples=10,     # change to None later for full run
        random_state=0,
        tol=1e-3,
    )
