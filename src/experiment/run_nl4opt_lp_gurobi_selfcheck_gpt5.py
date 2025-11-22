#!/usr/bin/env python3
"""
run_nl4opt_lp_gurobi_selfcheck_gpt5.py

Extension of the LP+Gurobi pipeline:
  Text  → GPT-5.1 (LP JSON, round 1)
        → GPT-5.1 (LP JSON, round 2, semantic self-check)
        → Gurobi solves final LP
        → Compare solver objective vs ground truth

Self-consistency pass:
  The second GPT call explicitly re-reads the problem and its own
  previous JSON, and is instructed to fix any mismatch between the
  story and the LP (objective, signs, directions, bounds, etc.).
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import gurobipy as gp
from gurobipy import GRB


# ==========================================
# 1. Dataset utilities (same as before)
# ==========================================

def load_nl4opt_numeric() -> pd.DataFrame:
    """
    Load NL4OPT_with_optimal_solution.json and keep only rows
    where en_answer is numeric (drop 'No Best Solution', etc.).
    """
    path = "hf://datasets/CardinalOperations/NL4OPT/NL4OPT_with_optimal_solution.json"
    print(f"[load_nl4opt_numeric] Loading from: {path}")
    df = pd.read_json(path, lines=True)

    df["en_answer_numeric"] = pd.to_numeric(df["en_answer"], errors="coerce")
    df = df[df["en_answer_numeric"].notna()].reset_index(drop=True)
    print(f"[load_nl4opt_numeric] Using {len(df)} numeric examples.")
    return df


def verify_answer_numeric(pred: Optional[float], gt: float, tol: float = 1e-3) -> Dict[str, Any]:
    """
    Simple numeric verifier: compares predicted number to ground truth
    with absolute tolerance 'tol'.
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


# ==========================================
# 2. GPT-5.1 caller
# ==========================================

class GPT5Caller:
    def __init__(self, model_name: str = "gpt-5.1"):
        load_dotenv()
        self.client = OpenAI()
        self.model_name = model_name

    def call(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Minimal wrapper to call GPT-5.1 with a plain text prompt.
        """
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
        )
        return response.output[0].content[0].text


# ==========================================
# 3. LP schema
# ==========================================

Sense = Literal["<=", ">=", "=="]
ObjSense = Literal["min", "max"]


@dataclass
class Constraint:
    name: str
    coeffs: Dict[str, float]  # variable_name -> coefficient
    sense: Sense
    rhs: float


@dataclass
class LinearProgram:
    sense: ObjSense                     # "min" or "max"
    objective: Dict[str, float]         # var_name -> coefficient
    constraints: List[Constraint] = field(default_factory=list)
    var_lower_bounds: Dict[str, float] = field(default_factory=dict)
    var_upper_bounds: Dict[str, float] = field(default_factory=dict)
    integer_vars: List[str] = field(default_factory=list)


# ==========================================
# 4. Gurobi solver wrapper
# ==========================================

def solve_lp_with_gurobi(lp: LinearProgram) -> Dict[str, Any]:
    """
    Build and solve an LP/MIP in Gurobi from a LinearProgram schema.

    Returns:
        {
            "status": "...",
            "obj_value": float or None,
            "solution": dict(var_name -> value) or None,
        }
    """
    model = gp.Model()
    model.Params.OutputFlag = 0  # silent

    # 1) Create variables
    vars_: Dict[str, gp.Var] = {}
    var_names = sorted(set(
        list(lp.objective.keys()) +
        [v for c in lp.constraints for v in c.coeffs.keys()]
    ))

    for name in var_names:
        lb = lp.var_lower_bounds.get(name, 0.0)
        ub = lp.var_upper_bounds.get(name, GRB.INFINITY)
        vtype = GRB.INTEGER if name in lp.integer_vars else GRB.CONTINUOUS
        vars_[name] = model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)

    model.update()

    # 2) Objective
    expr = gp.LinExpr()
    for name, coef in lp.objective.items():
        expr.add(vars_[name], coef)

    if lp.sense == "max":
        model.setObjective(expr, GRB.MAXIMIZE)
    else:
        model.setObjective(expr, GRB.MINIMIZE)

    # 3) Constraints
    for c in lp.constraints:
        lhs = gp.LinExpr()
        for name, coef in c.coeffs.items():
            lhs.add(vars_[name], coef)

        if c.sense == "<=":
            model.addConstr(lhs <= c.rhs, name=c.name)
        elif c.sense == ">=":
            model.addConstr(lhs >= c.rhs, name=c.name)
        else:  # "=="
            model.addConstr(lhs == c.rhs, name=c.name)

    # 4) Solve
    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }
    status_str = status_map.get(model.Status, f"STATUS_{model.Status}")

    if model.Status == GRB.OPTIMAL:
        obj = float(model.ObjVal)
        sol = {name: float(v.X) for name, v in vars_.items()}
    else:
        obj = None
        sol = None

    return {
        "status": status_str,
        "obj_value": obj,
        "solution": sol,
    }


# ==========================================
# 5. Prompt & parsing: Text → LP JSON → schema
# ==========================================

def build_lp_json_prompt(question: str) -> str:
    """
    Base prompt: from word problem to LP JSON.
    """
    return (
        "You are an expert in linear programming.\n"
        "Given the following word problem, you must convert it into a structured linear program.\n\n"
        "Output a single JSON object with the following fields exactly:\n\n"
        "{\n"
        '  \"sense\": \"max\" or \"min\",\n'
        '  \"objective\": { \"x\": 56, \"y\": 75 },\n'
        '  \"constraints\": [\n'
        "    {\n"
        '      \"name\": \"constraint_name\",\n'
        '      \"coeffs\": { \"x\": 6, \"y\": 3 },\n'
        '      \"sense\": \"<=\",\n'
        '      \"rhs\": 575\n'
        "    }\n"
        "  ],\n"
        '  \"var_lower_bounds\": { \"x\": 0, \"y\": 0 },\n'
        '  \"var_upper_bounds\": {},\n'
        '  \"integer_vars\": [\"x\", \"y\"]\n'
        "}\n\n"
        "Rules:\n"
        "- Use simple variable names like \"x\", \"y\" or \"x1\", \"x2\".\n"
        "- Translate ALL constraints from the story (capacity, budget, proportions, minimums, maximums, etc.).\n"
        "- If variables must be integer (e.g., number of items or trips), include them in \"integer_vars\".\n"
        "- Do not add comments or extra text.\n"
        "- Output ONLY valid JSON (no backticks, no code fences).\n\n"
        "Problem:\n"
        f"{question}\n"
    )


def build_lp_selfcheck_prompt(question: str, previous_json: str) -> str:
    """
    Self-consistency refinement prompt:

    - Show the original problem.
    - Show the model its own previous JSON LP.
    - Ask it to re-read the problem and fix *any* mismatch
      (wrong inequality direction, missing constraints,
       incorrect objective sense, missing bounds, etc.).
    - Output a NEW corrected JSON LP, same schema as before.
    """
    return (
        "You are an expert in linear programming.\n"
        "You previously converted the following word problem into a JSON linear program.\n"
        "Now you must carefully re-check your own formulation for correctness.\n\n"
        "Steps:\n"
        "1. Re-read the word problem and write down (internally) all constraints implied by the text.\n"
        "2. Compare each of these constraints to the JSON LP you produced.\n"
        "   - Check inequality directions (<= vs >=).\n"
        "   - Check right-hand-side values.\n"
        "   - Check that min vs max objective matches the story.\n"
        "   - Check lower/upper bounds and integrality constraints.\n"
        "3. If anything is missing or incorrect, FIX the LP in JSON form.\n\n"
        "IMPORTANT:\n"
        "- You MUST output a NEW JSON object of the same schema as before.\n"
        "- Do NOT add commentary, explanations, or code fences.\n"
        "- Output ONLY valid JSON.\n\n"
        "Word problem:\n"
        f"{question}\n\n"
        "Your previous JSON LP was:\n"
        f"{previous_json}\n\n"
        "Now output the corrected JSON LP:\n"
    )


def clean_json_text(raw: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if present.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def lp_from_json_dict(d: Dict[str, Any]) -> LinearProgram:
    """
    Convert a JSON dict (as produced by GPT) into a LinearProgram object.
    """
    sense = d.get("sense", "max")
    if sense not in ("max", "min"):
        sense = "max"

    objective = {str(k): float(v) for k, v in d.get("objective", {}).items()}

    constraints: List[Constraint] = []
    for c in d.get("constraints", []):
        name = str(c.get("name", "c"))
        coeffs = {str(k): float(v) for k, v in c.get("coeffs", {}).items()}
        s = c.get("sense", "<=")
        if s not in ("<=", ">=", "=="):
            s = "<="
        rhs = float(c.get("rhs", 0.0))
        constraints.append(Constraint(name=name, coeffs=coeffs, sense=s, rhs=rhs))

    var_lb = {str(k): float(v) for k, v in d.get("var_lower_bounds", {}).items()}
    var_ub = {str(k): float(v) for k, v in d.get("var_upper_bounds", {}).items()}
    integer_vars = [str(v) for v in d.get("integer_vars", [])]

    return LinearProgram(
        sense=sense,
        objective=objective,
        constraints=constraints,
        var_lower_bounds=var_lb,
        var_upper_bounds=var_ub,
        integer_vars=integer_vars,
    )


def parse_lp_json_to_schema(raw_text: str) -> Optional[LinearProgram]:
    """
    Parse raw LLM output (which should be JSON) into a LinearProgram.
    Returns None if parsing fails.
    """
    try:
        cleaned = clean_json_text(raw_text)
        d = json.loads(cleaned)
        lp = lp_from_json_dict(d)
        return lp
    except Exception as e:
        print("[LP-JSON] Error parsing JSON:", repr(e))
        return None


# ==========================================
# 6. LP + Gurobi with self-consistency pass
# ==========================================

def strategy_lp_gurobi_selfcheck(
    caller: GPT5Caller,
    question: str,
) -> Dict[str, Any]:
    """
    LLM+Gurobi strategy with a semantic self-consistency pass:

    1. First call: build LP JSON from the word problem.
    2. Second call: show the problem + previous JSON,
       ask model to re-check and fix inconsistencies.
    3. Parse final JSON → LinearProgram.
    4. Solve with Gurobi.

    Returns:
        {
            "status": ...,
            "obj_value": ...,
            "solution": ...,
            "lp_json_round1": ...,
            "lp_json_round2": ...,
            "num_calls": 2 (if both succeed),
        }
    """
    num_calls = 0

    # Round 1: initial LP JSON
    prompt1 = build_lp_json_prompt(question)
    raw_json1 = caller.call(prompt1, temperature=0.0)
    num_calls += 1

    # Round 2: self-consistency refinement
    prompt2 = build_lp_selfcheck_prompt(question, raw_json1)
    raw_json2 = caller.call(prompt2, temperature=0.0)
    num_calls += 1

    # Parse final JSON into schema
    lp = parse_lp_json_to_schema(raw_json2)
    if lp is None:
        return {
            "status": "PARSE_ERROR",
            "obj_value": None,
            "solution": None,
            "lp_json_round1": raw_json1,
            "lp_json_round2": raw_json2,
            "num_calls": num_calls,
        }

    result = solve_lp_with_gurobi(lp)

    return {
        "status": result["status"],
        "obj_value": result["obj_value"],
        "solution": result["solution"],
        "lp_json_round1": raw_json1,
        "lp_json_round2": raw_json2,
        "num_calls": num_calls,
    }


# ==========================================
# 7. Evaluation loop on NL4OPT (self-check)
# ==========================================

def evaluate_nl4opt_lp_gurobi_selfcheck(
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-3,
):
    """
    Run GPT-5.1 LP+Gurobi with semantic self-consistency pass on NL4OPT.

    Numeric evaluation:
      - Only on gt > 0 and status=OPTIMAL.
    """
    df_data = load_nl4opt_numeric()
    if n_examples is not None:
        df_data = df_data.sample(n=n_examples, random_state=random_state).reset_index(drop=True)
        print(f"[eval] Using sample of {len(df_data)} examples.")
    else:
        print(f"[eval] Using FULL numeric dataset: {len(df_data)} examples.")

    caller = GPT5Caller(model_name="gpt-5.1")

    rows = []
    for i, row in df_data.iterrows():
        q = row["en_question"]
        gt = float(row["en_answer_numeric"])

        print(f"\n=== Example {i+1}/{len(df_data)} ===")
        print(q[:160] + ("..." if len(q) > 160 else ""))
        print("GT objective:", gt)

        t0 = time.time()
        result = strategy_lp_gurobi_selfcheck(caller, q)
        latency = time.time() - t0

        status = result["status"]
        obj = result["obj_value"]
        num_calls = result["num_calls"]

        print(f"Status={status}, obj_value={obj}, calls={num_calls}")

        if gt > 0 and status == "OPTIMAL" and obj is not None:
            verif = verify_answer_numeric(obj, gt, tol=tol)
            correct = verif["correct"]
            abs_err = verif["abs_error"]
            rel_err = verif["rel_error"]
        else:
            correct = False
            abs_err = None
            rel_err = None

        rows.append({
            "example_idx": i,
            "gt": gt,
            "status": status,
            "pred_obj": obj,
            "correct": correct,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "latency_sec": latency,
            "num_calls": num_calls,
        })

    dfres = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_path = (
        "results/nl4opt_gpt5_lp_gurobi_selfcheck_full.csv"
        if n_examples is None
        else "results/nl4opt_gpt5_lp_gurobi_selfcheck_sample.csv"
    )
    dfres.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    # Summary over numeric cases (gt > 0 & status=OPTIMAL)
    df_num = dfres[(dfres["gt"] > 0) & (dfres["status"] == "OPTIMAL")]
    if not df_num.empty:
        summary = {
            "n_total": len(dfres),
            "n_numeric_eval": len(df_num),
            "n_ok": int(df_num["correct"].sum()),
            "acc": float(df_num["correct"].mean()),
            "mean_abs_err": float(df_num["abs_error"].mean()),
            "mean_rel_err": float(df_num["rel_error"].mean()),
            "mean_latency_sec": float(df_num["latency_sec"].mean()),
            "mean_calls": float(df_num["num_calls"].mean()),
        }
        print("\n=== Summary (LP + Gurobi + Self-Check, numeric gt>0 & OPTIMAL) ===")
        for k, v in summary.items():
            print(f"{k:16s} = {v}")
    else:
        print("\n[warn] No numeric cases with status=OPTIMAL to summarize.")


if __name__ == "__main__":
    # For quick tests: n_examples=10 or 20
    # For full run: n_examples=None
    evaluate_nl4opt_lp_gurobi_selfcheck(
        n_examples=None,   # full numeric subset
        random_state=0,
        tol=1e-3,
    )
