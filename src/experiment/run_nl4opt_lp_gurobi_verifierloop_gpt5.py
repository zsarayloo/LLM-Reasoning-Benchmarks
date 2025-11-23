#!/usr/bin/env python3
"""
run_nl4opt_lp_gurobi_verifierloop_gpt5.py

Pipeline: NL4OPT → GPT-5.1 (Text→LP JSON) → Gurobi (solve) → Verifier loop.

Verifier loop idea:
  - Round 1: GPT-5.1 generates an LP JSON from the word problem.
  - Solve with Gurobi.
  - If status is not OPTIMAL (e.g., INFEASIBLE, UNBOUNDED, INF_OR_UNBD, etc.),
    we create a natural-language feedback message based on the solver status
    and pass it back to GPT-5.1 to REGENERATE a *new* LP JSON from scratch.
  - We repeat this for up to max_rounds.
  - We do NOT feed ground-truth answers into the model.
  - We only use ground truth to evaluate final performance.

Outputs:
  - CSV: results/nl4opt_gpt5_lp_gurobi_verifierloop_full.csv
  - Summary: accuracy, abs/rel error, latency, calls per example.
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
# 1. Dataset utilities
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
    with absolute tolerance 'tol'. Used ONLY for evaluation.
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
    coeffs: Dict[str, float]
    sense: Sense
    rhs: float


@dataclass
class LinearProgram:
    sense: ObjSense                     # "max" or "min"
    objective: Dict[str, float]         # var_name -> coefficient
    constraints: List[Constraint] = field(default_factory=list)
    var_lower_bounds: Dict[str, float] = field(default_factory=dict)
    var_upper_bounds: Dict[str, float] = field(default_factory=dict)
    integer_vars: List[str] = field(default_factory=list)


# ==========================================
# 4. Gurobi solver wrapper + diagnostic helpers
# ==========================================

def solve_lp_with_gurobi(lp: LinearProgram) -> Dict[str, Any]:
    """
    Build and solve an LP/MIP in Gurobi from a LinearProgram schema.

    Returns:
        {
            "status": "...",
            "obj_value": float or None,
            "solution": dict(var_name -> value) or None,
            "model": gurobipy.Model,
        }
    Note: 'model' is returned for diagnostic use (IIS, etc.).
    """
    model = gp.Model()
    model.Params.OutputFlag = 0  # silent

    vars_: Dict[str, gp.Var] = {}
    var_names = sorted(set(
        list(lp.objective.keys()) +
        [v for c in lp.constraints for v in c.coeffs.keys()]
    ))

    # Variables
    for name in var_names:
        lb = lp.var_lower_bounds.get(name, 0.0)
        ub = lp.var_upper_bounds.get(name, GRB.INFINITY)
        vtype = GRB.INTEGER if name in lp.integer_vars else GRB.CONTINUOUS
        vars_[name] = model.addVar(lb=lb, ub=ub, vtype=vtype, name=name)

    model.update()

    # Objective
    expr = gp.LinExpr()
    for name, coef in lp.objective.items():
        expr.add(vars_[name], coef)

    if lp.sense == "max":
        model.setObjective(expr, GRB.MAXIMIZE)
    else:
        model.setObjective(expr, GRB.MINIMIZE)

    # Constraints
    for c in lp.constraints:
        lhs = gp.LinExpr()
        for name, coef in c.coeffs.items():
            lhs.add(vars_[name], coef)
        if c.sense == "<=":
            model.addConstr(lhs <= c.rhs, name=c.name)
        elif c.sense == ">=":
            model.addConstr(lhs >= c.rhs, name=c.name)
        else:
            model.addConstr(lhs == c.rhs, name=c.name)

    # Solve
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
        "model": model,
    }


def get_infeasibility_report(model: gp.Model) -> str:
    """
    If the model is infeasible, compute an IIS and return a short
    textual report listing constraint names in the IIS.
    If IIS computation fails or status is not INFEASIBLE, return a generic message.
    """
    if model.Status != GRB.INFEASIBLE:
        return "Model is not infeasible, no IIS available."

    try:
        model.computeIIS()
        lines = ["Gurobi reports the model is INFEASIBLE.",
                 "A minimal infeasible subsystem (IIS) includes constraints:"]

        for c in model.getConstrs():
            try:
                if c.IISConstr:
                    lines.append(f"  - {c.ConstrName}")
            except AttributeError:
                # Older/newer API differences: fallback generic
                pass

        if len(lines) == 2:
            lines.append("  (IIS constraints could not be identified individually.)")
        return "\n".join(lines)
    except Exception as e:
        return f"Gurobi reports the model is INFEASIBLE but IIS computation failed: {repr(e)}"


def build_solver_feedback(status: str, obj_value: Optional[float], model: gp.Model) -> str:
    """
    Build a natural-language feedback string for GPT based on Gurobi status and,
    when feasible, an IIS or unboundedness hint.
    """
    if status == "OPTIMAL":
        return (
            "Gurobi successfully solved your LP and found an OPTIMAL solution. "
            "This means the LP is internally consistent. However, the LP might still "
            "not exactly match the story. If asked to regenerate, focus on checking "
            "if all story constraints and objective are modelled correctly, not on "
            "fixing feasibility."
        )
    elif status == "INFEASIBLE":
        return get_infeasibility_report(model)
    elif status == "UNBOUNDED":
        return (
            "Gurobi reports the model is UNBOUNDED. This usually means that the objective "
            "can increase or decrease without limit. In many word problems, the story usually "
            "imposes implicit upper bounds (e.g., resource limits, maximum trips, etc.). "
            "You likely forgot some bounding constraints or used the wrong inequality direction."
        )
    elif status == "INF_OR_UNBD":
        return (
            "Gurobi reports the model is either INFEASIBLE or UNBOUNDED. "
            "This suggests that one or more constraints are missing, reversed, or numerically "
            "incorrect (e.g., wrong right-hand sides). You should re-derive the LP carefully."
        )
    else:
        return (
            f"Gurobi returned status {status}. This indicates something is wrong with the LP "
            "formulation (feasibility or bounds). Re-check your constraints and objective."
        )


# ==========================================
# 5. Text → LP JSON + parsing
# ==========================================

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
# 6. Prompts: base LP + verifier loop
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


def build_lp_verifier_prompt(
    question: str,
    previous_json: str,
    solver_status: str,
    solver_feedback: str,
    round_index: int,
) -> str:
    """
    Verifier-loop prompt: GPT sees the word problem, the last LP JSON, and
    Gurobi's diagnostic feedback, then must regenerate a better LP JSON.
    """
    return (
        "You are an expert in linear programming and optimization.\n"
        "You previously formulated the following LP for a word problem.\n"
        "A Gurobi solver tried to solve it and produced the status and diagnostic feedback below.\n"
        "Based on this feedback, you must RE-THINK the formulation and output a NEW LP JSON.\n\n"
        "IMPORTANT:\n"
        "- You MUST read the word problem again and derive the LP cleanly.\n"
        "- Use the solver feedback only as a hint about what might be wrong.\n"
        "- You are allowed to completely change the constraints, objective, and bounds.\n"
        "- Do NOT try to minimally patch the old JSON. Instead, derive a correct LP from scratch.\n"
        "- Output ONLY valid JSON (no explanations, no backticks, no comments).\n"
        "- Use the same JSON structure as before (sense, objective, constraints, var_lower_bounds, var_upper_bounds, integer_vars).\n\n"
        f"Round index: {round_index}\n\n"
        "Word problem:\n"
        f"{question}\n\n"
        "Your previous LP JSON was:\n"
        f"{previous_json}\n\n"
        "Gurobi solver status was:\n"
        f"{solver_status}\n\n"
        "Gurobi diagnostic feedback was:\n"
        f"{solver_feedback}\n\n"
        "Now output the NEW LP JSON:\n"
    )


# ==========================================
# 7. Strategy: LP + Gurobi + verifier loop
# ==========================================

def strategy_lp_gurobi_verifierloop(
    caller: GPT5Caller,
    question: str,
    max_rounds: int = 2,
) -> Dict[str, Any]:
    """
    Verifier-loop strategy:

      For up to max_rounds:
        - Ask GPT-5.1 to generate an LP JSON (round 1: base prompt; later: verifier prompt).
        - Parse JSON to LinearProgram.
        - Solve with Gurobi.
        - If status == OPTIMAL, stop and return.
        - Otherwise, build solver feedback and re-ask GPT-5.1 in next round.

    Returns:
      {
        "status": (final Gurobi status),
        "obj_value": float or None,
        "solution": dict(var->val) or None,
        "lp_json_rounds": [str, ...],
        "statuses": [str, ...],
        "num_calls": int,
        "parse_fail": bool,
      }
    """
    lp_json_rounds: List[str] = []
    statuses: List[str] = []
    num_calls = 0
    parse_fail = False

    # Round 1
    prompt = build_lp_json_prompt(question)
    raw_json = caller.call(prompt, temperature=0.0)
    num_calls += 1
    lp_json_rounds.append(raw_json)

    lp = parse_lp_json_to_schema(raw_json)
    if lp is None:
        # If we cannot parse even the first LP, abort for this example.
        return {
            "status": "PARSE_ERROR",
            "obj_value": None,
            "solution": None,
            "lp_json_rounds": lp_json_rounds,
            "statuses": [],
            "num_calls": num_calls,
            "parse_fail": True,
        }

    # Solve first LP
    result = solve_lp_with_gurobi(lp)
    status = result["status"]
    statuses.append(status)

    if status == "OPTIMAL" or max_rounds == 1:
        return {
            "status": status,
            "obj_value": result["obj_value"],
            "solution": result["solution"],
            "lp_json_rounds": lp_json_rounds,
            "statuses": statuses,
            "num_calls": num_calls,
            "parse_fail": False,
        }

    # Otherwise, we use verifier loop for additional rounds
    for r in range(2, max_rounds + 1):
        solver_feedback = build_solver_feedback(status, result["obj_value"], result["model"])
        verifier_prompt = build_lp_verifier_prompt(
            question=question,
            previous_json=raw_json,
            solver_status=status,
            solver_feedback=solver_feedback,
            round_index=r,
        )
        raw_json = caller.call(verifier_prompt, temperature=0.0)
        num_calls += 1
        lp_json_rounds.append(raw_json)

        lp = parse_lp_json_to_schema(raw_json)
        if lp is None:
            parse_fail = True
            break

        result = solve_lp_with_gurobi(lp)
        status = result["status"]
        statuses.append(status)

        if status == "OPTIMAL":
            break

    return {
        "status": status,
        "obj_value": result["obj_value"],
        "solution": result["solution"],
        "lp_json_rounds": lp_json_rounds,
        "statuses": statuses,
        "num_calls": num_calls,
        "parse_fail": parse_fail,
    }


# ==========================================
# 8. Evaluation loop on NL4OPT
# ==========================================

def evaluate_nl4opt_lp_gurobi_verifierloop(
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-3,
    max_rounds: int = 2,
):
    """
    Run GPT-5.1 LP+Gurobi with verifier loop on NL4OPT.

    Numeric evaluation:
      - Only on gt > 0 and final status = OPTIMAL.
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
        result = strategy_lp_gurobi_verifierloop(
            caller,
            question=q,
            max_rounds=max_rounds,
        )
        latency = time.time() - t0

        status = result["status"]
        obj = result["obj_value"]
        num_calls = result["num_calls"]

        print(f"Final status={status}, obj_value={obj}, calls={num_calls}")

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
            "parse_fail": result["parse_fail"],
            "statuses_per_round": "|".join(result["statuses"]),
        })

    dfres = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)

    out_path = (
        "results/nl4opt_gpt5_lp_gurobi_verifierloop_full.csv"
        if n_examples is None
        else "results/nl4opt_gpt5_lp_gurobi_verifierloop_sample.csv"
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
        print("\n=== Summary (LP + Gurobi + VerifierLoop, numeric gt>0 & OPTIMAL) ===")
        for k, v in summary.items():
            print(f"{k:16s} = {v}")
    else:
        print("\n[warn] No numeric cases with status=OPTIMAL to summarize.")


if __name__ == "__main__":
    # For quick tests: n_examples=10 or 20
    # For the full numeric subset: n_examples=None
    evaluate_nl4opt_lp_gurobi_verifierloop(
        n_examples=None,
        random_state=0,
        tol=1e-3,
        max_rounds=2,   # you can try 3 for stronger verifier loop
    )
