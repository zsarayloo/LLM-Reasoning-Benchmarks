#!/usr/bin/env python3
"""
run_nl4opt_lp_gurobi_selfcheck_patch_gpt5.py

LP + Gurobi pipeline with a *conservative* self-consistency pass:

  1) GPT-5.1 converts text → LP JSON (base formulation).
  2) GPT-5.1 sees the problem + base JSON and outputs a small PATCH JSON:
       - may flip constraint senses (<=, >=, ==),
       - may adjust constraint RHS,
       - may adjust objective sense or individual coefficients.
     NO adding/removing constraints (conservative option A).
  3) We apply the patch in Python (minimal edits) to get a refined LP.
  4) Gurobi solves the refined LP.
  5) We evaluate objective vs ground truth.

Outputs a CSV:
  results/nl4opt_gpt5_lp_gurobi_selfcheck_patch_full.csv
"""

import os
import json
import time
import copy
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
# 5. Text → LP JSON + parsing
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
# 6. Patch prompt + patch application (Option A: conservative)
# ==========================================

def build_lp_patch_prompt(question: str, previous_json: str) -> str:
    """
    Patch-style self-consistency prompt.

    Conservative rules (Option A):
      - Do NOT add new constraints.
      - Do NOT remove constraints.
      - You may:
          * change the objective sense (min/max) if clearly wrong,
          * adjust specific objective coefficients,
          * flip constraint sense (<, >, =) if mis-specified,
          * adjust constraint RHS if mis-specified.
    """
    return (
        "You are an expert in linear programming.\n"
        "You previously converted the following word problem into a JSON linear program.\n"
        "Now you must carefully re-check your own formulation for correctness.\n\n"
        "CONSERVATIVE PATCHING RULES (Option A):\n"
        "  - Do NOT add any new constraints that were not present before.\n"
        "  - Do NOT remove any constraints.\n"
        "  - You may:\n"
        "      * change the objective sense (\"max\" vs \"min\") if the story clearly says so,\n"
        "      * adjust individual objective coefficients if they clearly mismatch the story,\n"
        "      * flip constraint senses (\"<=\", \">=\", \"==\") if they are reversed,\n"
        "      * adjust right-hand side constants (rhs) if they are numerically wrong.\n"
        "  - If a field is not mentioned in the patch, you must assume it stays unchanged.\n\n"
        "PATCH FORMAT (JSON only, no extra text, no code fences):\n"
        "{\n"
        "  \"objective\": {\n"
        "    \"sense\": null or \"max\" or \"min\",\n"
        "    \"coeff_updates\": { \"x\": 56.0, \"y\": 75.0 }\n"
        "  },\n"
        "  \"constraint_updates\": [\n"
        "    {\n"
        "      \"name\": \"constraint_name\",   // name must match an existing constraint\n"
        "      \"sense\": \"<=\" or \">=\" or \"==\" (or null to keep),\n"
        "      \"rhs\": 123.0  // new RHS (or null to keep)\n"
        "    }\n"
        "  ],\n"
        "  \"var_bound_updates\": [\n"
        "    {\n"
        "      \"var\": \"x\",\n"
        "      \"lower\": 0.0,   // or null to keep\n"
        "      \"upper\": null   // or a number\n"
        "    }\n"
        "  ],\n"
        "  \"integer_add\": [\"x2\"],\n"
        "  \"integer_remove\": [\"x3\"]\n"
        "}\n\n"
        "Notes:\n"
        "- Use null where no change is needed.\n"
        "- For conservative patching, you will usually leave most fields null or empty.\n"
        "- Output ONLY valid JSON, no comments, no backticks, no extra text.\n\n"
        "Word problem:\n"
        f"{question}\n\n"
        "Your previous JSON LP was:\n"
        f"{previous_json}\n\n"
        "Now output the PATCH JSON:\n"
    )


def apply_lp_patch(base_lp: LinearProgram, patch: Dict[str, Any]) -> LinearProgram:
    """
    Apply a conservative patch to a base LP:

      - objective.sense: may change
      - objective.coeff_updates: may overwrite specific coefficients
      - constraint_updates: for existing constraint names only
      - var_bound_updates: adjust LB/UB
      - integer_add / integer_remove: update integrality set

    Unrecognized / malformed pieces are ignored.
    """
    lp = copy.deepcopy(base_lp)

    # 1) Objective patch
    obj_patch = patch.get("objective") or {}
    sense_patch = obj_patch.get("sense", None)
    if sense_patch in ("max", "min"):
        lp.sense = sense_patch

    coeff_updates = obj_patch.get("coeff_updates") or {}
    if isinstance(coeff_updates, dict):
        for vname, val in coeff_updates.items():
            try:
                lp.objective[str(vname)] = float(val)
            except (TypeError, ValueError):
                continue

    # 2) Constraint updates
    cu_list = patch.get("constraint_updates") or []
    if isinstance(cu_list, list):
        # build index by name
        by_name = {c.name: c for c in lp.constraints}
        for cu in cu_list:
            if not isinstance(cu, dict):
                continue
            name = cu.get("name", None)
            if name is None:
                continue
            name = str(name)
            if name not in by_name:
                # conservative: ignore updates to unknown constraints
                continue
            c = by_name[name]
            sense_new = cu.get("sense", None)
            rhs_new = cu.get("rhs", None)

            if sense_new in ("<=", ">=", "=="):
                c.sense = sense_new
            if rhs_new is not None:
                try:
                    c.rhs = float(rhs_new)
                except (TypeError, ValueError):
                    pass

    # 3) Var bound updates
    vb_list = patch.get("var_bound_updates") or []
    if isinstance(vb_list, list):
        for vb in vb_list:
            if not isinstance(vb, dict):
                continue
            vname = vb.get("var", None)
            if vname is None:
                continue
            vname = str(vname)
            lower = vb.get("lower", None)
            upper = vb.get("upper", None)
            if lower is not None:
                try:
                    lp.var_lower_bounds[vname] = float(lower)
                except (TypeError, ValueError):
                    pass
            if upper is not None:
                try:
                    lp.var_upper_bounds[vname] = float(upper)
                except (TypeError, ValueError):
                    pass

    # 4) Integer var updates
    int_add = patch.get("integer_add") or []
    int_remove = patch.get("integer_remove") or []
    int_set = set(lp.integer_vars)

    if isinstance(int_add, list):
        for v in int_add:
            int_set.add(str(v))
    if isinstance(int_remove, list):
        for v in int_remove:
            vn = str(v)
            if vn in int_set:
                int_set.remove(vn)

    lp.integer_vars = sorted(int_set)

    return lp


def parse_patch_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the patch JSON. Returns None if parsing fails.
    """
    try:
        cleaned = clean_json_text(raw_text)
        d = json.loads(cleaned)
        if not isinstance(d, dict):
            return None
        return d
    except Exception as e:
        print("[LP-PATCH] Error parsing patch JSON:", repr(e))
        return None


# ==========================================
# 7. Strategy: LP + Gurobi + conservative patch self-check
# ==========================================

def strategy_lp_gurobi_selfcheck_patch(
    caller: GPT5Caller,
    question: str,
) -> Dict[str, Any]:
    """
    Strategy:
      1) GPT-5.1 builds initial LP JSON.
      2) We parse to LinearProgram (base_lp).
      3) GPT-5.1 produces a *conservative patch* JSON.
      4) We apply patch to base_lp to get refined_lp.
      5) Gurobi solves refined_lp.

    Returns:
      {
        "status": ...,
        "obj_value": ...,
        "solution": ...,
        "lp_json_round1": ...,
        "patch_json": ...,
        "num_calls": int,
        "parse_error": bool,
      }
    """
    num_calls = 0

    # Round 1: base LP JSON
    prompt1 = build_lp_json_prompt(question)
    raw_json1 = caller.call(prompt1, temperature=0.0)
    num_calls += 1

    base_lp = parse_lp_json_to_schema(raw_json1)
    if base_lp is None:
        # Can't parse base; we cannot patch
        return {
            "status": "BASE_PARSE_ERROR",
            "obj_value": None,
            "solution": None,
            "lp_json_round1": raw_json1,
            "patch_json": None,
            "num_calls": num_calls,
            "parse_error": True,
        }

    # Round 2: patch JSON
    prompt2 = build_lp_patch_prompt(question, raw_json1)
    raw_patch = caller.call(prompt2, temperature=0.0)
    num_calls += 1

    patch_dict = parse_patch_json(raw_patch)
    if patch_dict is None:
        # If patch fails, just solve base LP
        refined_lp = base_lp
    else:
        refined_lp = apply_lp_patch(base_lp, patch_dict)

    result = solve_lp_with_gurobi(refined_lp)

    return {
        "status": result["status"],
        "obj_value": result["obj_value"],
        "solution": result["solution"],
        "lp_json_round1": raw_json1,
        "patch_json": raw_patch,
        "num_calls": num_calls,
        "parse_error": False,
    }


# ==========================================
# 8. Evaluation loop on NL4OPT
# ==========================================

def evaluate_nl4opt_lp_gurobi_selfcheck_patch(
    n_examples: Optional[int] = None,
    random_state: int = 0,
    tol: float = 1e-3,
):
    """
    Run GPT-5.1 LP+Gurobi with *conservative* patch-style self-check on NL4OPT.

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
        result = strategy_lp_gurobi_selfcheck_patch(caller, q)
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
            "parse_error": result["parse_error"],
        })

    dfres = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out_path = (
        "results/nl4opt_gpt5_lp_gurobi_selfcheck_patch_full.csv"
        if n_examples is None
        else "results/nl4opt_gpt5_lp_gurobi_selfcheck_patch_sample.csv"
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
        print("\n=== Summary (LP + Gurobi + Self-Check PATCH, numeric gt>0 & OPTIMAL) ===")
        for k, v in summary.items():
            print(f"{k:16s} = {v}")
    else:
        print("\n[warn] No numeric cases with status=OPTIMAL to summarize.")


if __name__ == "__main__":
    # For quick tests: n_examples=10 or 20
    # For full run: n_examples=None
    evaluate_nl4opt_lp_gurobi_selfcheck_patch(
        n_examples=None,   # full numeric subset
        random_state=0,
        tol=1e-3,
    )
