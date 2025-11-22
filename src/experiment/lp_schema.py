# src/experiment/lp_schema.py

from dataclasses import dataclass, field
from typing import Dict, List, Literal


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
