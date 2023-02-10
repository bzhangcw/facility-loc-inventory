"""
primal integral fixing heurstics
"""
import json
import pickle

import coptpy
import numpy as np
from coptpy import *

from utils import *


def query_invalid_qty(self, zl):
    invalid = {
        (p, l, s, t): v / self.data.line_prod_mpq[p, l, s]
        for (p, l, s, t), v in zl.items()
        if 0 < v < self.data.line_prod_mpq[p, l, s]
    }
    empty = {(p, l, s, t): v for (p, l, s, t), v in zl.items() if v == 0}
    return invalid, empty


def clear_integer_fixings(self, z_l):
    if len(self.integer_fix_constrs) > 0:
        self.model.remove(self.integer_fix_constrs)
    for _, v in z_l.items():
        v.setInfo(COPT.Info.LB, 0)
        v.setInfo(COPT.Info.UB, COPT.INFINITY)


@timer
def qty_fix_heuristic(self):
    model: coptpy.Model = self.model
    _logs = []
    z_p, z_l, x_p, x_w, x_c, *_ = self.variables
    # --- forward iteration ---
    # try to increase production qty
    zl = model.getInfo(COPT.Info.Value, z_l)
    zl_invalid, zl_empty = query_invalid_qty(self, zl)
    _logs.append(f"original, invalid size {len(zl_invalid)}")

    for (p, l, s, t), v in zl_empty.items():
        z_l[p, l, s, t].setInfo(COPT.Info.UB, 0)
    # if v is large, we make it as large as possible (up to min_prod).
    for (p, l, s, t), v in zl_invalid.items():
        z_l[p, l, s, t].setInfo(COPT.Info.UB, self.data.line_prod_mpq[p, l, s])

    model.setObjective(
        self.obj_expr
        - 1e3 * quicksum(z_l[p, l, s, t] * v for (p, l, s, t), v in zl_invalid.items()),
        COPT.MINIMIZE,
    )
    if DEFAULT_ALG_PARAMS.phase2_qty_heur_reset:
        model.reset()
    model.solve()

    # --- backward iteration ---
    # - set all invalid to 0
    # - set others x >= min_prod
    zl = model.getInfo(COPT.Info.Value, z_l)
    zl_invalid, zl_empty = query_invalid_qty(self, zl)
    _logs.append(f"after forward, invalid size {len(zl_invalid)}")

    for (p, l, s, t), v in z_l.items():
        if (p, l, s, t) in zl_invalid:
            v.setInfo(COPT.Info.UB, 0)
        elif (p, l, s, t) in zl_empty:
            v.setInfo(COPT.Info.UB, 0)
        elif self.data.line_prod_mpq[p, l, s] > 0:
            v.setInfo(COPT.Info.UB, COPT.INFINITY)
            v.setInfo(COPT.Info.LB, self.data.line_prod_mpq[p, l, s])
        else:
            pass

    model.setObjective(self.obj_expr, COPT.MINIMIZE)
    if DEFAULT_ALG_PARAMS.phase2_qty_heur_reset:
        model.reset()
    model.solve()
    zl = model.getInfo(COPT.Info.Value, z_l)
    zl_invalid, zl_empty = query_invalid_qty(self, zl)
    _logs.append(f"after backward, invalid size {len(zl_invalid)}")
    print("--- primal fixing heuristic ---")
    [print(f"- {l}") for l in _logs]
