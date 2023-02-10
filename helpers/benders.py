"""
implement the benders.
"""

import json
import pickle

import coptpy
import numpy as np
from coptpy import *

from utils import *


def set_state(self):
    if DEFAULT_ALG_PARAMS.phase2_use_full_model == 1:
        # todo, this is not realistic.
        # since |X_W|, |X_C| is too large.
        print("using full model...")
        model: coptpy.Model = self.model
        z_p, z_l, x_p, x_w, x_c, inv, surplus_inv, tmp, inv_gap, *_ = self.variables

        # Benders state variable
        x_w_rhs = {(*k, t): 0 if k not in set(self.data.warehouse_routes) else 1e4 for k in self.data.X_W for t in
                   self.data.T}
        x_c_rhs = {(*k, t): 0 if k not in set(self.data.available_routes) else 1e4 for k in self.data.X_C for t in
                   self.data.T}

        if self.state_constrs.__len__() == 0:
            bound_constr_w = model.addConstrs((x_w[k] <= x_w_rhs[k] for k in self.data.X_W))
            bound_constr_c = model.addConstrs((x_c[k] <= x_c_rhs[k] for k in self.data.X_C))
            self.state_constrs = (bound_constr_w, bound_constr_c)
        else:
            bound_constr_w, bound_constr_c = self.state_constrs
            model.setInfo(COPT.Info.UB, bound_constr_w, x_w_rhs)
            model.setInfo(COPT.Info.UB, bound_constr_c, x_c_rhs)