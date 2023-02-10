"""
customer covering
"""
import json
import pickle

import coptpy
import numpy as np
from coptpy import *

from utils import *


@timer
def addvars_xw(self, model):
    x_w = model.addVars(
        self.data.X_W
        if DEFAULT_ALG_PARAMS.phase2_use_full_model
        else self.data.warehouse_routes,
        self.data.T,
        nameprefix="x_w",
        vtype=COPT.CONTINUOUS,
    )  # 仓库->仓库 线路运输量
    return x_w


@timer
def addvars_xc(self, model):
    """
    add xc based on different settings.
    :param self:
    :param model:
    :return:
    """
    if DEFAULT_ALG_PARAMS.phase2_use_full_model == 1:
        x_c = model.addVars(
            self.data.X_C, self.data.T, nameprefix="x_c", vtype=COPT.CONTINUOUS
        )  # 仓库->客户 线路运输量
    elif DEFAULT_ALG_PARAMS.phase2_use_full_model == 2:
        print("using greedy selections")
        w2c_heur = self.routing_heuristics(DEFAULT_ALG_PARAMS.phase2_greedy_range)
        w2c_routes = set(w2c_heur).union(self.data.available_routes)
        print(
            f"routing expansion:#{DEFAULT_ALG_PARAMS.phase2_greedy_range}: {len(self.data.available_routes)} => {len(w2c_routes)} from {len(w2c_heur)} "
        )
        w2ct_list = [
            (i, k, s, t)
            for (i, k, s) in w2c_routes
            for t in self.data.T
            if (k, s, t) in self.data.cus_demand_periodly
        ]
        x_c = model.addVars(w2ct_list, nameprefix="x_c", vtype=COPT.CONTINUOUS)
    else:
        x_c = model.addVars(
            self.data.available_routes,
            self.data.T,
            nameprefix="x_c",
            vtype=COPT.CONTINUOUS,
        )  # 仓库->客户 线路运输量

    return x_c
