"""
utilities for initialize first columns in CG framework
"""
import time
from typing import Any, Dict, Tuple

import ray
from coptpy import COPT

import hyper_cg.cg_col_helper as cg_col_helper
import dnp_model
from entity import Customer, Plant
from utils import get_in_edges, get_out_edges


def init_cols_from_dual_feas_sol(self, dual_vars):
    full_lp_relaxation = dnp_model.DNP(self.arg, self.full_network, cus_num=472)
    full_lp_relaxation.modeling()
    # get the LP relaxation
    vars = full_lp_relaxation.model.getVars()
    binary_vars_index = []
    for v in vars:
        if v.getType() == COPT.BINARY:
            binary_vars_index.append(v.getIdx())
            v.setType(COPT.CONTINUOUS)
    ######################
    full_lp_relaxation.model.setParam("Logging", 1)
    full_lp_relaxation.solve()
    lp_dual = full_lp_relaxation.model.getDuals()
    dual_index = full_lp_relaxation.dual_index_for_RMP

    init_dual = dual_vars.copy()
    for edge in self.dual_index["transportation_capacity"].keys():
        init_dual[self.dual_index["transportation_capacity"][edge]] = lp_dual[
            dual_index["transportation_capacity"][edge]
        ]

    for node in self.dual_index["node_capacity"].keys():
        init_dual[self.dual_index["node_capacity"][node]] = lp_dual[
            dual_index["node_capacity"][node]
        ]

    init_dual_index = self.dual_index.copy()
    return init_dual, init_dual_index


def update_edge_capacity(self, customer, used, columns):
    for t in range(self.arg.T):
        for k, v in columns["sku_flow_sum"][t].items():
            used[t][k] = used.get(t).get(k, 0) + v


def update_warehouse_capacity(self, customer, used, columns):
    for t in range(self.arg.T):
        for k, v in columns["sku_inventory_sum"][t].items():
            used[t][k] = used.get(t).get(k, 0) + v


def update_plant_capacity(self, customer, used, columns):
    for t in range(self.arg.T):
        for k, v in columns["sku_production_sum"][t].items():
            used[t][k] = used.get(t).get(k, 0) + v


def primal_sweeping_method(self, sort_method=sorted):
    """
    :param self:
    :param sort_method:
        a sort method to do sweeping, default to id,
        one can use, e.g., a Lagrangian heuristic by sorting the dual price.
    :return:
    """
    ec, pc, wc = (
        {t: {} for t in range(self.arg.T)},
        {t: {} for t in range(self.arg.T)},
        {t: {} for t in range(self.arg.T)},
    )
    sequence = sort_method(range(self.customer_list.__len__()))

    for col_ind in sequence:
        _this_customer: Customer = self.customer_list[col_ind]

        # set capacity constraints based on what has been used

        if self.init_ray:
            oracle = self.worker_list[self.worker_cus_dict[_this_customer]]
            ray.get(oracle.del_constr_capacity.remote(_this_customer))
            ray.get(oracle.update_constr_capacity.remote(_this_customer, ec, pc, wc))
        else:
            oracle: dnp_model.DNP = self.oracles[_this_customer]

        self.subproblem(_this_customer, col_ind)

        if self.init_ray:
            columns = ray.get(oracle.query_columns.remote(_this_customer))

        else:
            columns = oracle.query_columns()
