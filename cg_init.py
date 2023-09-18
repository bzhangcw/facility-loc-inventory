"""
utilities for initialize first columns in CG framework
"""
from typing import Dict, Tuple, Any

from coptpy import COPT

import cg_col_helper
import dnp_model
from entity import Customer, Plant
from utils import get_in_edges, get_out_edges
import ray
import time


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

    # dual_index["weights_sum"] = self.dual_index["weights_sum"]

    # for customer in self.customer_list:
    #     index = dual_index["weights_sum"][customer]
    #     lp_dual[index] = dual_vars[index]

    # return lp_dual, dual_index
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
    # for t in range(self.oracles[customer].T):
    #     for k, v in self.columns_helpers[customer]["sku_flow_sum"][t].items():
    #         used[t][k] = used.get(t).get(k, 0) + (
    #             v.getValue() if type(v) is not float else v
    #         )

    for t in range(self.arg.T):
        for k, v in columns["sku_flow_sum"][t].items():
            used[t][k] = used.get(t).get(k, 0) + v

    # for debug
    # print(customer.idx, " ec after:")
    # for t in range(self.arg.T):
    #     print("t: ", t, " at ", time.time())
    #     for k, v in used[t].items():
    #         print(k, ":", v)
    # for debug


def update_warehouse_capacity(self, customer, used, columns):
    # for t in range(self.oracles[customer].T):
    #     for k, v in self.columns_helpers[customer]["sku_inventory_sum"][t].items():
    #         used[t][k] = used.get(t).get(k, 0) + (
    #             v.getValue() if type(v) is not float else v
    #         )

    for t in range(self.arg.T):
        for k, v in columns["sku_inventory_sum"][t].items():
            used[t][k] = used.get(t).get(k, 0) + v


def update_plant_capacity(self, customer, used, columns):
    # for t in range(self.oracles[customer].T):
    #     for k, v in self.columns_helpers[customer]["sku_production_sum"][t].items():
    #         used[t][k] = used.get(t).get(k, 0) + (
    #             v.getValue() if type(v) is not float else v
    #         )

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
    reset = {t: {} for t in range(self.arg.T)}
    sequence = sort_method(range(self.customer_list.__len__()))

    for col_ind in sequence:
        # # for debug
        # print("sweeping column: ", col_ind, ",", self.customer_list[col_ind].idx)
        # print("ec:")
        # for t in range(self.arg.T):
        #     print(t)
        #     for k, v in ec[t].items():
        #         print(k, ":", v)
        # print("pc:")
        # for t in range(self.arg.T):
        #     print(t)
        #     for k, v in pc[t].items():
        #         print(k, ":", v)
        # print("wc:")
        # for t in range(self.arg.T):
        #     print(t)
        #     for k, v in wc[t].items():
        #         print(k, ":", v)
        # # for debug

        _this_customer: Customer = self.customer_list[col_ind]

        # set capacity constraints based on what has been used
        oracle: dnp_model.DNP = self.oracles[_this_customer]

        if self.init_ray:
            ray.get(oracle.del_constr_capacity.remote())
            ray.get(oracle.update_constr_capacity.remote(ec, pc, wc))
        else:
            oracle.del_constr_capacity()

            (
                oracle.used_edge_capacity,
                oracle.used_plant_capacity,
                oracle.used_warehouse_capacity,
            ) = (
                ec,
                pc,
                wc,
            )
            # why this works, because of the typedef in line 96?
        # for t in range(oracle.T):
        for t in range(self.arg.T):
            if self.init_ray:
                ray.get(oracle.add_constr_holding_capacity.remote(t))
                ray.get(oracle.add_constr_production_capacity.remote(t))
                ray.get(oracle.add_constr_transportation_capacity.remote(t))
                # ray.get(oracle.add_constr_transportation_capacity.remote(t, True))

            else:
                oracle.add_constr_holding_capacity(t)
                oracle.add_constr_production_capacity(t)
                oracle.add_constr_transportation_capacity(t)

        self.subproblem(_this_customer, col_ind)

        if self.init_ray:
            # columns = ray.get(self.oracles[_this_customer].query_columns.remote())
            columns = ray.get(oracle.query_columns.remote())

        else:
            # columns = self.oracles[_this_customer].query_columns()
            columns = oracle.query_columns()

        update_edge_capacity(self, _this_customer, ec, columns)
        update_plant_capacity(self, _this_customer, pc, columns)
        update_warehouse_capacity(self, _this_customer, wc, columns)

        # then reset column constraints

        if self.init_ray:
            ray.get(oracle.del_constr_capacity.remote())
            ray.get(oracle.update_constr_capacity.remote(reset, reset, reset))
        else:
            oracle.del_constr_capacity()

            (
                oracle.used_edge_capacity,
                oracle.used_plant_capacity,
                oracle.used_warehouse_capacity,
            ) = (reset, reset, reset)

        # for t in range(oracle.T):

        for t in range(self.arg.T):
            if self.init_ray:
                ray.get(oracle.add_constr_holding_capacity.remote(t))
                ray.get(oracle.add_constr_production_capacity.remote(t))
                ray.get(oracle.add_constr_transportation_capacity.remote(t))
            else:
                oracle.add_constr_holding_capacity(t)
                oracle.add_constr_production_capacity(t)
                oracle.add_constr_transportation_capacity(t)
