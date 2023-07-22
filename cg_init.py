"""
utilities for initialize first columns in CG framework
"""
from coptpy import COPT

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


def update_edge_capacity(self, cus_idx, used):
    if cus_idx == 0:
        return
    customer_before = self.customer_list[cus_idx - 1]
    for k, v in self.columns_helpers[customer_before]["sku_flow_sum"].items():
        used[k] = used.get(k, 0) + v.getValue()


def update_warehouse_capacity(self, cus_idx, used):
    if cus_idx == 0:
        return
    customer_before = self.customer_list[cus_idx - 1]
    for k, v in self.columns_helpers[customer_before]["sku_inventory_sum"].items():
        used[k] = used.get(k, 0) + v.getValue()


def update_plant_capacity(self, cus_idx, used):
    if cus_idx == 0:
        return

    customer_before = self.customer_list[cus_idx - 1]
    for k, v in self.columns_helpers[customer_before]["sku_production_sum"].items():
        used[k] = used.get(k, 0) + v.getValue()


def primal_sweeping_method(self, sort_method=sorted):
    """
    :param self:
    :param sort_method:
        a sort method to do sweeping, default to id,
        one can use, e.g., a Lagrangian heuristic by sorting the dual price.
    :return:
    """
    ec, pc, wc = {}, {}, {}

    sequence = sort_method(range(self.customer_list.__len__()))

    for col_ind in sequence:
        _this_customer = self.customer_list[col_ind]
        update_edge_capacity(self, col_ind, ec)
        update_plant_capacity(self, col_ind, wc)
        update_warehouse_capacity(self, col_ind, pc)

        # reset capacity constraints
        oracle: dnp_model.DNP = self.oracles[_this_customer]
        oracle.del_constr_capacity()
        (
            oracle.used_edge_capacity,
            oracle.used_plant_capacity,
            oracle.used_warehouse_capacity,
        ) = (ec, pc, wc)
        for t in range(oracle.T):
            oracle.add_constr_holding_capacity(t)
            oracle.add_constr_production_capacity(t)
            oracle.add_constr_transportation_capacity(t)
        self.subproblem(_this_customer, col_ind)

        # then reset column constraints
        oracle.del_constr_capacity()
        (
            oracle.used_edge_capacity,
            oracle.used_plant_capacity,
            oracle.used_warehouse_capacity,
        ) = ({}, {}, {})
