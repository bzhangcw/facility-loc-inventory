import argparse
import logging
import os
from typing import List
import numpy as np
import coptpy as cp
import pandas as pd
import networkx as nx
from coptpy import COPT

import const
from entity import SKU
from read_data import read_data
from utils import get_in_edges, get_out_edges, logger


class DNP:
    """
    this is a class for dynamic network flow (DNP)
    """

    def __init__(
        self,
        arg: argparse.Namespace,
        network: nx.DiGraph,
        full_sku_list: List[SKU] = None,
        env_name: str = "DNP_env",
        model_name: str = "DNP",
        open_relationship: bool = True,
        capacity: bool = True,
        feasibility: bool = False,
        cus_num: int = 1,
        env=None,
    ) -> None:
        self.arg = arg
        self.T = arg.T
        self.network = network
        self.full_sku_list = (
            full_sku_list
            if full_sku_list is not None
            else self.network.graph["sku_list"]
        )
        if env is None:
            self.env = cp.Envr(env_name)
        else:
            self.env = env
        self.model = self.env.createModel(model_name)
        self.model.setParam(COPT.Param.Logging, 0)

        self.vars = dict()  # variables
        self.constrs = dict()  # constraints
        self.obj = dict()  # objective

        self.open_relationship = open_relationship
        self.capacity = capacity
        self.feasibility = feasibility
        self.original_obj = 0.0
        self.cus_num = cus_num
        self.total_cus_num = arg.total_cus_num
        self.cus_ratio = min(self.cus_num / self.total_cus_num, 1.0)
        # self.cus_ratio = 1.0
        self.var_idx = None
        self.dual_index_for_RMP = {
            "transportation_capacity": dict(),
            "node_capacity": dict(),
            # "weights_sum": dict(),
        }

    def modeling(self):
        """
        build DNP model
        """

        # print("add variables ...")
        self.add_vars()

        # print("add constraints ...")
        self.add_constraints()

        # print("set objective ...")

        self.set_objective()

    def add_vars(self):
        """
        add variables
        """

        self.var_types = {
            "sku_flow": {
                "lb": 0,
                "ub": COPT.INFINITY,
                "vtype": COPT.CONTINUOUS,
                "nameprefix": "w",
                "index": "(t, edge, k)",
            },
            "sku_production": {
                "lb": 0,
                "ub": COPT.INFINITY,
                "vtype": COPT.CONTINUOUS,
                "nameprefix": "x",
                "index": "(t, plant, k)",
            },
            "sku_inventory": {
                "lb": -COPT.INFINITY if self.arg.backorder is True else 0,
                "ub": COPT.INFINITY,
                "vtype": COPT.CONTINUOUS,
                "nameprefix": "I",
                "index": "(t, warehouse, k)",
            },
            "sku_demand_slack": {
                "lb": 0,
                "ub": [],  # TBD
                "vtype": COPT.CONTINUOUS,
                "nameprefix": "s",
                "index": "(t, warehouse with demand / customer, k)",
            },
        }

        if self.open_relationship:
            self.var_types["select_edge"] = {
                "lb": 0,
                "ub": 1,
                "vtype": COPT.BINARY,
                "nameprefix": "p",
                "index": "(t, edge)",
            }
            self.var_types["sku_select_edge"] = {
                "lb": 0,
                "ub": 1,
                "vtype": COPT.BINARY,
                "nameprefix": "pk",
                "index": "(t, edge, k)",
            }
            self.var_types["open"] = {
                "lb": 0,
                "ub": 1,
                "vtype": COPT.BINARY,
                "nameprefix": "y",
                "index": "(t, node)",
            }
            self.var_types["sku_open"] = {
                "lb": 0,
                "ub": 1,
                "vtype": COPT.BINARY,
                "nameprefix": "yk",
                "index": "(t, plant, k)",
            }

        # generate index tuple
        idx = dict()
        for vt in self.var_types.keys():
            idx[vt] = list()

        # periods
        for t in range(self.T):
            # edges
            for e in self.network.edges:
                edge = self.network.edges[e]["object"]

                # select edge (i,j) at t
                if self.open_relationship:
                    idx["select_edge"].append((t, edge))

                sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    # sku k select edge (i,j) at t
                    if self.open_relationship:
                        idx["sku_select_edge"].append((t, edge, k))
                    # flow of sku k on edge (i,j) at t
                    idx["sku_flow"].append((t, edge, k))

            # nodes
            for node in self.network.nodes:
                # open node i at t
                if self.open_relationship:
                    idx["open"].append((t, node))

                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    if node.type == const.PLANT:
                        # sku k produced on node i at t
                        if self.open_relationship:
                            idx["sku_open"].append((t, node, k))
                        # amount of sku k produced on node i at t
                        idx["sku_production"].append((t, node, k))
                    elif node.type == const.WAREHOUSE:
                        # amount of sku k stored on node i at t
                        idx["sku_inventory"].append((t, node, k))
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            # print(node.demand)
                            idx["sku_demand_slack"].append((t, node, k))
                            self.var_types["sku_demand_slack"]["ub"].append(
                                node.demand[(t, k)]
                            )
                    elif node.type == const.CUSTOMER:
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            idx["sku_demand_slack"].append((t, node, k))
                            self.var_types["sku_demand_slack"]["ub"].append(
                                node.demand[t, k]
                            )
        # for initializaiton in CG
        self.var_idx = {}
        for var in idx.keys():
            self.var_idx[var] = {key: 0 for key in idx[var]}
        ##########################
        # add variables
        for vt, param in self.var_types.items():
            # print(f"  - {vt}")
            self.vars[vt] = self.model.addVars(
                idx[vt],
                lb=param["lb"],
                ub=param["ub"],
                vtype=param["vtype"],
                nameprefix=f"{param['nameprefix']}_",
            )

    def add_constraints(self):
        self.constr_types = {
            "flow_conservation": {"index": "(t, node, k)"},
            "open_relationship": {
                "select_edge": {"index": "(t, edge, node)"},
                "sku_select_edge": {"index": "(t, edge, k)"},
                "open": {"index": "(t, warehouse with demand / customer)"},
                "sku_open": {"index": "(t, node, k)"},
            },
            "transportation_capacity": {"index": "(t, edge)"},
            "transportation_variable_lb": {"index": "(t, edge)"},
            "production_capacity": {"index": "(t, node)"},
            "holding_capacity": {"index": "(t, node)"},
            "holding_capacity_back_order": {"index": "(t, node)"},
        }

        for constr in self.constr_types.keys():
            self.constrs[constr] = dict()

        for constr in self.constr_types["open_relationship"].keys():
            self.constrs["open_relationship"][constr] = dict()

        # for t in tqdm(range(self.T)):
        for t in range(self.T):
            # initial status and flow conservation
            self.add_constr_flow_conservation(t)

            if self.open_relationship:
                # node status and open relationship
                self.add_constr_open_relationship(t)

            if self.capacity:
                # transportation/production/holding capacity
                self.add_constr_transportation_capacity(t)
                self.add_constr_production_capacity(t)
                self.add_constr_holding_capacity(t)

            # self.add_constr_inventory_capacity_for_RMP(t)
            # self.add_constr_transportation_capacity(t)
            # self.add_constr_production_capacity(t)

    def del_constr_for_RMP(self):
        # for t in tqdm(range(self.T)):
        for t in range(self.T):
            # self.del_constr_inventory_capacity_for_RMP(t)
            self.del_constr_transportation_capacity(t)
            self.del_constr_production_capacity(t)
            self.del_constr_holding_capacity(t)

    def del_constr_transportation_capacity(self, t: int):
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            self.constrs["transportation_capacity"][(t, edge)].remove()

    def del_constr_production_capacity(self, t: int):
        for node in self.network.nodes:
            if node.type == const.PLANT:
                self.constrs["production_capacity"][(t, node)].remove()

    def del_constr_holding_capacity(self, t: int):
        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                self.constrs["holding_capacity"][(t, node)].remove()

    def del_constr_inventory_capacity_for_RMP(self, t: int):
        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                self.constrs["holding_capacity"][(t, node)].remove()

                # if self.arg.backorder is True:
                # self.constrs['holding_capacity_back_order'][(t, node)].remove()

        return

    def add_constr_flow_conservation(self, t: int):
        for node in self.network.nodes:
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            for k in sku_list:
                in_edges = get_in_edges(self.network, node)
                out_edges = get_out_edges(self.network, node)

                constr_name = f"flow_conservation_{node.idx}_{k.idx}"

                if node.type == const.PLANT:
                    constr = self.model.addConstr(
                        self.vars["sku_production"][t, node, k]
                        - self.vars["sku_flow"].sum(t, out_edges, k)
                        == 0,
                        name=constr_name,
                    )
                elif node.type == const.WAREHOUSE:
                    fulfilled_demand = 0
                    if node.has_demand(t, k):
                        fulfilled_demand = (
                            node.demand[t, k]
                            - self.vars["sku_demand_slack"][t, node, k]
                        )

                    last_period_inventory = 0.0
                    # if t == 0:
                    #     if node.initial_inventory is not None:
                    #         if self.open_relationship:
                    #             self.model.addConstr(
                    #                 self.vars["open"][self.T - 1, node] == 1
                    #             )
                    #         last_period_inventory = (
                    #             node.initial_inventory[k]
                    #             if k in node.initial_inventory
                    #             else 0.0
                    #         )
                    #     else:
                    #         last_period_inventory = 0.0
                    # else:
                    #     last_period_inventory = self.vars["sku_inventory"][
                    #         t - 1, node, k
                    #     ]
                    # last_period_inventory *= self.cus_ratio

                    constr = self.model.addConstr(
                        self.vars["sku_flow"].sum(t, in_edges, k)
                        + last_period_inventory
                        - self.vars["sku_flow"].sum(t, out_edges, k)
                        - fulfilled_demand
                        == self.vars["sku_inventory"][t, node, k],
                        name=constr_name,
                    )

                elif node.type == const.CUSTOMER:
                    fulfilled_demand = (
                        node.demand[t, k] - self.vars["sku_demand_slack"][t, node, k]
                    )
                    constr = self.model.addConstr(
                        self.vars["sku_flow"].sum(t, in_edges, k) - fulfilled_demand
                        == 0,
                        name=constr_name,
                    )

                self.constrs["flow_conservation"][(t, node, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]

            sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

            constr = self.model.addConstr(
                self.vars["select_edge"][t, edge] <= self.vars["open"][t, edge.start]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.start)
            ] = constr

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.end)
            ] = constr = self.model.addConstr(
                self.vars["select_edge"][t, edge] <= self.vars["open"][t, edge.end]
            )

            for k in sku_list:
                constr = self.model.addConstr(
                    self.vars["sku_select_edge"][t, edge, k]
                    <= self.vars["select_edge"][t, edge]
                )
                self.constrs["open_relationship"]["sku_select_edge"][
                    (t, edge, k)
                ] = constr

        for node in self.network.nodes:
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            if (
                node.type == const.WAREHOUSE
                and node.has_demand(t)
                and len(node.demand_sku[t]) > 0
            ):
                constr = self.model.addConstr(self.vars["open"][t, node] == 1)
                self.constrs["open_relationship"]["open"][(t, node)] = constr
            elif node.type == const.CUSTOMER:
                constr = self.model.addConstr(self.vars["open"][t, node] == 1)
                self.constrs["open_relationship"]["open"][(t, node)] = constr

            for k in sku_list:
                if node.type == const.PLANT:
                    constr = self.model.addConstr(
                        self.vars["sku_open"][t, node, k] <= self.vars["open"][t, node]
                    )

                self.constrs["open_relationship"]["sku_open"][(t, node, k)] = constr

        return

    def add_constr_transportation_capacity(self, t: int):
        index = 0
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]

            flow_sum = self.vars["sku_flow"].sum(t, edge, "*")

            # variable lower bound
            if edge.variable_lb < np.inf:
                self.constrs["transportation_variable_lb"][
                    (t, edge)
                ] = self.model.addConstr(
                    flow_sum >= edge.variable_lb * self.vars["select_edge"][t, edge]
                )
                self.constrs["transportation_capacity"][
                    (t, edge)
                ] = self.model.addConstr(
                    # flow_sum <= edge.capacity * self.vars["select_edge"][t, edge]
                    # multiply by customer ratio to force feasibliity of first column of cg
                    flow_sum
                    <= edge.capacity
                    * self.vars["select_edge"][t, edge]
                    * self.cus_ratio
                )
            else:
                self.constrs["transportation_capacity"][
                    (t, edge)
                ] = self.model.addConstr(
                    flow_sum
                    <= edge.capacity
                    * self.vars["select_edge"][t, edge]
                    * self.cus_ratio
                )

            self.dual_index_for_RMP["transportation_capacity"][edge] = index
            index += 1

        return

    def add_constr_production_capacity(self, t: int):
        index = (
            list(self.dual_index_for_RMP["transportation_capacity"].values())[-1] + 1
        )

        for node in self.network.nodes:
            if node.type == const.PLANT:
                # constr = self.model.addConstr(self.vars['sku_production'].sum(t, node, '*') <= node.production_capacity * self.vars['open'][t, node])
                constr = self.model.addConstr(
                    self.vars["sku_production"].sum(t, node, "*")
                    # <= node.production_capacity
                    # multiply by customer ratio to force feasibliity of first column of cg
                    <= node.production_capacity * self.cus_ratio
                )

                self.constrs["production_capacity"][(t, node)] = constr

            self.dual_index_for_RMP["node_capacity"][node] = index
            index += 1

        return

    def add_constr_inventory_capacity_for_RMP(self, t: int):
        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                constr = self.model.addConstr(
                    self.vars["sku_inventory"].sum(t, node, "*")
                    <= node.inventory_capacity / self.cus_num
                )
                self.constrs["holding_capacity"][(t, node)] = constr

                if self.arg.backorder is True:
                    constr = self.model.addConstr(
                        self.vars["sku_inventory"].sum(t, node, "*") >= -self.arg.M
                    )
                    self.constrs["holding_capacity_back_order"][(t, node)] = constr

        return

    def add_constr_holding_capacity(self, t: int):
        index = list(self.dual_index_for_RMP["node_capacity"].values())[-1] + 1

        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                constr = self.model.addConstr(
                    self.vars["sku_inventory"].sum(t, node, "*")
                    # <= node.inventory_capacity * self.vars["open"][t, node]
                    # multiply by customer ratio to force feasibliity of first column of cg
                    <= node.inventory_capacity
                    * self.vars["open"][t, node]
                    * self.cus_ratio
                )
                self.constrs["holding_capacity"][(t, node)] = constr

                if self.arg.backorder is True:
                    self.model.addConstr(
                        self.vars["sku_inventory"].sum(t, node, "*")
                        >= -self.arg.M * self.vars["open"][t, node]
                    )

            self.dual_index_for_RMP["node_capacity"][node] = index
            index += 1

        return

    def get_original_objective(self):
        """
        Get the original objective value
        """
        obj = 0.0

        # for t in tqdm(range(self.T)):
        for t in range(self.T):
            obj = obj + self.cal_sku_producing_cost(t)
            obj = obj + self.cal_sku_holding_cost(t)

            if self.arg.backorder is True:
                obj = obj + self.cal_sku_backorder_cost(t)

            obj = obj + self.cal_sku_transportation_cost(t)

            obj = obj + self.cal_sku_unfulfill_demand_cost(t)

        obj = obj + self.cal_fixed_node_cost()
        obj = obj + self.cal_fixed_edge_cost()

        if self.arg.end_inventory:
            obj = obj + self.cal_end_inventory_bias_cost()

        return obj

    def extra_objective(self, dualvar, dual_index):
        obj = 0.0
        for edge in dual_index["transportation_capacity"].keys():
            obj -= dualvar[dual_index["transportation_capacity"][edge]] * self.vars[
                "sku_flow"
            ].sum(0, edge, "*")

        for node in dual_index["node_capacity"].keys():
            if node.type == const.PLANT:
                obj -= dualvar[dual_index["node_capacity"][node]] * self.vars[
                    "sku_production"
                ].sum(0, node, "*")
            elif node.type == const.WAREHOUSE:
                obj -= dualvar[dual_index["node_capacity"][node]] * self.vars[
                    "sku_inventory"
                ].sum(0, node, "*")
            else:
                continue

        for node in dual_index["weights_sum"].keys():
            obj -= dualvar[dual_index["weights_sum"][node]]

        return obj

    def update_objective(self, dualvar, dual_index):
        """
        Use dual variables to calculate the reduced cost
        """
        self.original_obj = self.get_original_objective()
        obj = self.original_obj + self.extra_objective(dualvar, dual_index)
        self.model.setObjective(obj, sense=COPT.MINIMIZE)

    def set_objective(self):
        self.obj_types = {
            "sku_producing_cost": {"index": "(t, plant, k)"},
            "sku_holding_cost": {"index": "(t, warehouse, k)"},
            "sku_backorder_cost": {"index": "(t, warehouse, k)"},
            "sku_transportation_cost": {"index": "(t, edge)"},
            "unfulfill_demand_cost": {
                "index": "(t, warehouse with demand / customer, k)"
            },
            "fixed_node_cost": {"index": "(t, plant / warehouse, k)"},
            "fixed_edge_cost": {"index": "(t, edge, k)"},
            "end_inventory_cost": {"index": "(node, k)"},
        }

        for obj in self.obj_types.keys():
            self.obj[obj] = dict()

        obj = 0.0
        self.original_obj = self.get_original_objective()

        if not self.feasibility:
            obj = self.original_obj

        self.model.setObjective(obj, sense=COPT.MINIMIZE)

        return

    def cal_sku_producing_cost(self, t: int):
        producing_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.PLANT:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    node_sku_producing_cost = 0.0

                    if (
                        node.production_sku_fixed_cost is not None
                        and self.open_relationship
                    ):
                        node_sku_producing_cost = (
                            node_sku_producing_cost
                            + node.production_sku_fixed_cost[k]
                            * self.vars["sku_open"][t, node, k]
                        )
                    if node.production_sku_unit_cost is not None:
                        node_sku_producing_cost = (
                            node_sku_producing_cost
                            + node.production_sku_unit_cost[k]
                            * self.vars["sku_production"][t, node, k]
                        )

                    producing_cost = producing_cost + node_sku_producing_cost

                    self.obj["sku_producing_cost"][
                        (t, node, k)
                    ] = node_sku_producing_cost

        return producing_cost

    def cal_sku_holding_cost(self, t: int):
        holding_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    node_sku_holding_cost = 0.0

                    if node.holding_sku_unit_cost is not None:
                        holding_sku_unit_cost = node.holding_sku_unit_cost[k]
                    else:
                        holding_sku_unit_cost = self.arg.holding_sku_unit_cost

                    # I_hat = max(I, 0)
                    I_hat = self.model.addVar(name=f"I_hat_({t},_{node},_{k})")
                    self.model.addConstr(
                        I_hat >= self.vars["sku_inventory"][t, node, k]
                    )
                    node_sku_holding_cost = (
                        node_sku_holding_cost + holding_sku_unit_cost * I_hat
                    )

                    holding_cost = holding_cost + node_sku_holding_cost

                    self.obj["sku_holding_cost"][(t, node, k)] = node_sku_holding_cost

        return holding_cost

    def cal_sku_backorder_cost(self, t: int):
        backorder_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    node_sku_backorder_cost = 0.0

                    if node.backorder_sku_unit_cost is not None:
                        backorder_sku_unit_cost = node.backorder_sku_unit_cost[k]
                    else:
                        backorder_sku_unit_cost = self.arg.backorder_sku_unit_cost

                    # I_tilde = max(-I, 0)
                    I_tilde = self.model.addVar(name=f"I_tilde_({t},_{node},_{k})")
                    self.model.addConstr(
                        I_tilde >= -self.vars["sku_inventory"][t, node, k]
                    )
                    node_sku_backorder_cost = (
                        node_sku_backorder_cost + backorder_sku_unit_cost * I_tilde
                    )

                    backorder_cost = backorder_cost + node_sku_backorder_cost

                    self.obj["sku_backorder_cost"][
                        (t, node, k)
                    ] = node_sku_backorder_cost

        return backorder_cost

    def cal_sku_transportation_cost(self, t: int):
        transportation_cost = 0.0

        for e in self.network.edges:
            edge = self.network.edges[e]["object"]

            edge_transportation_cost = 0.0

            (
                sku_list_with_fixed_transportation_cost,
                sku_list_with_unit_transportation_cost,
            ) = edge.get_edge_sku_list_with_transportation_cost(t, self.full_sku_list)

            if self.open_relationship:
                for k in sku_list_with_fixed_transportation_cost:
                    if (
                        edge.transportation_sku_fixed_cost is not None
                        and k in edge.transportation_sku_fixed_cost
                    ):
                        edge_transportation_cost = (
                            edge_transportation_cost
                            + edge.transportation_sku_fixed_cost[k]
                            * self.vars["sku_select_edge"][t, edge, k]
                        )

            for k in sku_list_with_unit_transportation_cost:
                if (
                    edge.transportation_sku_unit_cost is not None
                    and k in edge.transportation_sku_unit_cost
                ):
                    transportation_sku_unit_cost = edge.transportation_sku_unit_cost[k]
                else:
                    transportation_sku_unit_cost = self.arg.transportation_sku_unit_cost

                edge_transportation_cost = (
                    edge_transportation_cost
                    + transportation_sku_unit_cost * self.vars["sku_flow"][t, edge, k]
                )

            transportation_cost = transportation_cost + edge_transportation_cost

            self.obj["sku_transportation_cost"][(t, edge)] = edge_transportation_cost

        return transportation_cost

    def cal_sku_unfulfill_demand_cost(self, t: int):
        unfulfill_demand_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.WAREHOUSE or node.type == const.CUSTOMER:
                if node.has_demand(t):
                    for k in node.demand_sku[t]:
                        unfulfill_node_sku_cost = 0.0

                        if node.unfulfill_sku_unit_cost is not None:
                            unfulfill_sku_unit_cost = node.unfulfill_sku_unit_cost[
                                (t, k)
                            ]
                        else:
                            unfulfill_sku_unit_cost = self.arg.unfulfill_sku_unit_cost

                        unfulfill_node_sku_cost = (
                            unfulfill_sku_unit_cost
                            * self.vars["sku_demand_slack"][(t, node, k)]
                        )

                        unfulfill_demand_cost = (
                            unfulfill_demand_cost + unfulfill_node_sku_cost
                        )

                        self.obj["unfulfill_demand_cost"][
                            (t, node, k)
                        ] = unfulfill_node_sku_cost

        return unfulfill_demand_cost

    def cal_fixed_node_cost(self):
        fixed_node_cost = 0.0

        if not self.open_relationship:
            return fixed_node_cost

        for node in self.network.nodes:
            if node.type == const.PLANT:
                this_node_fixed_cost = node.production_fixed_cost
            elif node.type == const.WAREHOUSE:
                this_node_fixed_cost = node.holding_fixed_cost
            elif node.type == const.CUSTOMER:
                break

            y = self.model.addVar(vtype=COPT.BINARY, name=f"y_{node}")
            for t in range(self.T):
                self.model.addConstr(self.vars["open"][(t, node)] <= y)

            node_fixed_node_cost = this_node_fixed_cost * y

            fixed_node_cost = fixed_node_cost + node_fixed_node_cost

            self.obj["fixed_node_cost"][node] = node_fixed_node_cost

        return fixed_node_cost

    def cal_fixed_edge_cost(self):
        fixed_edge_cost = 0.0
        if not self.open_relationship:
            return fixed_edge_cost

        for e in self.network.edges:
            edge = self.network.edges[e]["object"]

            p = self.model.addVar(vtype=COPT.BINARY, name=f"p_{edge}")

            for t in range(self.T):
                self.model.addConstr(self.vars["select_edge"][(t, edge)] <= p)

            edge_fixed_edge_cost = edge.transportation_fixed_cost * p

            fixed_edge_cost = fixed_edge_cost + edge_fixed_edge_cost

            self.obj["fixed_edge_cost"][edge] = edge_fixed_edge_cost

        return fixed_edge_cost

    def cal_end_inventory_bias_cost(self):
        end_inventory_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.WAREHOUSE and node.end_inventory is not None:
                if self.open_relationship:
                    self.model.addConstr(self.vars["open"][self.T - 1, node] == 1)

                for k in node.end_inventory.index:
                    node_end_inventory_cost = 0.0

                    end_inventory_bias = self.model.addVar(name=f"end_I_bias_{node}")

                    self.model.addConstr(
                        end_inventory_bias
                        >= node.end_inventory[k]
                        - self.vars["sku_inventory"].sum(self.T - 1, node, "*")
                    )
                    self.model.addConstr(
                        end_inventory_bias
                        >= self.vars["sku_inventory"].sum(self.T - 1, node, "*")
                        - node.end_inventory[k]
                    )

                    end_inventory_bias_cost = (
                        node.end_inventory_bias_cost
                        if node.end_inventory_bias_cost > 0
                        else self.arg.end_inventory_bias_cost
                    )

                    node_end_inventory_cost = (
                        node_end_inventory_cost
                        + end_inventory_bias_cost * end_inventory_bias
                    )

                    end_inventory_cost = end_inventory_cost + node_end_inventory_cost

                    self.obj["end_inventory_cost"][(node, k)] = node_end_inventory_cost

        return end_inventory_cost

    def solve(self):
        self.model.solve()

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        # node output
        plant_sku_t_production = pd.DataFrame(
            index=range(len(self.vars["sku_production"])),
            columns=["node", "type", "sku", "t", "qty"],
        )
        warehouse_sku_t_storage = pd.DataFrame(
            index=range(len(self.vars["sku_inventory"])),
            columns=["node", "type", "sku", "t", "qty"],
        )
        node_sku_t_demand_slack = pd.DataFrame(
            index=range(len(self.vars["sku_demand_slack"])),
            columns=["node", "type", "sku", "t", "demand", "slack", "fulfill"],
        )

        if self.open_relationship:
            node_t_open = pd.DataFrame(
                index=range(len(self.vars["open"])),
                columns=["node", "type", "t", "open"],
            )

        # edge output
        edge_sku_t_flow = pd.DataFrame(
            index=range(len(self.vars["sku_flow"])),
            columns=["id", "start", "end", "sku", "t", "qty"],
        )

        node_open_index = 0
        plant_index = 0
        warehouse_index = 0
        demand_slack_index = 0
        edge_index = 0

        for node in self.network.nodes:
            for t in range(self.T):
                if self.open_relationship:
                    node_t_open.iloc[node_open_index] = {
                        "node": node.idx,
                        "type": node.type,
                        "t": t,
                        "open": self.vars["open"][(t, node)].x,
                    }
                    node_open_index += 1

                if node.type == const.PLANT:
                    if node.producible_sku is not None:
                        for k in node.producible_sku:
                            if (
                                preserve_zeros
                                or self.vars["sku_production"][(t, node, k)].x != 0
                            ):
                                plant_sku_t_production.iloc[plant_index] = {
                                    "node": node.idx,
                                    "type": node.type,
                                    "sku": k.idx,
                                    "t": t,
                                    "qty": self.vars["sku_production"][(t, node, k)].x,
                                }
                                plant_index += 1

                if node.type == const.WAREHOUSE:
                    sku_list = node.get_node_sku_list(t, self.full_sku_list)
                    for k in sku_list:
                        if (
                            preserve_zeros
                            or self.vars["sku_inventory"][(t, node, k)].x != 0
                        ):
                            warehouse_sku_t_storage.iloc[warehouse_index] = {
                                "node": node.idx,
                                "type": node.type,
                                "sku": k.idx,
                                "t": t,
                                "qty": self.vars["sku_inventory"][(t, node, k)].x,
                            }
                            warehouse_index += 1

            if node.type != const.PLANT and node.demand_sku is not None:
                t_list = set(range(self.arg.T)) & set(node.demand_sku.index)
                # for t in node.demand_sku.index:
                for t in t_list:
                    for k in node.demand_sku[t]:
                        slack = self.vars["sku_demand_slack"][(t, node, k)].x
                        demand = node.demand[t, k]
                        if preserve_zeros or slack != 0 or demand != 0:
                            node_sku_t_demand_slack.iloc[demand_slack_index] = {
                                "node": node.idx,
                                "type": node.type,
                                "sku": k.idx,
                                "t": t,
                                "demand": demand,
                                "slack": slack,
                                "fulfill": 1 - slack / demand,
                            }
                            demand_slack_index += 1

        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            for t in range(self.T):
                edge_sku_list = edge.get_edge_sku_list(t, self.full_sku_list)
                for k in edge_sku_list:
                    if preserve_zeros or self.vars["sku_flow"][(t, edge, k)].x != 0:
                        edge_sku_t_flow.iloc[edge_index] = {
                            "id": edge.idx,
                            "start": edge.start.idx,
                            "end": edge.end.idx,
                            "sku": k.idx,
                            "t": t,
                            "qty": self.vars["sku_flow"][(t, edge, k)].x,
                        }
                        edge_index += 1

        """
        kpi:
        1. demand fulfillment rate (for each sku and total)
        2. avg inventory for each warehouse along time 

        """
        plant_sku_t_production.dropna(inplace=True)
        warehouse_sku_t_storage.dropna(inplace=True)
        node_sku_t_demand_slack.dropna(inplace=True)
        edge_sku_t_flow.dropna(inplace=True)

        if (
            len(
                node_sku_t_demand_slack[
                    node_sku_t_demand_slack["type"] == const.CUSTOMER
                ]
            )
            != 0
        ):
            customer_fullfill_sku_rate = (
                node_sku_t_demand_slack[
                    node_sku_t_demand_slack["type"] == const.CUSTOMER
                ]
                .groupby("sku")
                .sum()[["demand", "slack"]]
            )
            customer_fullfill_sku_rate[
                "fulfill_rate"
            ] = customer_fullfill_sku_rate.apply(
                lambda x: 1 - x["slack"] / x["demand"], axis=1
            )
            customer_fullfill_total_rate = (
                1
                - customer_fullfill_sku_rate["slack"].sum()
                / customer_fullfill_sku_rate["demand"].sum()
            )
        else:
            customer_fullfill_sku_rate = node_sku_t_demand_slack[
                node_sku_t_demand_slack["type"] == const.CUSTOMER
            ][["demand", "slack"]]
            customer_fullfill_total_rate = 1

        if (
            len(
                node_sku_t_demand_slack[
                    node_sku_t_demand_slack["type"] == const.WAREHOUSE
                ]
            )
            != 0
        ):
            warehouse_fullfill_sku_rate = (
                node_sku_t_demand_slack[
                    node_sku_t_demand_slack["type"] == const.WAREHOUSE
                ]
                .groupby("sku")
                .sum()[["demand", "slack"]]
            )
            warehouse_fullfill_sku_rate[
                "fulfill_rate"
            ] = warehouse_fullfill_sku_rate.apply(
                lambda x: 1 - x["slack"] / x["demand"], axis=1
            )
            warehouse_fullfill_total_rate = (
                1
                - warehouse_fullfill_sku_rate["slack"].sum()
                / warehouse_fullfill_sku_rate["demand"].sum()
            )
        else:
            warehouse_fullfill_sku_rate = node_sku_t_demand_slack[
                node_sku_t_demand_slack["type"] == const.WAREHOUSE
            ][["demand", "slack"]]
            warehouse_fullfill_total_rate = 1

        if len(node_sku_t_demand_slack) != 0:
            total_fullfill_sku_rate = node_sku_t_demand_slack.groupby("sku").sum()[
                ["demand", "slack"]
            ]
            total_fullfill_sku_rate["fulfill_rate"] = total_fullfill_sku_rate.apply(
                lambda x: 1 - x["slack"] / x["demand"], axis=1
            )
            total_fullfill_rate = (
                1
                - total_fullfill_sku_rate["slack"].sum()
                / total_fullfill_sku_rate["demand"].sum()
            )
        else:
            total_fullfill_sku_rate = node_sku_t_demand_slack[["demand", "slack"]]
            total_fullfill_rate = 1
        
        try:
            warehouse_avg_inventory_t = (
                warehouse_sku_t_storage.groupby("node").sum()["qty"] / self.T
            )
            warehouse_total_avg_inventory = warehouse_avg_inventory_t.sum() / len(
                warehouse_avg_inventory_t
            )
        except Exception as e:
            logger.warn(f"table warehouse_total_avg_inventory failed")
            # logger.exception(e)
            warehouse_total_avg_inventory = 0
            warehouse_avg_inventory_t = None

        overall_kpi = {
            "customer_fullfill_rate": customer_fullfill_total_rate,
            "warehouse_fullfill_rate": warehouse_fullfill_total_rate,
            "overall_fullfill_rate": total_fullfill_rate,
            "warehouse_overall_avg_inventory": warehouse_total_avg_inventory,
        }
        print(overall_kpi)
        overall_kpi = pd.DataFrame(overall_kpi, index=[0])
        with pd.ExcelWriter(os.path.join(data_dir, "kpi.xlsx")) as writer:
            customer_fullfill_sku_rate.to_excel(
                writer, sheet_name="customer_fullfill_sku_rate"
            )
            warehouse_fullfill_sku_rate.to_excel(
                writer, sheet_name="warehouse_fullfill_sku_rate"
            )
            total_fullfill_sku_rate.to_excel(
                writer, sheet_name="node_fullfill_sku_rate"
            )
            if warehouse_avg_inventory_t: warehouse_avg_inventory_t.to_excel(
                writer, sheet_name="warehouse_avg_inventory"
            )
            overall_kpi.to_excel(writer, sheet_name="overall_kpi")

        plant_sku_t_production.to_csv(
            os.path.join(data_dir, "plant_sku_t_production.csv"), index=False
        )
        warehouse_sku_t_storage.to_csv(
            os.path.join(data_dir, "warehouse_sku_t_storage.csv"), index=False
        )
        node_sku_t_demand_slack.to_csv(
            os.path.join(data_dir, "node_sku_t_demand_slack.csv"), index=False
        )
        if self.open_relationship:
            node_t_open.to_csv(os.path.join(data_dir, "node_t_open.csv"), index=False)
        edge_sku_t_flow.to_csv(
            os.path.join(data_dir, "edge_sku_t_flow.csv"), index=False
        )
        logger.info("saving finished")


if __name__ == "__main__":
    from param import Param
    from network import constuct_network
    import pandas as pd

    param = Param()
    arg = param.arg
    arg.T = 1
    arg.backorder = False

    datapath = "data/data_0401_V3.xlsx"

    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=datapath, one_period=True
    )
    # best solution: 1206630185
    node_list = plant_list + warehouse_list + customer_list

    network = constuct_network(node_list, edge_list, sku_list)
    model = DNP(arg, network)
    model.modeling()
    model.solve()

    solpath = "/Users/liu/Desktop/MyRepositories/facility-loc-inventory/output"
    model.get_solution(solpath)
