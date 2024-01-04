import argparse
import asyncio
import logging
import os
import time
from typing import List

import coptpy as cp
import networkx as nx
import numpy as np
import pandas as pd
import ray
from coptpy import COPT

import const
from entity import SKU
from config.read_data import read_data
from utils import get_in_edges, get_out_edges, logger

ATTR_IN_RMP = ["sku_flow_sum", "sku_production_sum", "sku_inventory_sum"]
# macro for debugging
CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 1))
CG_SUBP_LOGGING = int(os.environ.get("CG_SUBP_LOGGING", 0))
CG_SUBP_THREADS = int(os.environ.get("CG_SUBP_THREADS", 2))
CG_SUBP_GAP = float(os.environ.get("CG_SUBP_GAP", 0.05))
CG_SUBP_TIMELIMIT = float(os.environ.get("CG_SUBP_TIMELIMIT", 100))


@ray.remote
class DNP_worker:
    """
    Worker class for DNP, each worker is responsible for num_cus customer
    """

    def __init__(
        self, cus_list, arg, bool_covering, bool_capacity, bool_edge_lb, bool_node_lb
    ):
        self.arg = arg
        self.cus_list = cus_list
        self.DNP_dict = {}
        self.bool_covering = bool_covering
        self.bool_capacity = bool_capacity
        self.bool_edge_lb = bool_edge_lb
        self.bool_node_lb = bool_node_lb

    def construct_DNPs(self, subgraph_dict):
        for customer in self.cus_list:
            subgraph = subgraph_dict[customer]
            full_sku_list = subgraph.graph["sku_list"]
            env_name = customer.idx + "_oracle_env"
            model_name = customer.idx + "_oracle"
            oracle = dnp_model.DNP(
                self.arg,
                subgraph,
                full_sku_list,
                env_name,
                model_name,
                bool_capacity=True,
                bool_fixed_cost=False,
                bool_covering=self.bool_covering,
                bool_edge_lb=self.bool_edge_lb,
                bool_node_lb=False,
                logging=CG_SUBP_LOGGING,
                threads=CG_SUBP_THREADS,
                gap=CG_SUBP_GAP,
                limit=CG_SUBP_TIMELIMIT,
                cus_list=[customer],
            )
            oracle.modeling()
            self.DNP_dict[customer] = oracle

    # for primal sweeping
    def del_constr_capacity(self, customer):
        self.DNP_dict[customer].del_constr_capacity()

    def update_constr_capacity(self, customer, ec, pc, wc):
        self.DNP_dict[customer].update_constr_capacity(ec, pc, wc)

    def add_constr_holding_capacity(self, customer, t):
        self.DNP_dict[customer].add_constr_holding_capacity(t)

    def add_constr_production_capacity(self, customer, t):
        self.DNP_dict[customer].add_constr_production_capacity(t)

    def add_constr_transportation_capacity(self, customer, t):
        self.DNP_dict[customer].add_constr_transportation_capacity(t)

    def query_columns(self, customer):
        return self.DNP_dict[customer].query_columns()

    def query_all_columns(self):
        columns = []
        for customer in self.cus_list:
            columns.append(self.DNP_dict[customer].query_columns())
        return columns

    # for subproblem
    def model_reset(self, customer):
        self.DNP_dict[customer].model_reset()

    def model_reset_all(self):
        for customer in self.cus_list:
            self.DNP_dict[customer].model_reset()

    def update_objective(self, customer, dual_vars, dual_index):
        self.DNP_dict[customer].update_objective(customer, dual_vars, dual_index)

    def update_objective_all(self, dual_vars, dual_index):
        for customer in self.cus_list:
            self.DNP_dict[customer].update_objective(customer, dual_vars, dual_index)

    def solve(self, customer):
        self.DNP_dict[customer].solve()

    def solve_all(self):
        for customer in self.cus_list:
            self.DNP_dict[customer].solve()

    def get_model_objval(self, customer):
        return self.DNP_dict[customer].get_model_objval()

    def get_model_status(self, customer):
        return self.DNP_dict[customer].get_model_status()

    def get_all_model_objval(self):
        objval = []
        for customer in self.cus_list:
            objval.append(self.DNP_dict[customer].get_model_objval())
        return objval

    def get_all_model_status(self):
        status = []
        for customer in self.cus_list:
            status.append(self.DNP_dict[customer].get_model_status())
        return status


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
        used_edge_capacity: dict = None,
        used_warehouse_capacity: dict = None,
        used_plant_capacity: dict = None,
        logging: int = 0,
        gap: float = 1e-4,
        threads: int = None,
        limit: int = 3600,
        cus_num: int = 1,
        env=None,
        cus_list=None,
    ) -> None:
        self.obj_types = {}
        self.cus_list = cus_list
        self.arg = arg
        self.T = self.arg.T
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
        self.model.setParam(COPT.Param.Logging, logging)
        self.model.setParam(COPT.Param.RelGap, gap)
        self.model.setParam(COPT.Param.TimeLimit, limit)
        if threads is not None:
            self.model.setParam(COPT.Param.Threads, threads)

        self.variables = dict()  # variables
        self.constrs = dict()  # constraints
        self.obj = dict()  # objective
        self.bool_fixed_cost = self.arg.fixed_cost
        self.bool_covering = self.arg.covering
        self.bool_capacity = self.arg.capacity
        self.add_in_upper = self.arg.add_in_upper
        self.add_distance = self.arg.add_distance
        self.add_cardinality = self.arg.add_cardinality
        self.bool_edge_lb = False
        self.bool_node_lb = False
        self.used_edge_capacity = {
            t: used_edge_capacity if used_edge_capacity is not None else {}
            for t in range(self.T)
        }
        self.used_warehouse_capacity = {
            t: used_warehouse_capacity if used_warehouse_capacity is not None else {}
            for t in range(self.T)
        }
        self.used_plant_capacity = {
            t: used_plant_capacity if used_plant_capacity is not None else {}
            for t in range(self.T)
        }
        # whether to add edge lower bound constraints
        if self.bool_covering:
            self.bool_edge_lb = self.arg.edge_lb
        else:
            if self.arg.edge_lb:
                logger.warning(
                    "bool_edge_lb is set to False because bool_covering is False."
                )
        # whether to add node lower bound constraints
        if self.bool_covering:
            self.bool_node_lb = self.arg.node_lb
        else:
            if self.arg.node_lb:
                logger.warning(
                    "bool_node_lb is set to False because bool_covering is False."
                )
        self.original_obj = 0.0
        self.total_cus_num = cus_num
        self.var_idx = None
        self.dual_index_for_RMP = {
            "node_capacity":dict(), "transportation_variable_lb":dict(), "transportation_capacity":dict(),
            "production_variable_lb":dict(), "holding_variable_lb":dict(), "in_upper":dict(),
        }
        self.index_for_dual_var = 0  # void bugs of index out of range
        # for remote
        self.columns_helpers = None

    # for ray.remote
    def update_constr_capacity(self, ec, pc, wc):
        self.used_edge_capacity = ec
        self.used_plant_capacity = pc
        self.used_warehouse_capacity = wc

    def writeLP(self, idx):
        self.model.write(f"oracle_lp/{idx}.lp")

    def get_model_status(self):
        return self.model.status

    def get_T(self):
        return self.T

    def get_variables(self):
        return self.variables

    def init_col_helpers(self):
        col_helper = {
            attr: {
                # t: {} for t in range(ray.get(self.oracles[customer].getT.remote()))
                t: {}
                for t in range(self.arg.T)
            }
            for attr in ATTR_IN_RMP
        }

        # saving column LinExpr
        for t in range(self.arg.T):
            for e in self.network.edges:
                edge = self.network.edges[e]["object"]
                # if edge.capacity == np.inf:
                #     continue
                col_helper["sku_flow_sum"][t][edge] = self.variables["sku_flow"].sum(
                    t, edge, "*"
                )

            for node in self.network.nodes:
                if node.type == const.PLANT:
                    # if node.production_capacity == np.inf:
                    #     continue
                    col_helper["sku_production_sum"][t][node] = self.variables[
                        "sku_production"
                    ].sum(t, node, "*")
                elif node.type == const.WAREHOUSE:
                    # if node.inventory_capacity == np.inf:
                    #     continue
                    col_helper["sku_inventory_sum"][t][node] = self.variables[
                        "sku_inventory"
                    ].sum(t, node, "*")

        col_helper["beta"] = (
            self.original_obj
            if type(self.original_obj) == float
            # else self.original_obj.getExpr()
            else self.solver.getExpr(self.original_obj)
        )

        self.columns_helpers = col_helper
        self.columns = []
        return

    def eval_helper(self):
        _vals = {
            attr: {
                t: {
                    k: v.getValue() if type(v) is not float else 0
                    for k, v in self.columns_helpers[attr][t].items()
                }
                for t in range(self.arg.T)
            }
            for attr in ATTR_IN_RMP
        }
        _vals["beta"] = (
            self.columns_helpers["beta"]
            if type(self.columns_helpers["beta"]) == float
            else self.columns_helpers["beta"].getValue()
        )

        return _vals

    def query_columns(self):
        new_col = self.eval_helper()
        return new_col

    def model_reset(self):
        self.model.reset()

    def get_model_objval(self):
        return self.model.objval

    #####################

    def modeling(self):
        """
        build DNP model
        """
        self.add_vars()

        self.add_constraints()

        self.set_objective()

        # for remote
        # self.init_col_helpers()

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
                "lb": 0,
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
                "index": "(t, customer, k)",
            },
        }

        if self.bool_covering:
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
                if self.bool_covering:
                    idx["select_edge"].append((t, edge))

                sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    # sku k select edge (i,j) at t
                    if self.bool_covering:
                        idx["sku_select_edge"].append((t, edge, k))
                    # flow of sku k on edge (i,j) at t
                    idx["sku_flow"].append((t, edge, k))

            # nodes
            for node in self.network.nodes:
                # open node i at t
                if self.bool_covering:
                    idx["open"].append((t, node))

                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    if node.type == const.PLANT:
                        # sku k produced on node i at t
                        if self.bool_covering:
                            idx["sku_open"].append((t, node, k))
                        # amount of sku k produced on node i at t
                        idx["sku_production"].append((t, node, k))
                    elif node.type == const.WAREHOUSE:
                        # amount of sku k stored on node i at t
                        idx["sku_inventory"].append((t, node, k))
                    elif node.type == const.CUSTOMER:
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            idx["sku_demand_slack"].append((t, node, k))
                            if self.arg.backorder:
                                self.var_types["sku_demand_slack"]["ub"].append(
                                   1e4
                                )
                            else:
                                self.var_types["sku_demand_slack"]["ub"].append(
                                    node.demand[t, k]
                                )
        self.var_idx = {}
        for var in idx.keys():
            self.var_idx[var] = {key: 0 for key in idx[var]}
        for vt, param in self.var_types.items():
            self.variables[vt] = self.model.addVars(
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
                "sku_flow_select": {"index": "(t, edge, k)"},
            },
            "transportation_capacity": {"index": "(t, edge)"},
            "production_capacity": {"index": "(t, node)"},
            "holding_capacity": {"index": "(t, node)"},
        }

        if self.add_cardinality:
            self.constr_types["cardinality"] = {"index": "(t, node)"}
        if self.add_distance:
            self.constr_types["distance"] = {"index": "(t, node)"}
        if self.bool_edge_lb:
            self.constr_types["transportation_variable_lb"] = {"index": "(t, edge)"}
        if self.bool_node_lb:
            self.constr_types["production_variable_lb"] = {"index": "(t, node)"}
            self.constr_types["holding_variable_lb"] = {"index": "(t, node)"}
        if self.add_in_upper:
            self.constr_types["in_upper"] = {"index": "(t, node)"}

        for constr in self.constr_types.keys():
            self.constrs[constr] = dict()

        for constr in self.constr_types["open_relationship"].keys():
            self.constrs["open_relationship"][constr] = dict()

        for t in range(self.T):
            # initial status and flow conservation
            self.add_constr_flow_conservation(t)
            if self.bool_covering:
                self.add_constr_open_relationship(t)
                if self.add_cardinality:
                    self.add_constr_cardinality(t)
            if self.bool_capacity:
                self.add_constr_transportation_capacity(t)
                self.add_constr_production_capacity(t)
                self.add_constr_holding_capacity(t)
            if self.bool_edge_lb:
                self.add_constr_transportation_lb(t)
            if self.bool_node_lb:
                self.add_constr_node_lb(t)
            if self.add_in_upper:
                self.add_constr_flow_in_upper(t)
            if self.add_distance:
                self.add_constr_distance(t)

    def del_constr_capacity(self):
        to_remove = [
            "transportation_capacity",
            "production_capacity",
            "holding_capacity",
        ]
        for k in to_remove:
            for cc in self.constrs[k].values():
                self.model.remove(cc)
    def add_constr_flow_conservation(self, t: int):
        for node in self.network.nodes:
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            for k in sku_list:
                in_edges = get_in_edges(self.network, node)
                out_edges = get_out_edges(self.network, node)
                constr_name = f"flow_conservation_{t}_{node.idx}_{k.idx}"

                if node.type == const.PLANT:
                    constr = self.model.addConstr(
                        self.variables["sku_production"][t, node, k]
                        - self.variables["sku_flow"].sum(t, out_edges, k)
                        == 0,
                        name=constr_name,
                    )
                elif node.type == const.WAREHOUSE:
                    last_period_inventory = 0.0

                    if t == 0:
                        if node.initial_inventory is not None:
                            # self.model.addConstr(
                            #     self.variables["open"][self.T - 1, node] == 1
                            # )
                            last_period_inventory = (
                                node.initial_inventory[k]
                                if k in node.initial_inventory
                                else 0.0
                            )
                        else:
                            last_period_inventory = 0.0
                    else:
                        last_period_inventory = self.variables["sku_inventory"][
                            t - 1, node, k
                        ]

                    constr = self.model.addConstr(
                        self.variables["sku_flow"].sum(t, in_edges, k)
                        + last_period_inventory
                        - self.variables["sku_flow"].sum(t, out_edges, k)
                        == self.variables["sku_inventory"][t, node, k],
                        name=constr_name,
                    )

                elif node.type == const.CUSTOMER:
                    demand = node.demand.loc[(t, k)]
                    if self.arg.backorder:
                        constr = self.model.addConstr(
                            self.variables["sku_flow"].sum(t, in_edges, k)
                            + self.variables["sku_demand_slack"][t, node, k]
                            == self.variables["sku_demand_slack"].get(
                                (t - 1, node, k), 0
                            ) + demand,
                            name=constr_name,
                        )
                    else:
                        constr = self.model.addConstr(
                            self.variables["sku_flow"].sum(t, in_edges, k)
                            == demand - self.variables["sku_demand_slack"][t, node, k],
                            name=constr_name,
                        )

                self.constrs["flow_conservation"][(t, node, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]

            sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

            constr = self.model.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.start]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.start)
            ] = constr

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.end)
            ] = constr = self.model.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.end]
            )

            for k in sku_list:
                constr = self.model.addConstr(
                    self.variables["sku_select_edge"][t, edge, k]
                    <= self.variables["select_edge"][t, edge]
                )
                self.constrs["open_relationship"]["sku_select_edge"][
                    (t, edge, k)
                ] = constr

                constr = self.model.addConstr(
                    self.variables["sku_flow"][t, edge, k]
                    <= 1e6 * self.variables["sku_select_edge"][t, edge, k]
                )
                self.constrs["open_relationship"]["sku_flow_select"][
                    (t, edge, k)
                ] = constr

        for node in self.network.nodes:
            sku_list = node.get_node_sku_list(t, self.full_sku_list)
            # TODO: 后面可以改成有库存的话应该开
            # if (
            #     node.type == const.WAREHOUSE
            #     and node.has_demand(t)
            #     and len(node.demand_sku[t]) > 0
            # ):
            #     constr = self.model.addConstr(self.variables["open"][t, node] == 1)
            #     self.constrs["open_relationship"]["open"][(t, node)] = constr
            if node.type == const.CUSTOMER:
                constr = self.model.addConstr(self.variables["open"][t, node] == 1)
                self.constrs["open_relationship"]["open"][(t, node)] = constr
            # TODO: Check 是否有必要
            for k in sku_list:
                if node.type == const.PLANT:
                    constr = self.model.addConstr(
                        self.variables["sku_open"][t, node, k]
                        <= self.variables["open"][t, node]
                    )
                self.constrs["open_relationship"]["sku_open"][(t, node, k)] = constr

        return

    def add_constr_transportation_capacity(self, t: int, verbose=False):
        # for debug
        if verbose:
            print(self.cus_list[0].idx, " ec before:", t, " at ", time.time())
            for k, v in self.used_edge_capacity[t].items():
                print(k, ":", v)
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")
            if edge.capacity < np.inf:
                left_capacity = edge.capacity - self.used_edge_capacity.get(t).get(
                    edge, 0
                )
                bound = (
                    self.variables["select_edge"][t, edge]
                    if self.bool_covering
                    else 1.0
                )

                self.constrs["transportation_capacity"][
                    (t, edge)
                ] = self.model.addConstr(
                    flow_sum <= left_capacity * bound,
                    name=f"edge_capacity{t, edge}",
                )
                self.dual_index_for_RMP["transportation_capacity"][
                    edge
                ] = self.index_for_dual_var
                self.index_for_dual_var += 1
        return

    def add_constr_transportation_lb(self, t: int, verbose=False):
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")

            # variable lower bound
            if edge.variable_lb < np.inf:
                self.constrs["transportation_variable_lb"][
                    (t, edge)
                ] = self.model.addConstr(
                    flow_sum
                    >= edge.variable_lb * self.variables["select_edge"][t, edge]
                )
                self.dual_index_for_RMP["transportation_variable_lb"][
                    edge
                ] = self.index_for_dual_var
                self.index_for_dual_var += 1
        return

    def add_constr_node_lb(self, t: int):
        for node in self.network.nodes:
            if node.type == const.PLANT:
                node_sum = self.variables["sku_production"].sum(t, node, "*")
                if node.production_lb < np.inf:
                    self.constrs["production_variable_lb"][
                        (t, node)
                    ] = self.model.addConstr(
                        node_sum >= node.production_lb * self.variables["open"][t, node],
                        name=f"node_plant_lb{t, node}",
                    )
                self.dual_index_for_RMP["production_variable_lb"][node] = self.index_for_dual_var
                self.index_for_dual_var += 1
            if node.type == const.WAREHOUSE:
                if node.inventory_lb < np.inf:
                    node_sum = self.variables["sku_inventory"].sum(t, node, "*")
                    self.constrs["holding_variable_lb"][
                        (t, node)
                    ] = self.model.addConstr(
                        node_sum
                        >= node.inventory_lb * self.variables["open"][(t, node)],
                        name=f"node_warehouse_lb{t, node}",
                    )
                    self.dual_index_for_RMP["holding_variable_lb"][node] = self.index_for_dual_var
                    self.index_for_dual_var += 1
        return

    def add_constr_production_capacity(self, t: int):
        for node in self.network.nodes:
            if node.type != const.PLANT:
                continue

            node_sum = self.variables["sku_production"].sum(t, node, "*")
            # capacity constraint
            if node.production_capacity < np.inf:
                left_capacity = node.production_capacity - self.used_plant_capacity.get(
                    node, 0
                )
                bound = self.variables["open"][t, node] if self.bool_covering else 1.0

                self.constrs["production_capacity"][(t, node)] = self.model.addConstr(
                    node_sum <= bound * left_capacity,
                    name=f"node_capacity{t, node}",
                )
                self.dual_index_for_RMP["node_capacity"][node] = self.index_for_dual_var
                self.index_for_dual_var += 1

        return

    def add_constr_holding_capacity(self, t: int):
        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                node_sum = self.variables["sku_inventory"].sum(t, node, "*")
                # capacity constraint
                if node.inventory_capacity < np.inf:
                    left_capacity = (
                        node.inventory_capacity
                        - self.used_warehouse_capacity.get(t).get(node, 0)
                    )
                    bound = (
                        self.variables["open"][(t, node)] if self.bool_covering else 1.0
                    )

                    constr = self.model.addConstr(
                        self.variables["sku_inventory"].sum(t, node, "*")
                        <= left_capacity * bound
                    )
                    self.constrs["holding_capacity"][(t, node)] = constr

                    self.dual_index_for_RMP["node_capacity"][
                        node
                    ] = self.index_for_dual_var
                    self.index_for_dual_var += 1

        return

    def add_constr_flow_in_upper(self, t: int):
        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                in_edges = get_in_edges(self.network, node)
                inbound_sum = self.variables["sku_flow"].sum(t, in_edges, "*")
                self.constrs["in_upper"][(t, node)] = self.model.addConstr(
                    inbound_sum <= node.inventory_capacity * self.arg.in_upper_ratio
                )
                self.dual_index_for_RMP["in_upper"][node] = self.index_for_dual_var
                self.index_for_dual_var += 1
        return

    def add_constr_cardinality(self, t: int):
        for node in self.network.nodes:
            if node.type == const.CUSTOMER:
                in_edges = get_in_edges(self.network, node)
                used_edge = self.variables["select_edge"].sum(t, in_edges)
                constr = self.model.addConstr(used_edge <= self.arg.cardinality_limit)
                self.constrs["cardinality"][(t, node)] = constr
                self.index_for_dual_var += 1
        return

    def add_constr_distance(self, t: int):
        for node in self.network.nodes:
            if node.type == const.CUSTOMER:
                used_distance = 0
                for e in self.network.edges:
                    edge = self.network.edges[e]["object"]
                    if edge.end == node:
                        used_distance += (
                                self.variables["select_edge"][t, edge] * edge.distance
                        )
                constr = self.model.addConstr(used_distance <= self.arg.distance_limit)
                self.constrs["distance"][(t, node)] = constr
                self.index_for_dual_var += 1
        return

    def get_original_objective(self):
        obj = 0.0
        for t in range(self.T):
            obj = obj + self.cal_sku_producing_cost(t)
            obj = obj + self.cal_sku_holding_cost(t)
            obj = obj + self.cal_sku_transportation_cost(t)
            obj = obj + self.cal_sku_unfulfilled_demand_cost(t)
        if self.bool_fixed_cost:
            obj = obj + self.cal_fixed_node_cost()
            # obj = obj + self.cal_fixed_edge_cost()
        return obj

    def extra_objective(self, customer, dualvar=None, dual_index=None):
        obj = 0.0
        if dualvar is None:
            return obj
        for t, edge in tuple(dual_index["transportation_capacity"].keys()):
            obj -= dualvar[
                dual_index["transportation_capacity"][(t, edge)]
            ] * self.variables["sku_flow"].sum(t, edge, "*")

        for t, node in tuple(dual_index["node_capacity"].keys()):
            if node.type == const.PLANT:
                obj -= dualvar[dual_index["node_capacity"][(t, node)]] * self.variables[
                    "sku_production"
                ].sum(t, node, "*")
            elif node.type == const.WAREHOUSE:
                obj -= dualvar[dual_index["node_capacity"][(t, node)]] * self.variables[
                    "sku_inventory"
                ].sum(t, node, "*")
            else:
                continue
        obj -= dualvar[dual_index["weights_sum"][customer]]
        return obj

    def update_objective(self, customer, dualvar, dual_index):
        """
        Use dual variables to calculate the reduced cost
        """

        obj = self.original_obj + self.extra_objective(customer, dualvar, dual_index)

        self.model.setObjective(obj, sense=COPT.MINIMIZE)

    def set_objective(self):

        self.obj_types = {
            "producing_cost": {"index": "(t, plant)"},
            "holding_cost": {"index": "(t, warehouse)"},
            "sku_backorder_cost": {"index": "(t, warehouse, k)"},
            "transportation_cost": {"index": "(t, edge)"},
            "unfulfilled_demand_cost": {
                "index": "(t, customer)"
            },
            "fixed_node_cost": {"index": "(t, plant / warehouse, k)"},
            # "fixed_edge_cost": {"index": "(t, edge, k)"},
            # "end_inventory_cost": {"index": "(node, k)"},
        }

        for obj in self.obj_types.keys():
            self.obj[obj] = dict()

        self.original_obj = self.get_original_objective()
        self.model.setObjective(self.original_obj, sense=COPT.MINIMIZE)

        return

    def cal_sku_producing_cost(self, t: int):
        producing_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.PLANT:
                node_producing_cost = 0
                sku_list = node.get_node_sku_list(t, self.full_sku_list)
                for k in sku_list:
                    # TODO
                    # if (
                    #     node.production_sku_fixed_cost is not None
                    #     and self.bool_covering
                    # ):
                    #     node_sku_producing_cost = (
                    #         node_sku_producing_cost
                    #         + node.production_sku_fixed_cost[k]
                    #         * self.variables["sku_open"][t, node, k]
                    #     )\
                    if node.production_sku_unit_cost is not None and k in node.production_sku_unit_cost.index.to_list():
                    # if k in node.production_sku_unit_cost.index.to_list():
                        node_producing_cost += node.production_sku_unit_cost[k] * self.variables["sku_production"][t, node, k]
                    else:
                        node_producing_cost += self.arg.production_sku_unit_cost * self.variables["sku_production"][t, node, k]

                producing_cost = producing_cost + node_producing_cost

                self.obj["producing_cost"][(t, node)] = node_producing_cost
        return producing_cost

    def cal_sku_holding_cost(self, t: int):
        holding_cost = 0.0

        for node in self.network.nodes:
            if node.type == const.WAREHOUSE:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)
                node_holding_cost = 0.0
                for k in sku_list:

                    if node.holding_sku_unit_cost is not None:
                        holding_sku_unit_cost = node.holding_sku_unit_cost[k]
                    else:
                        holding_sku_unit_cost = self.arg.holding_sku_unit_cost

                    node_holding_cost += holding_sku_unit_cost * self.variables["sku_inventory"][t, node, k]

                holding_cost = holding_cost + node_holding_cost

                self.obj["holding_cost"][(t, node)] = node_holding_cost

        return holding_cost

    def cal_sku_transportation_cost(self, t: int):
        transportation_cost = 0.0

        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            edge_transportation_cost = 0.0

            (
                sku_list_with_fixed_transportation_cost,
                sku_list_with_unit_transportation_cost,
            ) = edge.get_edge_sku_list_with_transportation_cost(t, self.full_sku_list)

            # if self.bool_covering:
            #     for k in sku_list_with_fixed_transportation_cost:
            #         if (
            #             edge.transportation_sku_fixed_cost is not None
            #             and k in edge.transportation_sku_fixed_cost
            #         ):
            #             edge_transportation_cost = (
            #                 edge_transportation_cost
            #                 + edge.transportation_sku_fixed_cost[k]
            #                 * self.variables["sku_select_edge"][t, edge, k]
            #             )

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
                    + transportation_sku_unit_cost
                    * self.variables["sku_flow"][t, edge, k]
                )

            transportation_cost = transportation_cost + edge_transportation_cost

            self.obj["transportation_cost"][(t, edge)] = edge_transportation_cost

        return transportation_cost

    def cal_sku_unfulfilled_demand_cost(self, t: int):
        unfulfilled_demand_cost = 0.0
        for node in self.network.nodes:
            if node.type == const.CUSTOMER:
                unfulfilled_node_cost = 0.0
                if node.has_demand(t):
                    for k in node.demand_sku[t]:
                        if node.unfulfill_sku_unit_cost is not None:
                            unfulfilled_sku_unit_cost = node.unfulfill_sku_unit_cost[
                                (t, k)
                            ]
                        else:
                            # TODO:diversity
                            unfulfilled_sku_unit_cost = self.arg.unfulfill_sku_unit_cost
                        unfulfilled_node_cost += unfulfilled_sku_unit_cost * self.variables["sku_demand_slack"][(t, node, k)]
                    unfulfilled_demand_cost += unfulfilled_node_cost
                    self.obj["unfulfilled_demand_cost"][(t, node)] = unfulfilled_node_cost
        return unfulfilled_demand_cost

    def cal_fixed_node_cost(self):
        fixed_node_cost = 0.0

        if not self.bool_covering:
            return fixed_node_cost

        for node in self.network.nodes:
            if node.type == const.PLANT:
                if node.production_fixed_cost is not None:
                    this_node_fixed_cost = node.production_fixed_cost
                else:
                    this_node_fixed_cost = self.arg.plant_fixed_cost
            elif node.type == const.WAREHOUSE:
                if node.holding_fixed_cost is not None:
                    this_node_fixed_cost = node.holding_fixed_cost
                else:
                    this_node_fixed_cost = self.arg.warehouse_fixed_cost
            else:
                continue
            node_fixed_node_cost = 0.0
            for t in range(self.T):
                node_fixed_node_cost += this_node_fixed_cost * self.variables["open"][(t, node)]

            fixed_node_cost += node_fixed_node_cost
            self.obj["fixed_node_cost"][node] = node_fixed_node_cost

        return fixed_node_cost

    # TODO:REMOVE
    def cal_fixed_edge_cost(self):
        fixed_edge_cost = 0.0
        if not self.bool_covering:
            return fixed_edge_cost

        for e in self.network.edges:
            edge = self.network.edges[e]["object"]

            p = self.model.addVar(vtype=COPT.BINARY, name=f"p_{edge}")

            for t in range(self.T):
                self.model.addConstr(self.variables["select_edge"][(t, edge)] <= p)

            # edge_fixed_edge_cost = edge.transportation_fixed_cost * p
            edge.transportation_fixed_cost = 10
            edge_fixed_edge_cost = edge.transportation_fixed_cost * p
            fixed_edge_cost = fixed_edge_cost + edge_fixed_edge_cost

            self.obj["fixed_edge_cost"][edge] = edge_fixed_edge_cost

        return fixed_edge_cost

    def solve(self):
        self.model.solve()

    def write(self, name):
        self.model.write(name)


    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        # node output

        plant_sku_t_production = pd.DataFrame(
            index=range(len(self.variables["sku_production"])),
            columns=["node", "type", "sku", "t", "qty"],
        )
        warehouse_sku_t_storage = pd.DataFrame(
            index=range(len(self.variables["sku_inventory"])),
            columns=["node", "type", "sku", "t", "qty"],
        )
        node_sku_t_demand_slack = pd.DataFrame(
            index=range(len(self.variables["sku_demand_slack"])),
            columns=["node", "type", "sku", "t", "demand", "slack", "fulfill"],
        )

        if self.bool_covering:
            node_t_open = pd.DataFrame(
                index=range(len(self.variables["open"])),
                columns=["node", "type", "t", "open"],
            )

        # edge output
        edge_sku_t_flow = pd.DataFrame(
            index=range(len(self.variables["sku_flow"])),
            columns=[
                "id",
                "start",
                "end",
                "sku",
                "t",
                "y",
                "qty",
                "vlb",
                "cap",
                "obj_start",
                "obj_end",
                "obj_edge",
            ],
        )

        node_open_index = 0
        plant_index = 0
        warehouse_index = 0
        demand_slack_index = 0
        edge_index = 0

        for node in self.network.nodes:
            for t in range(self.T):
                if self.bool_covering:
                    node_t_open.iloc[node_open_index] = {
                        "node": node.idx,
                        "type": node.type,
                        "t": t,
                        "open": self.variables["open"][(t, node)].x if node.type != const.CUSTOMER else 1,
                    }
                    node_open_index += 1

                if node.type == const.PLANT:
                    if node.producible_sku is not None:
                        for k in node.producible_sku:
                            if (
                                preserve_zeros
                                or self.variables["sku_production"][(t, node, k)].x != 0
                            ):
                                plant_sku_t_production.iloc[plant_index] = {
                                    "node": node.idx,
                                    "type": node.type,
                                    "sku": k.idx,
                                    "t": t,
                                    "qty": self.variables["sku_production"][
                                        (t, node, k)
                                    ].x,
                                }
                                plant_index += 1

                if node.type == const.WAREHOUSE:
                    sku_list = node.get_node_sku_list(t, self.full_sku_list)
                    for k in sku_list:
                        if (
                            preserve_zeros
                            or self.variables["sku_inventory"][(t, node, k)].x != 0
                        ):
                            warehouse_sku_t_storage.iloc[warehouse_index] = {
                                "node": node.idx,
                                "type": node.type,
                                "sku": k.idx,
                                "t": t,
                                "qty": self.variables["sku_inventory"][(t, node, k)].x,
                            }
                            warehouse_index += 1

            if node.type != const.PLANT and node.demand_sku is not None:
                t_list = set(range(self.arg.T)) & set(node.demand_sku.index)
                # for t in node.demand_sku.index:
                for t in t_list:
                    for k in node.demand_sku[t]:
                        slack = self.variables["sku_demand_slack"][(t, node, k)].x
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
                    if (
                        preserve_zeros
                        or self.variables["sku_flow"][(t, edge, k)].x != 0
                    ):
                        edge_sku_t_flow.iloc[edge_index] = {
                            "id": edge.idx,
                            "start": edge.start.idx,
                            "end": edge.end.idx,
                            "sku": k.idx,
                            "t": t,
                            "qty": self.variables["sku_flow"][(t, edge, k)].x,
                            "y": self.variables["select_edge"][(t, edge)].x,
                            "vlb": edge.variable_lb,
                            "cap": edge.capacity,
                            "obj_start": edge.start,
                            "obj_end": edge.end,
                            "obj_edge": edge,
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
        edge_sku_t_flow = edge_sku_t_flow.assign(
            bool_lb_vio=lambda df: df["qty"] < df["vlb"]
        )

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
                warehouse_sku_t_storage.groupby("node").sum(numeric_only=True)["qty"]
                / self.T
            )
            warehouse_total_avg_inventory = warehouse_avg_inventory_t.sum() / len(
                warehouse_avg_inventory_t
            )
        except Exception as e:
            logger.warning(f"table warehouse_total_avg_inventory failed")
            # logger.exception(e)
            warehouse_total_avg_inventory = 0
            warehouse_avg_inventory_t = None

        overall_kpi = {
            "customer_fullfill_rate": customer_fullfill_total_rate,
            "warehouse_fullfill_rate": warehouse_fullfill_total_rate,
            "overall_fullfill_rate": total_fullfill_rate,
            "warehouse_overall_avg_inventory": warehouse_total_avg_inventory,
        }
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
            if warehouse_avg_inventory_t is not None:
                warehouse_avg_inventory_t.to_excel(
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
        if self.bool_covering:
            node_t_open.to_csv(os.path.join(data_dir, "node_t_open.csv"), index=False)
        edge_sku_t_flow.to_csv(
            os.path.join(data_dir, "edge_sku_t_flow.csv"), index=False
        )
        logger.info("saving finished")
        return edge_sku_t_flow


if __name__ == "__main__":
    import datetime

    import pandas as pd

    import utils as utils
    from config.network import construct_network
    from config.param import Param

    starttime = datetime.datetime.now()
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
    cap = pd.read_csv("./data/random_capacity_updated.csv").set_index("id")
    for e in edge_list:
        e.capacity = cap["qty"].get(e.idx, np.inf)
        e.variable_lb = cap["lb"].get(e.idx, np.inf)

    lb_df = pd.read_csv("./data/node_lb_V3.csv").set_index("id")
    for n in node_list:
        if n.type == const.PLANT:
            n.production_lb = lb_df["lb"].get(n.idx, np.inf)
        if n.type == const.WAREHOUSE:
            n.warehouse_lb = lb_df["lb"].get(n.idx, np.inf)
    network = construct_network(node_list, edge_list, sku_list)
    model = DNP(arg, network, bool_covering=True, logging=1)
    model.modeling()
    model.solve()

    solpath = utils.CONF.DEFAULT_SOL_PATH
    model.get_solution(data_dir=solpath)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
