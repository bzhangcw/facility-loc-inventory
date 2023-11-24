import argparse
import asyncio
import logging
import os
import time
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import ray

import const
import update_constr
from entity import SKU
from read_data import read_data
from update_constr import *
from utils import get_in_edges, get_out_edges, logger

import coptpy as cp
from coptpy import COPT

from solver_wrapper import GurobiWrapper, CoptWrapper
from solver_wrapper.CoptConstant import CoptConstant
from solver_wrapper.GurobiConstant import GurobiConstant

ATTR_IN_RMPSLIM = ["sku_flow"]
# macro for debugging
CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 1))
CG_SUBP_LOGGING = int(os.environ.get("CG_SUBP_LOGGING", 0))
CG_SUBP_THREADS = int(os.environ.get("CG_SUBP_THREADS", 2))
CG_SUBP_GAP = float(os.environ.get("CG_SUBP_GAP", 0.05))
CG_SUBP_TIMELIMIT = float(os.environ.get("CG_SUBP_TIMELIMIT", 100))


@ray.remote
class PricingWorker:
    """
    Worker class for pricing problem, each worker is responsible for num_cus customer
    """

    def __init__(
        self,
        cus_list,
        arg,
        bool_covering,
        bool_edge_lb,
        bool_node_lb,
        solver="COPT",
    ):
        self.arg = arg
        self.cus_list = cus_list
        self.DNP_dict = {}
        self.bool_covering = bool_covering
        self.bool_edge_lb = bool_edge_lb
        self.bool_node_lb = bool_node_lb
        self.solver = solver

        if solver == "COPT":
            self.solver_constant = CoptConstant
        elif solver == "GUROBI":
            self.solver_constant = GurobiConstant
        else:
            raise ValueError("solver must be either COPT or GUROBI")

    def construct_Pricings(self, subgraph_dict):
        for customer in self.cus_list:
            subgraph = subgraph_dict[customer]
            model_name = customer.idx + "_oracle"
            oracle = Pricing(
                self.arg,
                subgraph,
                model_name,
                bool_covering=self.bool_covering,
                bool_edge_lb=self.bool_edge_lb,
                bool_node_lb=self.bool_node_lb,
                customer=customer,
                solver=self.solver,
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
            if customer in self.skipped:
                continue
            self.DNP_dict[customer].model_reset()

    def update_objective(self, customer, dual_vars, dual_index):
        self.DNP_dict[customer].update_objective(customer, dual_vars, dual_index)

    def update_objective_all(self, dual_packs):
        for customer in self.cus_list:
            if customer in self.skipped:
                continue
            self.DNP_dict[customer].update_objective(customer, dual_packs)

    def solve(self, customer):
        self.DNP_dict[customer].solve()

    def solve_all(self):
        for customer in self.cus_list:
            if customer in self.skipped:
                continue
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

    def set_scope(self, skipped):
        self.skipped = skipped


class Pricing(object):
    """
    this is a class for dynamic network flow (DNP)
    """

    def __init__(
        self,
        arg: argparse.Namespace,
        network: nx.DiGraph,
        model_name: str = "PricingDelivery",
        bool_edge_lb: bool = True,
        bool_node_lb: bool = True,
        bool_fixed_cost: bool = True,
        bool_covering: bool = True,
        bool_dp: bool = False,
        logging: int = 0,
        gap: float = 1e-4,
        threads: int = None,
        limit: int = 3600,
        customer: Customer = None,
        solver: str = "COPT",
    ) -> None:
        self.solver_name = solver
        if solver == "COPT":
            self.solver_constant = CoptConstant
        elif solver == "GUROBI":
            self.solver_constant = GurobiConstant
        else:
            raise ValueError("solver must be either COPT or GUROBI")

        self.customer: Customer = customer
        assert isinstance(customer, Customer)

        self.arg = arg
        self.T = arg.T
        self.network = network
        self.sku_list = self.network.graph["sku_list"]

        self.bool_fixed_cost = bool_fixed_cost

        # whether to add edge lower bound constraints
        if bool_covering:
            self.bool_edge_lb = bool_edge_lb
        else:
            if bool_edge_lb:
                logger.warning(
                    "bool_edge_lb is set to False because bool_covering is False."
                )
            self.bool_edge_lb = False
        # whether to add node lower bound constraints
        if bool_covering:
            self.bool_node_lb = bool_node_lb
        else:
            if bool_node_lb:
                logger.warning(
                    "bool_node_lb is set to False because bool_covering is False."
                )
            self.bool_node_lb = False
        self.original_obj = 0.0
        self.hc = 0.0
        self.pc = 0.0
        self.tc = 0.0
        self.ud = 0.0
        self.nf = 0.0
        self.ef = 0.0
        self.var_idx = None
        self.dual_index_for_RMP = {
            "transportation_capacity": dict(),
            "node_capacity": dict(),
        }
        self.index_for_dual_var = 0  # void bugs of index out of range

        # for remote
        self.columns_helpers = None

        self.args_modeling = (bool_dp, model_name, gap, limit, threads, logging)

    def model_reset(self):
        self.model.reset()

    def get_model_objval(self):
        return self.model.objval

    def get_model_status(self):
        return self.model.status

    def modeling(self):
        bool_dp, model_name, gap, limit, threads, logging, *_ = self.args_modeling
        if bool_dp:
            raise ValueError("not implemented")
        else:
            self.modeling_milp(model_name, gap, limit, threads, logging)

    #####################

    def modeling_milp(self, model_name, gap, limit, threads, logging):
        """
        build DNP model
        """
        if self.solver_name == "COPT":
            self.solver = CoptWrapper.CoptWrapper(model_name)
        elif self.solver_name == "GUROBI":
            self.solver = GurobiWrapper.GurobiWrapper(model_name)
        else:
            raise ValueError("solver must be either COPT or GUROBI")

        # self.env = cp.Envr("pricing_env")
        # self.model = self.env.createModel(model_name)
        self.env = self.solver.ENVR
        self.model = self.solver.model

        self.model.setParam(self.solver_constant.Param.Logging, logging)
        self.model.setParam(self.solver_constant.Param.RelGap, gap)
        self.model.setParam(self.solver_constant.Param.TimeLimit, limit)
        if threads is not None:
            self.model.setParam(self.solver_constant.Param.Threads, threads)

        self.variables = dict()  # variables
        self.constrs = dict()  # constraints
        self.obj = dict()  # objective
        # print("add variables ...")
        self.add_vars()

        # print("add constraints ...")
        self.add_constraints()

        # print("set objective ...")
        self.set_objective()

        # for remote
        self.init_col_helpers()

    def _iterate_edges(self):
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            yield edge

    def add_vars(self):
        """
        add variables
        """
        self.var_types = {
            "sku_flow": {
                "lb": 0,
                "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "w",
                "index": "(t, edge, k)",
            },
            "sku_backorder": {
                "lb": 0,
                # "ub": [],  # TBD
                "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "s",
                "index": "(t, k)",
            },
            "select_edge": {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.BINARY,
                "nameprefix": "p",
                "index": "(t, edge)",
            },
        }

        # generate index tuple
        idx = dict()
        for vt in self.var_types.keys():
            idx[vt] = list()
        for edge in self._iterate_edges():
            for t in range(self.T):
                # select edge (i,j) at t
                idx["select_edge"].append((t, edge))

                for k in self.sku_list:
                    # flow of sku k on edge (i,j) at t
                    idx["sku_flow"].append((t, edge, k))
        for t in range(self.T):
            for k in self.sku_list:
                idx["sku_backorder"].append((t, k))

        # for initialization in CG
        self.var_idx = {}
        for var in idx.keys():
            self.var_idx[var] = {key: 0 for key in idx[var]}
        ##########################
        # add variables
        for vt, param in self.var_types.items():
            # self.variables[vt] = self.model.addVars(
            self.variables[vt] = self.solver.addVars(
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
            },
            "transportation_capacity": {"index": "(t, edge)"},
        }

        if self.bool_edge_lb:
            self.constr_types["transportation_variable_lb"] = {"index": "(t, edge)"}

        for constr in self.constr_types.keys():
            self.constrs[constr] = dict()

        for constr in self.constr_types["open_relationship"].keys():
            self.constrs["open_relationship"][constr] = dict()

        # for t in tqdm(range(self.T)):
        for t in range(self.T):
            # initial status and flow conservation
            self.add_constr_flow_conservation(t)

            self.add_constr_open_relationship(t)

            self.add_constr_transportation_capacity(t)

    def add_constr_flow_conservation(self, t: int):
        """flow constraint in the pricing problem
        for each exclusive customer
        """
        edges = list(self._iterate_edges())
        for k in self.sku_list:
            constr_name = f"flow_conservation_{t}_{k.idx}"

            # constr = self.model.addConstr(
            constr = self.solver.addConstr(
                self.variables["sku_flow"].sum(t, edges, k)
                + self.variables["sku_backorder"][t, k]
                == self.variables["sku_backorder"].get((t - 1, k), 0)
                + self.customer.demand.get((t, k), 0),
                name=constr_name,
            )

            self.constrs["flow_conservation"][(t, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):
        return

    def add_constr_transportation_capacity(self, t: int, verbose=False):
        for edge in self._iterate_edges():
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")

            # variable lower bound
            if self.bool_edge_lb and edge.variable_lb < np.inf:
                self.constrs["transportation_variable_lb"][
                    (t, edge)
                    # ] = self.model.addConstr(
                ] = self.solver.addConstr(
                    flow_sum
                    >= edge.variable_lb * self.variables["select_edge"][t, edge],
                    name=f"edge_lb_{t}_{edge.start}_{edge.end}",
                )

                self.index_for_dual_var += 1

            # capacity constraint
            if edge.capacity < np.inf:
                bound = self.variables["select_edge"][t, edge]

                self.constrs["transportation_capacity"][
                    (t, edge)
                    # ] = self.model.addConstr(
                ] = self.solver.addConstr(
                    flow_sum <= edge.capacity * bound,
                    name=f"edge_capacity_{t}_{edge}",
                )
                self.dual_index_for_RMP["transportation_capacity"][
                    edge
                ] = self.index_for_dual_var
                self.index_for_dual_var += 1

        return

    def get_original_objective(self):
        """
        Get the original objective value
        """
        obj = 0.0
        hc = 0.0
        pc = 0.0
        tc = 0.0
        ud = 0.0
        nf = 0.0
        ef = 0.0
        # for t in tqdm(range(self.T)):
        for t in range(self.T):
            tc += self.cal_sku_transportation_cost(t)
            ud += self.cal_sku_unfulfill_demand_cost(t)

        if self.bool_fixed_cost:
            # nf = self.cal_fixed_node_cost()
            # ef = self.cal_fixed_edge_cost()
            pass

        obj = pc + tc + ud + nf + ef

        return obj, hc, pc, tc, ud, nf, ef

    def extra_objective(self, customer, dual_packs=None):
        if dual_packs is None:
            return 0.0
        dual, dual_ws = dual_packs
        obj = sum(
            self.variables["sku_flow"].get((t, ee, k), 0) * v
            for (ee, k, t), v in dual.items()
        )
        obj -= dual_ws[customer]

        return obj

    def update_objective(self, customer, dual_packs):
        """
        Use dual variables to calculate the reduced cost
        """

        obj = self.original_obj + self.extra_objective(customer, dual_packs)

        # self.model.setObjective(obj, sense=self.solver_constant.MINIMIZE)
        self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)

    def set_objective(self):
        self.obj_types = {
            "sku_transportation_cost": {"index": "(t, edge)"},
            "unfulfill_demand_cost": {"index": "(t, c, k)"},
        }

        for obj in self.obj_types.keys():
            self.obj[obj] = dict()

        (
            self.original_obj,
            self.hc,
            self.pc,
            self.tc,
            self.ud,
            self.nf,
            self.ef,
        ) = self.get_original_objective()

        # self.model.setObjective(self.original_obj, sense=self.solver_constant.MINIMIZE)
        self.solver.setObjective(self.original_obj, sense=self.solver_constant.MINIMIZE)

        return

    def cal_sku_transportation_cost(self, t: int):
        transportation_cost = 0.0

        for edge in self._iterate_edges():
            edge_transportation_cost = 0.0

            (
                sku_list_with_fixed_transportation_cost,
                sku_list_with_unit_transportation_cost,
            ) = edge.get_edge_sku_list_with_transportation_cost(t, self.sku_list)
            for k in sku_list_with_fixed_transportation_cost:
                if (
                    edge.transportation_sku_fixed_cost is not None
                    and k in edge.transportation_sku_fixed_cost
                ):
                    edge_transportation_cost = (
                        edge_transportation_cost
                        + edge.transportation_sku_fixed_cost[k]
                        * self.variables["sku_select_edge"].get((t, edge, k), 0)
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
                    + transportation_sku_unit_cost
                    * self.variables["sku_flow"].get((t, edge, k), 0)
                )

            transportation_cost = transportation_cost + edge_transportation_cost

            self.obj["sku_transportation_cost"][(t, edge)] = edge_transportation_cost

        return transportation_cost

    def cal_sku_unfulfill_demand_cost(self, t: int):
        unfulfill_demand_cost = 0.0

        if self.customer.has_demand(t):
            for k in self.customer.demand_sku[t]:
                if self.customer.unfulfill_sku_unit_cost is not None:
                    unfulfill_sku_unit_cost = self.customer.unfulfill_sku_unit_cost[
                        (t, k)
                    ]
                else:
                    unfulfill_sku_unit_cost = self.arg.unfulfill_sku_unit_cost

                unfulfill_node_sku_cost = (
                    unfulfill_sku_unit_cost * self.variables["sku_backorder"][(t, k)]
                )

                unfulfill_demand_cost = unfulfill_demand_cost + unfulfill_node_sku_cost

                self.obj["unfulfill_demand_cost"][(t, k)] = unfulfill_node_sku_cost

        return unfulfill_demand_cost

    def cal_fixed_node_cost(self, t):
        return 0
        pass

    def cal_fixed_edge_cost(self, t):
        return 0
        pass

    def solve(self):
        # self.model.solve()
        self.solver.solve()

    def write(self, name):
        # self.model.write(name)
        self.solver.write(name)

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        pass

    @staticmethod
    def _query_a_expr_or_float_or_variable(v):
        if isinstance(v, float):
            return v
        return v.getValue()

    def eval_helper(self):
        _vals = {}
        _vals["sku_flow"] = {k: v.x for k, v in self.variables["sku_flow"].items()}
        # for t in range(T):
        #     for attr in ATTR_IN_RMP:
        #         if col_helper[attr][t] != {}:
        #             for k, v in col_helper[attr][t].items():
        #                 if type(v) is not float:
        #                     if v.getValue() > 0:
        #                         _vals[attr][t][k] = v.getValue()
        # _vals[attr][t][k] = v.getValue()
        # _vals = {
        #     t: {attr: {k: v.getValue() for k, v in col_helper[attr][t].items()}
        #     for attr in ATTR_IN_RMP} for t in range(7)}
        _vals["beta"] = self._query_a_expr_or_float_or_variable(
            self.columns_helpers["beta"]
        )
        return _vals

    def init_col_helpers(self):
        """
        Initialize the column helpers to extract frequently
            needed quantities for the subproblems

        """
        col_helper = {}

        try:
            col_helper["beta"] = self.original_obj.getExpr()
        except:
            logger.warning(f"customer {self.customer} has null objective")
            col_helper["beta"] = 0.0

        self.columns_helpers = col_helper
        self.columns = []

    def query_columns(self):
        new_col = self.eval_helper()

        # visualize this column
        # oracle = cg_object.oracles[customer]
        # if CG_EXTRA_DEBUGGING:
        #     flow_records = []
        #     for e in self.network.edges:
        #         edge = self.network.edges[e]["object"]
        #         for t in range(cg_object.oracles[customer].T):
        #             edge_sku_list = edge.get_edge_sku_list(t, cg_object.full_sku_list)
        #             for k in edge_sku_list:
        #                 try:
        #                     if oracle.variables["sku_flow"][(t, edge, k)].x != 0:
        #                         flow_records.append(
        #                             {
        #                                 "c": customer.idx,
        #                                 "start": edge.start.idx,
        #                                 "end": edge.end.idx,
        #                                 "sku": k.idx,
        #                                 "t": t,
        #                                 "qty": oracle.variables["sku_flow"][(t, edge, k)].x,
        #                             }
        #                         )
        #                 except:
        #                     pass

        #         new_col["records"] = flow_records
        #         if CG_EXTRA_VERBOSITY:
        #             df = pd.DataFrame.from_records(flow_records).set_index(["c", "col_id"])

        return new_col
