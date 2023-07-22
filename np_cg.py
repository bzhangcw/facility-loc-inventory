import json
import logging
import pickle

import coptpy
import coptpy as cp
from coptpy import COPT
from typing import *

import cg_col_helper
import const
import utils
import cg_init
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from entity import SKU, Customer
from tqdm import tqdm
import os
from network import construct_network, get_pred_reachable_nodes
from read_data import read_data
import dnp_model

# import dnp_model
from param import Param

# macro for debugging
CG_EXTRA_VERBOSITY = os.environ.get("CG_EXTRA_VERBOSITY", 0)
CG_EXTRA_DEBUGGING = os.environ.get("CG_EXTRA_DEBUGGING", 1)


class NetworkColumnGeneration:
    def __init__(
        self,
        arg: argparse.Namespace,
        network: nx.DiGraph,
        customer_list: List[Customer],
        full_sku_list: List[SKU] = None,
        bool_covering=False,
        bool_edge_lb=False,
        bool_node_lb=False,
        max_iter=500,
        init_primal=None,
        init_sweeping=True,
        init_dual=None,
    ) -> None:
        self._logger = utils.logger
        self._logger.setLevel(logging.DEBUG if CG_EXTRA_VERBOSITY else logging.INFO)
        self._logger.info(
            f"the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: {CG_EXTRA_DEBUGGING}"
        )
        self.arg = arg
        self.network = network
        self.full_sku_list = (
            full_sku_list
            if full_sku_list is not None
            else self.network.graph["sku_list"]
        )

        self.RMP_env = cp.Envr("RMP_env")
        self.RMP_model = self.RMP_env.createModel("RMP")
        self.customer_list = customer_list  # List[Customer]
        self.subgraph = {}  # Dict[customer, nx.DiGraph]
        self.columns = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.columns_helpers = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.oracles: Dict[Customer, dnp_model.DNP] = {}
        # @note: new,
        #   the oracle extension saves extra constraints for primal feasibility
        #   every time it is used, remove this
        self.oracles_extension: Dict[Customer, dnp_model.DNP] = {}  #
        self.bool_covering = bool_covering
        self.bool_edge_lb = bool_edge_lb
        self.bool_node_lb = bool_node_lb
        self.dual_index = dict()
        self.vars = dict()  # variables
        self.num_cols = 0
        self.max_iter = max_iter
        self.red_cost = np.zeros((max_iter, len(customer_list)))
        self.full_network = self.network.copy()
        self.fix_val_constr = {}
        self.init_primal = init_primal
        self.init_dual = init_dual
        self.init_sweeping = init_sweeping
        self.RMP_constrs = dict()

    def get_subgraph(self):
        """
        Get a subgraph for each customer from the original graph
        """
        for customer in self.customer_list:
            cus_sku_list = customer.get_node_sku_list(0, self.full_sku_list)
            pred_reachable_nodes = set()
            get_pred_reachable_nodes(self.network, customer, pred_reachable_nodes)
            # @note: reset visited status
            # @update: 070523
            for k in pred_reachable_nodes:
                k.visited = False
            related_nodes = pred_reachable_nodes.copy()
            # @note:
            #   use "node.visited = Bool" to avoid the excessive recursive times
            #   in the situation that two nodes can reach each other
            for node in pred_reachable_nodes:
                if bool(node.get_node_sku_list(0, self.full_sku_list)):
                    if not set(node.get_node_sku_list(0, self.full_sku_list)) & set(
                        cus_sku_list
                    ):
                        # If the sku_list associated with a pred_node
                        # doesn't have any same element with the cus_sku_list,
                        # then remove the pred_node
                        related_nodes.remove(node)
                else:
                    # If a pred_node doesn't store or produce any
                    # SKU in period 0, then remove it
                    related_nodes.remove(node)
            related_nodes.add(customer)

            self.subgraph[customer] = nx.DiGraph(sku_list=self.full_sku_list)

            # can we do better?
            this_subgraph = self.network.subgraph(related_nodes)
            self.subgraph[customer].add_nodes_from(this_subgraph.nodes(data=True))
            self.subgraph[customer].add_edges_from(this_subgraph.edges(data=True))
            self.subgraph[customer].graph["sku_list"] = cus_sku_list

            self._logger.debug(f"{cus_sku_list}")
            self._logger.debug(f"{sorted(k.__str__() for k in related_nodes)}")

        return

    def construct_oracle(
        self,
        customer: Customer,
    ):
        """
        Construct oracles for each customer
        """
        subgraph = self.subgraph[customer]
        full_sku_list = subgraph.graph["sku_list"]
        arg = self.arg
        arg.T = 1
        env_name = customer.idx + "_oracle_env"
        model_name = customer.idx + "_oracle"
        oracle = dnp_model.DNP(
            arg,
            subgraph,
            full_sku_list,
            env_name,
            model_name,
            bool_covering=self.bool_covering,
            bool_capacity=True,
            bool_edge_lb=self.bool_edge_lb,
            bool_node_lb=self.bool_node_lb,
            env=self.RMP_env,
        )  # for initial column, set obj = 0
        oracle.modeling()

        return oracle

    def solve_lp_relaxation(self):
        full_lp_relaxation = dnp_model.DNP(self.arg, self.full_network)
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

        return lp_dual, dual_index

    def solve_RMP(self):
        """
        Solve the RMP and get the dual variables to construct the subproblem
        """
        self.RMP_model.setParam("LpMethod", 2)
        self.RMP_model.setParam("Crossover", 0)

        self.RMP_model.solve()

    def update_RMP_by_cols(self):
        """
        Incrementally update the RMP with new columns
        """

        self.RMP_model.reset()

        for customer in self.customer_list:
            capacity_cons_coef = []
            for e in self.network.edges:
                edge = self.network.edges[e]["object"]
                if edge.capacity < np.inf:
                    capacity_cons_coef.append(
                        (
                            self.columns[customer][-1]["sku_flow_sum"][edge]
                            if e in self.subgraph[customer].edges
                            else 0.0
                        )
                    )

            for node in self.network.nodes:
                if node.type == const.PLANT:
                    if node.production_capacity < np.inf:
                        capacity_cons_coef.append(
                            (
                                self.columns[customer][-1]["sku_production_sum"][node]
                                if node in self.subgraph[customer].nodes
                                else 0.0
                            )
                        )
                elif node.type == const.WAREHOUSE:
                    if node.inventory_capacity < np.inf:
                        capacity_cons_coef.append(
                            (
                                self.columns[customer][-1]["sku_inventory_sum"][node]
                                if node in self.subgraph[customer].nodes
                                else 0.0
                            )
                        )

            capacity_cnstr = []
            capacity_cnstr.append(self.RMP_constrs["transportation_capacity"].values())
            capacity_cnstr.append(self.RMP_constrs["node_capacity"].values())

            new_col = cp.Column(
                capacity_cnstr + [self.RMP_constrs["weights_sum"][customer]],
                capacity_cons_coef + [1.0],
            )

            self.vars["column_weights"].append(
                self.RMP_model.addVar(
                    obj=self.columns[customer][-1]["beta"],
                    name=f"lambda_{customer.idx}_{len(self.columns[customer])}",
                    lb=0.0,
                    ub=1.0,
                    column=new_col,
                )
            )

    def subproblem(self, customer: Customer, col_ind, dual_vars=None, dual_index=None):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

        added = False  # whether a new column is added
        oracle = self.oracles[customer]
        oracle.model.reset()
        oracle.update_objective(customer, dual_vars, dual_index)

        oracle.solve()
        v = oracle.model.objval
        self.red_cost[self.num_cols, col_ind] = v
        # successfully added if dual price < 0
        #   or there is no dual variables
        #   implying initialization
        added = v < -1e-2 or dual_vars is None
        # querying column
        new_col = cg_col_helper.query_columns(self, customer)
        self.columns[customer].append(new_col)
        if CG_EXTRA_VERBOSITY:
            _xval = pd.Series({v.name: v.x for v in oracle.model.getVars() if v.x > 0})
            _cost = pd.Series(
                {v.name: v.obj for v in oracle.model.getVars() if v.x > 0}
            )
            column_debugger = pd.DataFrame({"value": _xval, "objective": _cost})
            print(column_debugger)

        return added

    def run(self):
        """
        The main loop of column generation algorithm
        """

        self.get_subgraph()

        # construct oracles
        self._logger.info("initialization complete, start generating columns...")
        self._logger.info("generating column oracles")
        for customer in tqdm(self.customer_list):
            self.oracles[customer] = self.construct_oracle(customer)
        # initialize column helpers
        cg_col_helper.init_col_helpers(self)

        # use the sweeping method to initialize
        cg_init.primal_sweeping_method(self)
        self.init_RMP()
        self._logger.info("initialization of restricted master finished")
        while True:  # may need to add a termination condition
            try:
                bool_early_stop = False
                self.solve_RMP()

                if self.RMP_model.status == COPT.INFEASIBLE:
                    self._logger.info("Initial column of RMP is infeasible")
                    break
                ######################################
                rmp_dual_vars = self.RMP_model.getDuals()

                dual_vars = rmp_dual_vars
                dual_index = self.dual_index
                ######################################

                added = False
                for col_ind, customer in enumerate(self.customer_list):
                    added = (
                        self.subproblem(customer, col_ind, dual_vars, dual_index)
                        or added
                    )
                    if self.oracles[customer].model.status == coptpy.COPT.INTERRUPTED:
                        bool_early_stop = True
                        self._logger.info("early terminated")
                        break

                self.num_cols += 1

                print(
                    "k: ",
                    "{:5d}".format(self.num_cols),
                    "/",
                    "{:d}".format(self.max_iter),
                    " f: {:.6e}".format(self.RMP_model.objval),
                    " c': %.4e" % np.min(self.red_cost[self.num_cols - 1, :]),
                )

                if not added or self.num_cols >= self.max_iter:
                    self.red_cost = self.red_cost[: self.num_cols, :]
                    break

                if bool_early_stop:
                    self._logger.info("early terminated")
                    break
                # self.update_RMP()
                self.update_RMP_by_cols()

            except KeyboardInterrupt as _unused_e:
                self._logger.info("early terminated")
                break
        self._logger.info(f"save solutions to {utils.CONF.DEFAULT_SOL_PATH}")
        self.get_solution(utils.CONF.DEFAULT_SOL_PATH)

    def init_RMP(self):
        """
        Initialize the RMP with initial columns
        """
        self.RMP_model.setParam(COPT.Param.Logging, 0)

        ################# add variables #################
        self.var_types = {
            "column_weights": {
                "lb": 0,
                "ub": 1,
                "vtype": COPT.CONTINUOUS,
                "nameprefix": "lambda",
                "index": "(customer, number)",
            },
        }

        # generate index tuple
        idx = dict()
        for vt in self.var_types.keys():
            idx[vt] = list()

        for customer in self.customer_list:
            for number in range(len(self.columns[customer])):
                idx["column_weights"].append((customer, number))

        # add variables
        self.vars = dict()
        for vt, param in self.var_types.items():
            # print(f"  - {vt}")
            self.vars[vt] = self.RMP_model.addVars(
                idx[vt],
                lb=param["lb"],
                ub=param["ub"],
                vtype=param["vtype"],
                nameprefix=f"{param['nameprefix']}_",
            )

        ################# add constraints #################
        constr_types = {
            "transportation_capacity": {"index": "(edge)"},
            # "production_capacity": {"index": "(node)"},
            # "holding_capacity": {"index": "(node)"},
            "node_capacity": {"index": "(node)"},
            "weights_sum": {"index": "(customer)"},
        }

        for constr in constr_types.keys():
            self.RMP_constrs[constr] = dict()

        self.dual_index = {
            "transportation_capacity": dict(),
            "node_capacity": dict(),
            "weights_sum": dict(),
        }
        index = 0

        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            if edge.capacity == np.inf:
                continue
            transportation = 0.0
            for customer in self.customer_list:
                # if e in self.subgraph[customer].edges:  # todo: can we do better?
                #     for number in range(len(self.columns[customer])):
                #         transportation += (
                #             self.vars["column_weights"][customer, number]
                #             * self.columns[customer][number]["sku_flow_sum"][edge]
                #         )

                for number in range(len(self.columns[customer])):
                    transportation += self.vars["column_weights"][customer, number] * (
                        self.columns[customer][number]["sku_flow_sum"][edge]
                        if e in self.subgraph[customer].edges
                        else 0.0
                    )

            if type(transportation) == float:
                # continue
                transportation = 0 * self.vars["column_weights"].sum(customer, "*")
            sumcoef = 0
            for i in range(transportation.getSize()):
                sumcoef += transportation.getCoeff(i)
            if sumcoef == 0:
                # continue
                transportation = 0 * self.vars["column_weights"].sum(customer, "*")

            constr = self.RMP_model.addConstr(
                transportation <= edge.capacity,
                name=f"transportation_capacity_{edge.idx}",
            )
            self.RMP_constrs["transportation_capacity"][edge] = constr
            self.dual_index["transportation_capacity"][edge] = index
            index += 1

        for node in self.network.nodes:
            # node production capacity
            if node.type == const.PLANT:
                if node.production_capacity == np.inf:
                    continue
                production = 0.0
                for customer in self.customer_list:
                    # if node in self.subgraph[customer].nodes:
                    #     for number in range(len(self.columns[customer])):
                    #         production += (
                    #             self.vars["column_weights"][customer, number]
                    #             * self.columns[customer][number]["sku_production_sum"][
                    #                 node
                    #             ]
                    #         )

                    for number in range(len(self.columns[customer])):
                        production += self.vars["column_weights"][customer, number] * (
                            self.columns[customer][number]["sku_production_sum"][node]
                            if node in self.subgraph[customer].nodes
                            else 0.0
                        )

                if type(production) == float:
                    # continue
                    production = 0 * self.vars["column_weights"].sum(customer, "*")

                sumcoef = 0
                for i in range(production.getSize()):
                    sumcoef += production.getCoeff(i)
                if sumcoef == 0:
                    # continue
                    production = 0 * self.vars["column_weights"].sum(customer, "*")

                constr = self.RMP_model.addConstr(
                    production <= node.production_capacity,
                    name=f"production_capacity_{node.idx}",
                )
                # self.RMP_constrs["production_capacity"][node] = constr
                self.RMP_constrs["node_capacity"][node] = constr

            elif node.type == const.WAREHOUSE:
                # node holding capacity
                if node.inventory_capacity == np.inf:
                    continue
                holding = 0.0
                for customer in self.customer_list:
                    # if node in self.subgraph[customer].nodes:
                    #     for number in range(len(self.columns[customer])):
                    #         holding += (
                    #             self.vars["column_weights"][customer, number]
                    #             * self.columns[customer][number]["sku_inventory_sum"][
                    #                 node
                    #             ]
                    #         )

                    for number in range(len(self.columns[customer])):
                        holding += self.vars["column_weights"][customer, number] * (
                            self.columns[customer][number]["sku_inventory_sum"][node]
                            if node in self.subgraph[customer].nodes
                            else 0.0
                        )

                if type(holding) == float:
                    # continue
                    holding = 0 * self.vars["column_weights"].sum(customer, "*")

                sumcoef = 0
                for i in range(holding.getSize()):
                    sumcoef += holding.getCoeff(i)
                if sumcoef == 0:
                    # continue
                    holding = 0 * self.vars["column_weights"].sum(customer, "*")

                constr = self.RMP_model.addConstr(
                    holding <= node.inventory_capacity,
                    name=f"inventory_capacity_{node.idx}",
                )
                # self.RMP_constrs["holding_capacity"][node] = constr
                self.RMP_constrs["node_capacity"][node] = constr

            else:
                continue

            self.dual_index["node_capacity"][node] = index
            index += 1

        for customer in self.customer_list:
            # weights sum to 1
            constr = self.RMP_model.addConstr(
                self.vars["column_weights"].sum(customer, "*") == 1,
                name=f"weights_sum_{customer.idx}",
            )
            self.RMP_constrs["weights_sum"][customer] = constr

            self.dual_index["weights_sum"][customer] = index
            index += 1

        ################# set objective #################
        obj = 0
        for customer in self.customer_list:
            for number in range(len(self.columns[customer])):
                obj += (
                    self.vars["column_weights"][customer, number]
                    * self.columns[customer][number]["beta"]
                )

        self.RMP_model.setObjective(obj, COPT.MINIMIZE)

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        cus_col_value = pd.DataFrame(
            index=range(self.num_cols), columns=[c.idx for c in self.customer_list]
        )
        cus_col_weights = pd.DataFrame(
            index=range(self.num_cols), columns=[c.idx for c in self.customer_list]
        )
        reduced_cost = pd.DataFrame(
            self.red_cost, columns=[c.idx for c in self.customer_list]
        )

        for customer in self.customer_list:
            # for number in range(self.num_cols):
            for number in range(self.num_cols):
                cus_col_value.loc[number, customer.idx] = self.columns[customer][
                    number
                ]["beta"]
                cus_col_weights.loc[number, customer.idx] = self.vars["column_weights"][
                    customer, number
                ].value

        num_cus = len(self.customer_list)
        cus_col_value.to_csv(
            os.path.join(data_dir, "cus" + str(num_cus) + "_col_cost.csv"), index=False
        )
        cus_col_weights.to_csv(
            os.path.join(data_dir, "cus" + str(num_cus) + "_col_weight.csv"),
            index=False,
        )
        reduced_cost.to_csv(
            os.path.join(data_dir, "cus" + str(num_cus) + "_reduced_cost.csv"),
            index=False,
        )

        with open(
            os.path.join(data_dir, "cus" + str(num_cus) + "_details.json"), "w"
        ) as f:
            for customer in self.customer_list:
                for col in self.columns[customer]:
                    f.write(json.dumps(col, skipkeys=True))
                    f.write("\n")


if __name__ == "__main__":
    pass
