import pickle

import coptpy
import coptpy as cp
from coptpy import COPT
import const
import utils
from utils import get_in_edges, get_out_edges
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from entity import SKU, Customer, Warehouse, Plant, Edge
from tqdm import tqdm
import os
from network import constuct_network, get_pred_reachable_nodes
from read_data import read_data
from dnp_model import DNP
from param import Param

# macro for debugging
CG_EXTRA_VERBOSITY = os.environ.get("CG_EXTRA_VERBOSITY", 0)


class NP_CG:
    def __init__(
        self,
        arg: argparse.Namespace,
        network: nx.DiGraph,
        customer_list: List[Customer],
        full_sku_list: List[SKU] = None,
        open_relationship=False,
        max_iter=500,
    ) -> None:
        self._logger = utils.logger
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
        self.oracles = {}  # Dict[customer, copt.model]
        self.open_relationship = open_relationship
        self.dual_index = dict()
        self.vars = dict()  # variables
        self.num_cols = 0
        self.max_iter = max_iter
        self.red_cost = np.zeros((max_iter, len(customer_list)))

    def get_subgraph(self):
        """
        Get a subgraph for each customer from the original graph
        """
        for customer in self.customer_list:
            cus_sku_list = customer.get_node_sku_list(0, self.full_sku_list)
            pred_reachable_nodes = set()
            get_pred_reachable_nodes(self.network, customer, pred_reachable_nodes)
            related_nodes = pred_reachable_nodes.copy()
            for node in pred_reachable_nodes:
                # todo, previous
                # if bool(node.get_node_sku_list(0, sku_list)):
                if bool(node.get_node_sku_list(0, self.full_sku_list)):
                    if not set(node.get_node_sku_list(0, self.full_sku_list)) & set(
                        cus_sku_list
                    ):
                        related_nodes.remove(node)
                else:
                    related_nodes.remove(node)
            related_nodes.add(customer)
            self.subgraph[customer] = self.network.subgraph(related_nodes)
            self.subgraph[customer].graph["sku_list"] = cus_sku_list

    def construct_oracle(self, customer: Customer):
        """
        Construct oracles for each customer
        """
        subgraph = self.subgraph[customer]
        full_sku_list = subgraph.graph["sku_list"]
        arg = self.arg
        arg.T = 1
        env_name = customer.idx + "_oracle_env"
        model_name = customer.idx + "_oracle"

        return DNP(
            arg,
            subgraph,
            full_sku_list,
            env_name,
            model_name,
            self.open_relationship,
            # False,
            True,
            True,
            # len(self.customer_list),
            1,
            env=self.RMP_env,
        )  # for initial column, set obj = 0
        # return DNP(arg, subgraph, full_sku_list, env_name, model_name, self.open_relationship, False, False) # for initial column, set obj to be the original obj

    def init_cols(self, customer: Customer):
        """
        Initialize the columns for the subproblem according to the oracles
        """

        oracle = self.oracles[customer]
        oracle.modeling()
        oracle.solve()
        init_col = {
            "beta": 0,
            "sku_flow_sum": {},
            "sku_production_sum": {},
            "sku_inventory_sum": {},
        }
        self.columns[customer] = [init_col]

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
            "production_capacity": {"index": "(node)"},
            "holding_capacity": {"index": "(node)"},
            "weights_sum": {"index": "(customer)"},
        }
        constrs = dict()
        for constr in constr_types.keys():
            constrs[constr] = dict()

        self.dual_index = {
            "transportation_capacity": dict(),
            "node_capacity": dict(),
            "weights_sum": dict(),
        }
        index = 0

        # edge transportation capacity
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            if edge.capacity == np.inf:
                continue
            transportation = 0.0
            for customer in self.customer_list:
                if e in self.subgraph[customer].edges:  # can we do better?
                    self.columns[customer][len(self.columns[customer]) - 1][
                        "sku_flow_sum"
                    ][edge] = (
                        self.oracles[customer]
                        .vars["sku_flow"]
                        .sum(0, edge, "*")
                        .getValue()
                    )
                    for number in range(len(self.columns[customer])):
                        transportation += (
                            self.vars["column_weights"][customer, number]
                            * self.columns[customer][number]["sku_flow_sum"][edge]
                        )

            if type(transportation) == float:
                continue
            sumcoef = 0
            for i in range(transportation.getSize()):
                sumcoef += transportation.getCoeff(i)
            if sumcoef == 0:
                continue

            constr = self.RMP_model.addConstr(
                transportation <= edge.capacity,
                name=f"transportation_capacity_{edge.idx}",
            )
            constrs["transportation_capacity"][edge] = constr
            self.dual_index["transportation_capacity"][edge] = index
            index += 1

        for node in self.network.nodes:
            # node production capacity
            if node.type == const.PLANT:
                if node.production_capacity == np.inf:
                    continue
                production = 0.0
                for customer in self.customer_list:
                    if node in self.subgraph[customer].nodes:
                        self.columns[customer][len(self.columns[customer]) - 1][
                            "sku_production_sum"
                        ][node] = (
                            self.oracles[customer]
                            .vars["sku_production"]
                            .sum(0, node, "*")
                            .getValue()
                        )
                        for number in range(len(self.columns[customer])):
                            production += (
                                self.vars["column_weights"][customer, number]
                                * self.columns[customer][number]["sku_production_sum"][
                                    node
                                ]
                            )

                if type(production) == float:
                    continue
                sumcoef = 0
                for i in range(production.getSize()):
                    sumcoef += production.getCoeff(i)
                if sumcoef == 0:
                    continue

                constr = self.RMP_model.addConstr(
                    production <= node.production_capacity,
                    name=f"production_capacity_{node.idx}",
                )
                constrs["production_capacity"][node] = constr

            elif node.type == const.WAREHOUSE:
                # node holding capacity
                if node.inventory_capacity == np.inf:
                    continue
                holding = 0.0
                for customer in self.customer_list:
                    if node in self.subgraph[customer].nodes:
                        self.columns[customer][len(self.columns[customer]) - 1][
                            "sku_inventory_sum"
                        ][node] = (
                            self.oracles[customer]
                            .vars["sku_inventory"]
                            .sum(0, node, "*")
                            .getValue()
                        )
                        for number in range(len(self.columns[customer])):
                            holding += (
                                self.vars["column_weights"][customer, number]
                                * self.columns[customer][number]["sku_inventory_sum"][
                                    node
                                ]
                            )

                if type(holding) == float:
                    continue
                sumcoef = 0
                for i in range(holding.getSize()):
                    sumcoef += holding.getCoeff(i)
                if sumcoef == 0:
                    continue

                constr = self.RMP_model.addConstr(
                    holding <= node.inventory_capacity,
                    name=f"inventory_capacity_{node.idx}",
                )
                constrs["holding_capacity"][node] = constr
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
            constrs["weights_sum"][customer] = constr

            self.dual_index["weights_sum"][customer] = index
            index += 1

        ################# set objective #################
        obj = 0
        for customer in self.customer_list:
            self.columns[customer][len(self.columns[customer]) - 1]["beta"] = (
                self.oracles[customer].original_obj.getExpr().getValue()
            )
            for number in range(len(self.columns[customer])):
                obj += (
                    self.vars["column_weights"][customer, number]
                    * self.columns[customer][number]["beta"]
                )

        self.RMP_model.setObjective(obj, COPT.MINIMIZE)

    def solve_RMP(self):
        """
        Solve the RMP and get the dual variables to construct the subproblem
        """
        self.RMP_model.solve()

    def update_RMP(self):
        """
        Update the RMP with new columns
        """

        # can incrementally update the RMP?
        self.RMP_model.clear()
        self.init_RMP()

    def subproblem(self, customer: Customer, dual_vars, col_ind):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

        added = False  # whether a new column is added
        oracle = self.oracles[customer]
        oracle.model.reset()
        oracle.update_objective(dual_vars, self.dual_index)
        oracle.solve()
        # new_column = oracle.vars
        v = oracle.model.objval
        self.red_cost[self.num_cols, col_ind] = v

        if v < 0:
            added = True
            new_col = {
                "beta": 0,
                "sku_flow_sum": {},
                "sku_production_sum": {},
                "sku_inventory_sum": {},
            }
            self.columns[customer].append(new_col)
        if CG_EXTRA_VERBOSITY:
            for e in self.network.edges:
                edge = self.network.edges[e]["object"]
                for t in range(1):
                    edge_sku_list = edge.get_edge_sku_list(t, self.full_sku_list)
                    for k in edge_sku_list:
                        if oracle.vars["sku_flow"][(t, edge, k)].x != 0:
                            data = {
                                "id": edge.idx,
                                "start": edge.start.idx,
                                "end": edge.end.idx,
                                "sku": k.idx,
                                "t": t,
                                "qty": oracle.vars["sku_flow"][(t, edge, k)].x,
                            }
                            print(data)
        return added

    def CG(self):
        """
        The main loop of column generation algorithm
        """

        self.get_subgraph()

        for customer in tqdm(self.customer_list):
            self.oracles[customer] = self.construct_oracle(customer)
            self.init_cols(customer)

            # for test
            if self.oracles[customer].model.status == COPT.INFEASIBLE:
                continue
            else:
                continue

        self.init_RMP()
        for customer in self.customer_list:
            self.oracles[customer].del_constr_for_RMP()

        self._logger.info("Initialization complete, start generating columns...")
        while True:  # may need to add a termination condition
            try:
                bool_early_stop = False
                self.solve_RMP()

                added = False
                for customer, col_ind in zip(
                    self.customer_list, range(len(self.customer_list))
                ):
                    added = (
                        self.subproblem(customer, self.RMP_model.getDuals(), col_ind)
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
                self.update_RMP()
            except KeyboardInterrupt as _unused_e:
                self._logger.info("early terminated")
                break
        self._logger.info(f"save solutions to {utils.CONF.DEFAULT_SOL_PATH}")
        self.get_solution(utils.CONF.DEFAULT_SOL_PATH)

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
            for number in range(len(self.columns[customer])):
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


if __name__ == "__main__":
    pass
