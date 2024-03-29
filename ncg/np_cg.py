import argparse
import json
import logging
import os
import pickle
from typing import *
from typing import List

# import gurobipy as gp
import coptpy
import coptpy as cp
import networkx as nx
import numpy as np
import pandas as pd
import ray
from coptpy import COPT
from gurobipy import GRB
from tqdm import tqdm

import const as const
import dnp_model
import ncg.cg_col_helper as cg_col_helper
import ncg.cg_init as cg_init
import utils as utils
from config.network import construct_network, get_pred_reachable_nodes
from config.param import Param
from config.read_data import read_data
from entity import SKU, Customer

CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 1))

class NetworkColumnGeneration:
    def __init__(
        self,
        arg: argparse.Namespace,
        network: nx.DiGraph,
        customer_list: List[Customer],
        full_sku_list: List[SKU] = None,
        max_iter=500,
        init_primal=None,
        init_sweeping=True,
        init_dual=None,
        init_ray=False,
        num_workers=8,
        num_cpus=8,
    ) -> None:
        self._logger = utils.logger
        self._logger.setLevel(
            logging.DEBUG if cg_col_helper.CG_EXTRA_VERBOSITY else logging.INFO
        )
        self._logger.info(
            f"the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: {cg_col_helper.CG_EXTRA_DEBUGGING}"
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
        self.cus_num = len(self.customer_list)
        self.subgraph = {}  # Dict[customer, nx.DiGraph]
        self.columns = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.columns_helpers = {}  # Dict[customer, List[tuple(x, y, p)]]
        # self.oracles: Dict[Customer, dnp_model.DNP] = {}
        self.oracles: Dict[Customer, ray.actor.ActorHandle] = {}
        # @note: new,
        #   the oracle extension saves extra constraints for primal feasibility
        #   every time it is used, remove this
        self.vars_basis, self.cons_basis = None, None
        self.bool_edge_lb = self.arg.edge_lb
        self.bool_node_lb = self.arg.node_lb
        self.bool_fixed_cost = self.arg.fixed_cost
        self.bool_covering = self.arg.covering
        self.bool_capacity = self.arg.capacity
        self.add_in_upper = self.arg.add_in_upper
        self.dual_index = dict()
        self.vars = dict()  # variables
        self.iter = 0
        self.max_iter = max_iter
        self.red_cost = np.zeros((max_iter, len(customer_list)))
        self.full_network = self.network.copy()
        self.fix_val_constr = {}
        self.init_primal = init_primal
        self.init_dual = init_dual
        self.init_sweeping = init_sweeping
        self.RMP_constrs = dict()
        self.rmp_capacity_cnstr = []
        self.init_ray = init_ray
        self.num_workers = num_workers
        self.num_cpus = num_cpus
        if num_workers < 1:
            raise ValueError("num_workers must be greater than 0")
        if self.num_workers > self.cus_num:
            self.num_workers = self.cus_num
        self.worker_list = []
        self.worker_cus_dict = {}
        if self.init_ray:
            if ray.is_initialized():
                ray.shutdown()
            ray.init(num_cpus=num_cpus)

    def get_subgraph(self):
        """
        Get a subgraph for each customer from the original graph
        """
        for customer in self.customer_list:
            cus_sku_list = []
            for t in range(self.arg.T):
                cus_sku_list = list(
                    set().union(
                        cus_sku_list, customer.get_node_sku_list(t, self.full_sku_list)
                    )
                )
            # cus_sku_list = customer.get_node_sku_list(0, self.full_sku_list)
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
                final_sku_node_list = []
                for t in range(self.arg.T):
                    final_sku_node_list = list(
                        set().union(
                            final_sku_node_list,
                            node.get_node_sku_list(t, self.full_sku_list),
                        )
                    )
                if bool(final_sku_node_list):
                    # if not set(node.get_node_sku_list(0, self.full_sku_list)) & set(
                    #         cus_sku_list
                    # ):
                    if not set(final_sku_node_list) & set(cus_sku_list):
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

    def construct_worker(self):
        cus_per_worker = int(np.ceil(self.cus_num / self.num_workers))
        for customer, n in zip(self.customer_list, range(self.cus_num)):
            if n % cus_per_worker == 0:
                cus_list = self.customer_list[n : min(n + cus_per_worker, self.cus_num)]
                worker = dnp_model.DNP_worker.remote(
                    cus_list,
                    self.arg,
                )
                self.worker_list.append(worker)
            cus_worker_id = n // cus_per_worker
            self.worker_cus_dict[customer] = cus_worker_id

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
        # arg.T = 7
        env_name = customer.idx + "_oracle_env"
        model_name = customer.idx + "_oracle"

        if self.init_ray:
            # oracle = dnp_model.DNP.remote(
            #     arg,
            #     subgraph,
            #     full_sku_list,
            #     env_name,
            #     model_name,
            #     bool_covering=self.bool_covering,
            #     bool_capacity=True,
            #     bool_edge_lb=self.bool_edge_lb,
            #     bool_node_lb=self.bool_node_lb,
            #     cus_list=[customer],
            # )  # for initial column, set obj = 0
            oracle = self.worker_cus_dict[customer]

        else:
            oracle = dnp_model.DNP(
                arg,
                subgraph,
                full_sku_list,
                env_name,
                model_name,
                env=self.RMP_env,
                cus_list=[customer],
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
        # self.RMP_model.write("out/rmpmodel.lp")
        # m = gp.read("out/rmpmodel.lp")
        # m.setParam('TimeLimit', 3600)
        # m.setParam('MIPGap', 0.01)
        # m.optimize()
        self.RMP_model.solve()
        self.RMP_model.setParam(COPT.Param.Logging, 0)

    def subproblem(self, customer: Customer, col_ind, dual_vars=None, dual_index=None):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

        added = False  # whether a new column is added
        # oracle = self.oracles[customer]

        if self.init_ray:
            oracle = self.worker_list[self.worker_cus_dict[customer]]
            ray.get(oracle.model_reset.remote(customer))
            ray.get(oracle.update_objective.remote(customer, dual_vars, dual_index))
            ray.get(oracle.solve.remote(customer))
            v = ray.get(oracle.get_model_objval.remote(customer))
        else:
            oracle = self.oracles[customer]
            oracle.model.reset()
            oracle.update_objective(customer, dual_vars, dual_index)
            oracle.solve()
            v = oracle.model.objval

        self.red_cost[self.iter, col_ind] = v
        added = v < -1e-9 or dual_vars is None

        if self.init_ray:
            # new_col = ray.get(oracle.query_columns.remote())
            new_col = ray.get(oracle.query_columns.remote(customer))
        else:
            new_col = oracle.query_columns()

        self.columns[customer].append(new_col)
        if cg_col_helper.CG_EXTRA_VERBOSITY:
            if self.init_ray:
                _xval = pd.Series(
                    {v.name: v.x for v in oracle.model.getVars.remote() if v.x > 0}
                )
            else:
                _xval = pd.Series(
                    {v.name: v.x for v in oracle.model.getVars() if v.x > 0}
                )

            if self.init_ray:
                _cost = pd.Series(
                    {v.name: v.obj for v in oracle.model.getVars.remote() if v.x > 0}
                )
            else:
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
        # initialize column helpers
        with utils.TimerContext(self.iter, f"initialize_columns"):
            if self.init_ray:
                self.construct_worker()
                ray.get(
                    [
                        worker.construct_DNPs.remote(self.subgraph)
                        for worker in self.worker_list
                    ]
                )

            for customer in tqdm(self.customer_list):
                self.oracles[customer] = self.construct_oracle(customer)
                self.columns[customer] = []

        # use the sweeping method to initialize
        cg_init.primal_sweeping_method(self)
        with utils.TimerContext(self.iter, f"initialize_rmp"):
            self.init_RMP()
        self.RMP_model.setParam(COPT.Param.Logging, 1)

        self._logger.info("initialization of restricted master finished")
        self._logger.info("solving the first rmp")
        while True:  # may need to add a termination condition
            try:
                bool_early_stop = False
                with utils.TimerContext(self.iter, f"solve_rmp"):
                    self.solve_RMP()

                if self.RMP_model.status == COPT.INFEASIBLE:
                    self._logger.info("initial column of RMP is infeasible")
                    self.RMP_model.write(f"rmp@{self.iter}.lp")
                    self.RMP_model.computeIIS()
                    self.RMP_model.write(f"rmp@{self.iter}.iis")
                    break
                with utils.TimerContext(self.iter, f"get_duals"):
                    ######################################
                    rmp_dual_vars = self.RMP_model.getDuals()
                    dual_vars = rmp_dual_vars
                    dual_index = self.dual_index
                ######################################
                added = False
                with utils.TimerContext(self.iter, f"solve_columns"):
                    # modify for parallel
                    if self.init_ray:
                        for worker in tqdm(self.worker_list, ncols=80, leave=False):
                            worker.model_reset_all.remote()
                            worker.update_objective_all.remote(dual_vars, dual_index)
                            worker.solve_all.remote()
                    else:
                        for col_ind, customer in tqdm(
                            enumerate(self.customer_list), ncols=80, leave=False
                        ):
                            oracle = self.oracles[customer]
                            oracle.model.reset()
                            oracle.update_objective(customer, dual_vars, dual_index)
                            oracle.solve()

                    if self.init_ray:
                        all_new_cols = ray.get(
                            [
                                worker.query_all_columns.remote()
                                for worker in self.worker_list
                            ]
                        )
                        all_v = ray.get(
                            [
                                worker.get_all_model_objval.remote()
                                for worker in self.worker_list
                            ]
                        )
                        all_model_status_list = ray.get(
                            [
                                worker.get_all_model_status.remote()
                                for worker in self.worker_list
                            ]
                        )

                        new_cols = []
                        v = []
                        model_status_list = []
                        for new_col, _v, _model_status in zip(
                            all_new_cols, all_v, all_model_status_list
                        ):
                            new_cols.extend(new_col)
                            v.extend(_v)
                            model_status_list.extend(_model_status)

                    else:
                        new_cols = [
                            oracle.query_columns() for oracle in self.oracles.values()
                        ]
                        v = [oracle.model.objval for oracle in self.oracles.values()]
                        model_status_list = [
                            oracle.model.status for oracle in self.oracles.values()
                        ]

                    for col_ind, customer in enumerate(self.customer_list):
                        self.red_cost[self.iter, col_ind] = v[col_ind]
                        added = (v[col_ind] < -1e-9 or dual_vars is None) or added
                        new_col = new_cols[col_ind]
                        self.columns[customer].append(new_col)

                        model_status = model_status_list[col_ind]
                        if (
                            # self.oracles[customer].model.status
                            # ray.get(self.oracles[customer].get_model_status.remote())
                            model_status
                            == coptpy.COPT.INTERRUPTED
                        ):
                            bool_early_stop = True
                            self._logger.info("early terminated")
                            break
                    # modify for parallel

                self.iter += 1

                self._logger.info(
                    f"k: {self.iter:5d} / {self.max_iter:d} f: {self.RMP_model.objval:.6e}, c': {np.min(self.red_cost[self.iter - 1, :]):.4e}",
                )

                if not added or self.iter >= self.max_iter:
                    self.red_cost = self.red_cost[: self.iter, :]
                    break

                if bool_early_stop:
                    self._logger.info("early terminated")
                    break
                with utils.TimerContext(self.iter, f"update_rmp"):
                    self.update_RMP_by_cols()

            except KeyboardInterrupt as _unused_e:
                self._logger.info("early terminated")
                break
        utils.visualize_timers()
        self._logger.info(f"save solutions to {utils.CONF.DEFAULT_SOL_PATH}")
        self.get_solution(utils.CONF.DEFAULT_SOL_PATH)

    def update_RMP_by_cols(self):
        """
        Incrementally update the RMP with new columns
        """

        self.RMP_model.reset()

        for customer in self.customer_list:
            edge_capacity_cons_coef = []
            node_capacity_cons_coef = []
            # edge_capacity_cons_name = []
            # node_capacity_cons_name = []
            # capacity_cons_name = []
            for t in range(self.arg.T):
                for e in self.network.edges:
                    edge = self.network.edges[e]["object"]
                    if edge.capacity < np.inf:
                        # for debug
                        # if edge.idx == "T0003_C00120540" and t == 0:
                        #     print(self.columns[customer][-1]["sku_flow_sum"][t].keys())
                        # for debug

                        edge_capacity_cons_coef.append(
                            (
                                # self.columns[customer][-1]["sku_flow_sum"][t][edge.idx]
                                self.columns[customer][-1]["sku_flow_sum"][t][edge]
                                # if e in self.subgraph[customer].edges
                                if e in self.subgraph[customer].edges
                                else 0.0
                            )
                        )
                        # edge_capacity_cons_name.append((t, edge))

                for node in self.network.nodes:
                    if node.type == const.PLANT:
                        if node.production_capacity < np.inf:
                            node_capacity_cons_coef.append(
                                (
                                    self.columns[customer][-1]["sku_production_sum"][t][
                                        # node.idx
                                        node
                                    ]
                                    # if node in self.subgraph[customer].nodes
                                    if node in self.subgraph[customer].nodes
                                    else 0.0
                                )
                            )
                            # node_capacity_cons_name.append((t, node))
                    elif node.type == const.WAREHOUSE:
                        if node.inventory_capacity < np.inf:
                            node_capacity_cons_coef.append(
                                (
                                    self.columns[customer][-1]["sku_inventory_sum"][t][
                                        # node.idx
                                        node
                                    ]
                                    # if node in self.subgraph[customer].nodes
                                    if node in self.subgraph[customer].nodes
                                    else 0.0
                                )
                            )
                            # node_capacity_cons_name.append((t, node))
            capacity_cons_coef = [*edge_capacity_cons_coef, *node_capacity_cons_coef]
            # capacity_cons_name = [*edge_capacity_cons_name, *node_capacity_cons_name]
            new_col = cp.Column(
                [*self.rmp_capacity_cnstr, self.RMP_constrs["weights_sum"][customer]],
                [*capacity_cons_coef, 1.0],
            )

            self.vars["column_weights"][
                customer, self.columns[customer].__len__() - 1
            ] = self.RMP_model.addVar(
                obj=self.columns[customer][-1]["beta"],
                name=f"lambda_{customer.idx}_{len(self.columns[customer])}",
                lb=0.0,
                ub=1.0,
                column=new_col,
            )

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
            self.vars[vt] = self.RMP_model.addVars(
                idx[vt],
                lb=param["lb"],
                ub=param["ub"],
                vtype=param["vtype"],
                nameprefix=f"{param['nameprefix']}_",
            )

        ################# add constraints #################
        constr_types = {
            "transportation_capacity": {"index": "(t, edge)"},
            "node_capacity": {"index": "(t,node)"},
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
        for t in range(self.arg.T):
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
                        transportation += self.vars["column_weights"][
                            customer, number
                        ] * (
                            # self.columns[customer][number]["sku_flow_sum"][t][edge]
                            # mismatch bug of edge as keys, use idx instead
                            # self.columns[customer][number]["sku_flow_sum"][t][edge.idx]
                            self.columns[customer][number]["sku_flow_sum"][t][edge]
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
                    name=f"transportation_capacity_{t, edge.idx}",
                )
                self.RMP_constrs["transportation_capacity"][(t, edge)] = constr
                self.dual_index["transportation_capacity"][(t, edge)] = index
                index += 1

            for node in self.network.nodes:
                # node production capacity
                if node.type == const.PLANT:
                    if node.production_capacity == np.inf:
                        continue
                    production = 0.0
                    for customer in self.customer_list:
                        for number in range(len(self.columns[customer])):
                            production += self.vars["column_weights"][
                                customer, number
                            ] * (
                                self.columns[customer][number]["sku_production_sum"][t][
                                    # node
                                    # mismatch bug of node as keys, use idx instead
                                    # node.idx
                                    node
                                ]
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
                    self.RMP_constrs["node_capacity"][(t, node)] = constr

                elif node.type == const.WAREHOUSE:
                    # node holding capacity
                    if node.inventory_capacity == np.inf:
                        continue
                    holding = 0.0
                    for customer in self.customer_list:
                        for number in range(len(self.columns[customer])):
                            holding += self.vars["column_weights"][customer, number] * (
                                self.columns[customer][number]["sku_inventory_sum"][t][
                                    # node
                                    # mismatch bug of node as keys, use idx instead
                                    # node.idx
                                    node
                                ]
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
                        name=f"inventory_capacity_{t, node.idx}",
                    )
                    # self.RMP_constrs["holding_capacity"][node] = constr
                    self.RMP_constrs["node_capacity"][(t, node)] = constr

                else:
                    continue

                self.dual_index["node_capacity"][(t, node)] = index
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
        self.rmp_capacity_cnstr = [
            *self.RMP_constrs["transportation_capacity"].values(),
            *self.RMP_constrs["node_capacity"].values(),
        ]

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        cus_col_value = pd.DataFrame(
            index=range(self.iter), columns=[c.idx for c in self.customer_list]
        )
        cus_col_weights = pd.DataFrame(
            index=range(self.iter), columns=[c.idx for c in self.customer_list]
        )
        reduced_cost = pd.DataFrame(
            self.red_cost, columns=[c.idx for c in self.customer_list]
        )

        for customer in self.customer_list:
            # for number in range(self.num_cols):
            for number in range(self.iter):
                if self.columns[customer][number] != {}:
                    cus_col_value.loc[number, customer.idx] = self.columns[customer][
                        number
                    ]["beta"]
                    cus_col_weights.loc[number, customer.idx] = self.vars[
                        "column_weights"
                    ][customer, number].value
                else:
                    cus_col_value.loc[number, customer.idx] = 0
                    cus_col_weights.loc[number, customer.idx] = 0

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
