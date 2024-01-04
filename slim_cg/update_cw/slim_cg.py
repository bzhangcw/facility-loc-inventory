import argparse
import json
import logging
import os
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
from tqdm import tqdm
from config.network import *
import const as const
import utils as utils
from entity import SKU, Customer
from slim_cg.slim_rmp_model import DNPSlim
from slim_cg.slim_pricing import Pricing, PricingWorker, CG_PRICING_LOGGING

from solver_wrapper import GurobiWrapper, CoptWrapper
from solver_wrapper.CoptConstant import CoptConstant
from solver_wrapper.GurobiConstant import GurobiConstant

CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 1))


class NetworkColumnGenerationSlim(object):
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
        init_ray=False,
        num_workers=8,
        num_cpus=8,
        solver="COPT",
    ) -> None:
        if solver == "COPT":
            self.solver_name = "COPT"
            self.solver_constant = CoptConstant
        elif solver == "GUROBI":
            self.solver_name = "GUROBI"
            self.solver_constant = GurobiConstant
        else:
            raise ValueError("solver must be either COPT or GUROBI")

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

        self.customer_list = customer_list  # List[Customer]
        self.cus_num = len(self.customer_list)
        self.subgraph = {}  # Dict[customer, nx.DiGraph]
        self.columns = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.columns_helpers = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.oracles: Dict[Customer, Optional[ray.actor.ActorHandle, Pricing]] = {}
        # @note: new,
        #   the oracle extension saves extra constraints for primal feasibility
        #   every time it is used, remove this
        # self.bool_covering = bool_covering
        self.bool_edge_lb = self.arg.edgelb
        self.bool_node_lb = self.arg.nodelb
        self.bool_fixed_cost = self.arg.fixed_cost
        self.bool_covering = self.arg.covering
        self.bool_capacity = self.arg.capacity
        self.add_in_upper = self.arg.add_in_upper
        # self.add_distance = self.arg.add_distance
        # self.add_cardinality = self.arg.add_cardinality
        self.dual_index = dict()
        self.variables = dict()  # variables
        self.iter = 0
        self.max_iter = max_iter
        self.red_cost = np.zeros((max_iter, len(customer_list)))
        self.full_network = self.network.copy()
        self.fix_val_constr = {}
        self.init_primal = init_primal
        self.init_dual = init_dual
        self.init_sweeping = init_sweeping
        self.bool_rmp_update_initialized = False
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
        self.skip_customers = set()

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

            self.subgraph[customer] = nx.DiGraph(sku_list=self.full_sku_list)

            # can we do better?
            this_subgraph = self.network.edge_subgraph(self.network.in_edges(customer))
            self.subgraph[customer].add_nodes_from(this_subgraph.nodes(data=True))
            self.subgraph[customer].add_edges_from(this_subgraph.edges(data=True))
            self.subgraph[customer].graph["sku_list"] = cus_sku_list

            self._logger.debug(f"{cus_sku_list}")
            self._logger.debug(f"{sorted(k.__str__() for k in this_subgraph.nodes)}")

        return

    def construct_worker(self):
        cus_per_worker = int(np.ceil(self.cus_num / self.num_workers))
        for customer, n in zip(self.customer_list, range(self.cus_num)):
            if n % cus_per_worker == 0:
                cus_list = self.customer_list[n : min(n + cus_per_worker, self.cus_num)]
                worker = PricingWorker.remote(
                    cus_list,
                    self.arg,
                    self.bool_covering,
                    self.bool_edge_lb,
                    self.bool_node_lb,
                    self.solver_name,
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
        arg = self.arg
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
            oracle = Pricing(
                arg,
                subgraph,
                model_name=model_name,
                # lk: bool_covering应该是False比较好
                # bool_covering=self.bool_covering,
                # bool_capacity=self.arg.capacity,
                # bool_edge_lb=self.arg.lowerbound,
                # bool_node_lb=self.arg.nodelb,
                # bool_fixed_cost=self.arg.node_cost,
                customer=customer,
                solver=self.solver_name,
                logging=CG_PRICING_LOGGING,
            )  # for initial column, set obj = 0
            oracle.modeling(customer)
            # oracle.write(f"{customer}.pricing.lp")

        return oracle

    # todo, remove later.
    # def solve_lp_relaxation(self):
    #     full_lp_relaxation = dnp_model.DNP(self.arg, self.full_network)
    #     full_lp_relaxation.modeling()
    #     # get the LP relaxation
    #     vars = full_lp_relaxation.model.getVars()
    #     binary_vars_index = []
    #     for v in vars:
    #         if v.getType() == self.solver_constant.BINARY:
    #             binary_vars_index.append(v.getIdx())
    #             v.setType(self.solver_constant.CONTINUOUS)
    #     ######################
    #     full_lp_relaxation.model.setParam("Logging", 1)
    #     full_lp_relaxation.solve()
    #     lp_dual = full_lp_relaxation.model.getDuals()
    #     dual_index = full_lp_relaxation.dual_index_for_rmp
    #
    #     return lp_dual, dual_index

    def solve_rmp(self):
        """
        Solve the RMP and get the dual variables to construct the subproblem
        """
        self.rmp_model.setParam("LpMethod", 2)
        self.rmp_model.setParam("Crossover", 0)
        if self.arg.rmp_relaxation:
            variables = self.rmp_model.getVars()
            binary_vars_index = []
            for v in variables:
                if v.getType() == COPT.BINARY:
                    binary_vars_index.append(v.getIdx())
                    v.setType(COPT.CONTINUOUS)
        # self.rmp_model.solve()
        self.solver.solve()
        self.rmp_model.setParam(self.solver_constant.Param.Logging, 0)

    def subproblem(self, customer: Customer, col_ind, dual_vars=None, dual_index=None):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

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
            if self.arg.pricing_relaxation:
                variables = oracle.model.getVars()
                binary_vars_index = []
                for v in variables:
                    if v.getType() == COPT.BINARY:
                        binary_vars_index.append(v.getIdx())
                        v.setType(COPT.CONTINUOUS)
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
        if CG_EXTRA_VERBOSITY:
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

    def init_column(self, customer):
        _vals = {}
        _vals["sku_flow"] = {
            k: 0.0 for k, v in self.oracles[customer].variables["sku_flow"].items()
        }
        unfulfilled_cost = 0.0
        for t in range(self.arg.T):
            unfulfilled_cost += self.init_sku_unfulfill_demand_cost(t, customer)
        _vals["beta"] = unfulfilled_cost
        return _vals

    def init_sku_unfulfill_demand_cost(self, t: int, customer: Customer):
        # lk：check完了没问题 就是把unfulfill_demand_cost取小了
        unfulfill_demand_cost = 0.0

        if customer.has_demand(t):
            for k in customer.demand_sku[t]:
                if customer.unfulfill_sku_unit_cost is not None:
                    unfulfill_sku_unit_cost = customer.unfulfill_sku_unit_cost[(t, k)]
                else:
                    unfulfill_sku_unit_cost = self.arg.unfulfill_sku_unit_cost
                if self.arg.customer_backorder:
                    unfulfill_node_sku_cost = unfulfill_sku_unit_cost * (
                        customer.demand.get((t, k), 0)
                    )*(self.arg.T - t)
                else:
                    unfulfill_node_sku_cost = unfulfill_sku_unit_cost * customer.demand.get(
                        (t, k), 0
                    )

                unfulfill_demand_cost = unfulfill_demand_cost + unfulfill_node_sku_cost

                # self.obj["unfulfill_demand_cost"][(t, k)] = unfulfill_node_sku_cost

        return unfulfill_demand_cost

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
                        worker.construct_pricings.remote(self.subgraph)
                        for worker in self.worker_list
                    ]
                )
            for customer in tqdm(self.customer_list):
                self.oracles[customer] = self.construct_oracle(customer)
                init_col = self.init_column(customer)
                self.columns[customer] = []
                self.columns[customer].append(init_col)

        # use the sweeping method to initialize
        # @note,
        #   is the primal sweeping needed for this formulation ?
        #   do we ever need an initialization
        # cg_init.primal_sweeping_method(self)
        with utils.TimerContext(self.iter, f"initialize_rmp"):
            self.init_rmp()
            # self.rmp_model.write("init_rmp0.lp")
            self.init_rmp_by_cols()
            # self.rmp_model.write("init_rmp1.lp")
        self.rmp_model.setParam(self.solver_constant.Param.Logging, 1)

        self._logger.info("initialization of restricted master finished")
        self._logger.info("solving the first rmp")
        while True:  # may need to add a termination condition
            try:
                bool_early_stop = False
                with utils.TimerContext(self.iter, f"solve_rmp"):
                    self.solve_rmp()
                if self.rmp_model.status != self.solver_constant.OPTIMAL:
                    print(self.rmp_model.status, iter)
                    self.rmp_model.write("rmp{}.lp".format(self.iter))
                # if self.rmp_model.status == self.solver_constant.NUMERICAL:
                #     # print("NUMERICAL",self.rmp_model.status, iter)
                #     self.rmp_model.write("rmp{}.lp".format(self.iter))
                if self.rmp_model.status == self.solver_constant.INFEASIBLE:
                    self._logger.info("initial column of RMP is infeasible")
                    # self.rmp_model.write("1109/rmp.lp")
                    self.rmp_model.computeIIS()
                    self.rmp_model.write("rmp_0102.iis")
                    print("iis written")
                    # vv = self.rmp_oracle.constrs.get("flow_conservation")
                    # if vv is not None:
                    #     self.rmp_model.remove(vv)
                    # for t in range(self.arg.T):
                    #     self.add_new_flow_conservation(t,self.rmp_model)
                    # self.rmp_model.update_objective()
                    # self.solve_rmp()
                if CG_EXTRA_VERBOSITY:
                    self._logger.info("extra verbosity lp")
                    self.rmp_model.write(f"rmp@{self.iter}.lp")
                with utils.TimerContext(self.iter, f"get_duals"):
                    ######################################
                    dual_packs = (
                        self.rmp_oracle.fetch_dual_info() if self.iter >= 1 else None
                    )

                ######################################
                # todo: 额外写个early stopping的功能
                added = False
                # pre-sort dual variables,
                #   this should reduce to 1/|C| update time
                if self.iter >= 1:
                    dual, dual_ws = dual_packs
                    dual_series = pd.Series(
                        {(ee.end, t, ee, k): v for (ee, k, t), v in dual.items()}
                    )

                with utils.TimerContext(self.iter, f"solve_columns"):
                    # modify for parallel
                    if self.init_ray:
                        for worker in tqdm(self.worker_list, ncols=80, leave=False):
                            worker.set_scope.remote(self.skip_customers)
                            worker.model_reset_all.remote()
                            # todo: not sure,
                            worker.update_objective_all.remote(
                                (dual_series[customer, :], dual_ws[customer])
                            )
                            worker.solve_all.remote()
                    else:
                        for customer in tqdm(self.customer_list, ncols=80, leave=False):
                            # lk: 这里的oracle是pricing问题的oracle init_RMP之前的oracle
                            #   emm那么请问这里reset之后右不加新的约束不是没有任何约束了嘛？？？
                            oracle: Pricing = self.oracles[customer]
                            oracle.model.reset()
                            # 把对偶变量加上后update oracle 然后solve 这是第一次求解
                            if CG_PRICING_LOGGING:
                                self._logger.info(f"start update {customer.idx}")
                            with utils.TimerContext(self.iter, f"update pricing"):
                                oracle.update_objective(
                                    customer,
                                    dual_packs=(
                                        dual_series[customer, :],
                                        dual_ws[customer],
                                    )
                                    if self.iter >= 1
                                    else None,
                                )
                            if CG_PRICING_LOGGING:
                                self._logger.info(f"end update {customer.idx}")
                            if self.arg.pricing_relaxation:
                                variables = oracle.model.getVars()
                                binary_vars_index = []
                                for v in variables:
                                    if v.getType() == COPT.BINARY:
                                        binary_vars_index.append(v.getIdx())
                                        v.setType(COPT.CONTINUOUS)
                            oracle.solve()
                            if oracle.model.status == self.solver_constant.INFEASIBLE:
                                self._logger.info("oracle is infeasible")
                                oracle.model.write(
                                    f"{utils.CONF.DEFAULT_TMP_PATH}/oracle{customer}.lp"
                                )
                                oracle.model.computeIIS()
                                oracle.model.write(
                                    f"{utils.CONF.DEFAULT_TMP_PATH}/oracle{customer}.iis"
                                )
                                print("iis written")

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
                        # lk: 是否RMP初始化结束之后就要给出一些column？或者说这个时候column都是0也应该给出一个 🚬
                        new_cols = [
                            oracle.query_columns() for oracle in self.oracles.values()
                        ]
                        v = [oracle.model.objval for oracle in self.oracles.values()]
                        model_status_list = [
                            oracle.model.status for oracle in self.oracles.values()
                        ]

                    for col_ind, customer in enumerate(self.customer_list):
                        self.red_cost[self.iter, col_ind] = v[col_ind]
                        added = (v[col_ind] < -1e-9 or dual_packs is None) or added
                        new_col = new_cols[col_ind]
                        self.columns[customer].append(new_col)

                        model_status = model_status_list[col_ind]
                        if (
                            # self.oracles[customer].model.status
                            # ray.get(self.oracles[customer].get_model_status.remote())
                            model_status
                            == self.solver_constant.INTERRUPTED
                        ):
                            bool_early_stop = True
                            self._logger.info("early terminated")
                            break
                    # modify for parallel

                self.iter += 1

                self._logger.info(
                    f"k: {self.iter:5d} / {self.max_iter:d} f: {self.rmp_model.objval:.6e}, c': {np.min(self.red_cost[self.iter - 1, :]):.4e}",
                )

                if not added or self.iter >= self.max_iter:
                    self.red_cost = self.red_cost[: self.iter, :]
                    break

                if bool_early_stop:
                    self._logger.info("early terminated")
                    break
                with utils.TimerContext(self.iter, f"update_rmp"):
                    self.update_rmp_by_cols()

            except KeyboardInterrupt as _unused_e:
                self._logger.info("early terminated")
                break
        utils.visualize_timers()
        self._logger.info(f"save solutions to {utils.CONF.DEFAULT_SOL_PATH}")
        self.get_solution(utils.CONF.DEFAULT_SOL_PATH)

    def init_rmp_by_cols(self):
        """
        update the RMP with new columns
        """
        # lk：就是说没看懂这个update_rmp_by_cols的东西 和上面的solve_columns的东西有什么关系吗？？？ lambda的体现呢？
        # self.rmp_model.reset()
        if not self.bool_rmp_update_initialized:
            # 这个是放binding约束的
            self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
            # 这个是放sku_flow的值的
            self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
            # 这个是放ws的约束的
            self.ws_cons_idx = {customer: 0 for customer in self.customer_list}
            # lk:cg_binding_constrs要加点名字啊不然都不知道谁是谁
            for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
                edges = self.rmp_oracle.cg_downstream[node]
                for ee in edges:
                    c = ee.end
                    self.delievery_cons_idx[c].append(v)

            for c, v in self.rmp_oracle.cg_binding_constrs_ws.items():
                self.ws_cons_idx[c] = v
                # v.lb = 1.0
                # v.ub = 1.0
                self.solver.setEqualConstr(v, 1.0)

        for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
            edges = self.rmp_oracle.cg_downstream[node]
            for ee in edges:
                c = ee.end
                this_col = self.columns[c][-1]
                self.delievery_cons_coef[c].append(
                    -this_col["sku_flow"].get((t, ee, k), 0)
                )

        for c in self.customer_list:
            _cons_idx = [*self.delievery_cons_idx[c], self.ws_cons_idx[c]]
            _cons_coef = [*self.delievery_cons_coef[c], 1]
            col_idxs = list(zip(_cons_idx, _cons_coef))
            # capacity_cons_name = [*edge_capacity_cons_name, *node_capacity_cons_name]
            try:
                # new_col = cp.Column(
                #     col_idxs,
                # )
                new_col = self.solver.addColumn(_cons_idx, _cons_coef)
                # 这块还要再看看怎么搞
                self.rmp_oracle.variables["column_weights"][
                    c, self.columns[c].__len__() - 1
                ] = self.rmp_model.addVar(
                    obj=self.columns[c][-1]["beta"],
                    name=f"lambda_{c.idx}_{len(self.columns[c])}",
                    lb=0.0,
                    ub=1.0,
                    column=new_col,
                )
            except Exception as e:
                print(f"failed at {c}\n\t{col_idxs}")
                raise e

        vv = self.rmp_oracle.variables.get("cg_temporary")
        if vv is not None:
            self.rmp_model.remove(vv)
            self.rmp_oracle.variables["cg_temporary"] = None
            print(f"removed initial skeleton")

    def update_rmp_by_cols(self):
        """
        update the RMP with new columns
        """
        # lk：就是说没看懂这个update_rmp_by_cols的东西 和上面的solve_columns的东西有什么关系吗？？？ lambda的体现呢？
        # self.rmp_model.reset()
        if not self.bool_rmp_update_initialized:
            # 这个是放binding约束的
            self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
            # 这个是放sku_flow的值的
            self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
            # 这个是放ws的约束的
            self.ws_cons_idx = {customer: 0 for customer in self.customer_list}
            # lk:cg_binding_constrs要加点名字啊不然都不知道谁是谁
            for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
                edges = self.rmp_oracle.cg_downstream[node]
                for ee in edges:
                    c = ee.end
                    self.delievery_cons_idx[c].append(v)

            for c, v in self.rmp_oracle.cg_binding_constrs_ws.items():
                self.ws_cons_idx[c] = v
                # v.lb = 1.0
                # v.ub = 1.0
                self.solver.setEqualConstr(v, 1.0)

        for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
            edges = self.rmp_oracle.cg_downstream[node]
            for ee in edges:
                c = ee.end
                this_col = self.columns[c][-1]
                self.delievery_cons_coef[c].append(
                    -this_col["sku_flow"].get((t, ee, k), 0)
                )

        for c in self.customer_list:
            _cons_idx = [*self.delievery_cons_idx[c], self.ws_cons_idx[c]]
            _cons_coef = [*self.delievery_cons_coef[c], 1]
            col_idxs = list(zip(_cons_idx, _cons_coef))
            # capacity_cons_name = [*edge_capacity_cons_name, *node_capacity_cons_name]
            try:
                # new_col = cp.Column(
                #     col_idxs,
                # )
                new_col = self.solver.addColumn(_cons_idx, _cons_coef)
                # 这块还要再看看怎么搞
                self.rmp_oracle.variables["column_weights"][
                    c, self.columns[c].__len__() - 1
                ] = self.rmp_model.addVar(
                    obj=self.columns[c][-1]["beta"],
                    name=f"lambda_{c.idx}_{len(self.columns[c])}",
                    lb=0.0,
                    ub=1.0,
                    column=new_col,
                )
            except Exception as e:
                print(f"failed at {c}\n\t{col_idxs}")
                raise e
        # vv = self.rmp_oracle.variables.get("cg_temporary")
        # if vv is not None:
        #     self.rmp_model.remove(vv)
        #     self.rmp_oracle.variables["cg_temporary"] = None
        #     print(f"removed initial skeleton")

    def init_rmp(self):
        """
        Initialize the RMP with initial columns
        """
        self.rmp_oracle = DNPSlim(
            arg=self.arg,
            network=self.network,
            full_sku_list=self.full_sku_list,
            customer_list=self.customer_list,
            # bool_covering=self.bool_covering,
            # bool_edge_lb=self.bool_edge_lb,
            # bool_node_lb=self.bool_node_lb,
            # bool_fixed_cost=self.arg.node_cost,
            # bool_capacity=self.arg.capacity,
            used_edge_capacity=None,
            used_warehouse_capacity=None,
            used_plant_capacity=None,
            solver=self.solver_name,
        )
        self.rmp_oracle.modeling()
        self.rmp_model = self.rmp_oracle.model
        self.rmp_oracle.create_cg_bindings()
        self.solver = self.rmp_oracle.solver

    # def add_new_flow_conservation(self, t: int,solver):
    #     for node in self._iterate_no_c_nodes():
    #         in_edges = utils.get_in_edges(self.network, node)
    #         out_edges = utils.get_out_edges(self.network, node)
    #         out_edges_master = [e for e in out_edges if e.end.type != const.CUSTOMER]
    #
    #         sku_list = node.get_node_sku_list(t, self.full_sku_list)
    #         for k in sku_list:
    #             constr_name = f"flow_conservation_{t}_{node.idx}_{k.idx}"
    #
    #             if node.type == const.PLANT:
    #                 # constr = self.model.addConstr(
    #                 constr = self.solver.addConstr(
    #                     self.variables["sku_production"][t, node, k]
    #                     - self.variables["sku_flow"].sum(t, out_edges_master, k)
    #                     == 0,
    #                     name=constr_name,
    #                 )
    #
    #             elif node.type == const.WAREHOUSE:
    #                 fulfilled_demand = 0
    #                 if node.has_demand(t, k):
    #                     fulfilled_demand = (
    #                             node.demand[t, k]
    #                             - self.variables["sku_demand_slack"][t, node, k]
    #                     )
    #
    #                 last_period_inventory = 0.0
    #
    #                 if t == 0:
    #                     if self.bool_covering:
    #                         if node.initial_inventory is not None:
    #                             # if self.open_relationship:
    #                             # self.model.addConstr(
    #                             self.solver.addConstr(
    #                                 self.variables["open"][self.T - 1, node] == 1
    #                             )
    #                             last_period_inventory = (
    #                                 node.initial_inventory[k]
    #                                 if k in node.initial_inventory
    #                                 else 0.0
    #                             )
    #                         else:
    #                             last_period_inventory = 0.0
    #                     else:
    #                         last_period_inventory = 0.0
    #                 else:
    #                     last_period_inventory = self.variables["sku_inventory"][
    #                         t - 1, node, k
    #                     ]
    #                 # last_period_inventory *= self.cus_ratio
    #                 # if self.arg.backorder:
    #                 #     constr = self.solver.addConstr(
    #                 #         self.variables["sku_flow"].sum(t, in_edges, k)
    #                 #         + last_period_inventory
    #                 #         + self.variables["sku_backorder"][t,node, k]
    #                 #         - fulfilled_demand
    #                 #         - self.variables["sku_flow"].sum(t, out_edges_master, k)
    #                 #         - self.variables["sku_delivery"][t, node, k]
    #                 #         - self.variables["sku_backorder"].get((t - 1,node, k), 0)
    #                 #         == self.variables["sku_inventory"][t, node, k],
    #                 #         name=constr_name,
    #                 #     )
    #                 # else:
    #                 constr = self.solver.addConstr(
    #                     self.variables["sku_flow"].sum(t, in_edges, k)
    #                     + last_period_inventory
    #                     + self.variables["sku_backorder"][t,node, k]
    #                     - fulfilled_demand
    #                     - self.variables["sku_flow"].sum(t, out_edges_master, k)
    #                     - self.variables["sku_delivery"][t, node, k]
    #                     == self.variables["sku_inventory"][t, node, k],
    #                     name=constr_name,
    #                 )
    #             elif node.type == const.CUSTOMER:
    #                 raise ValueError("flow in RMP do not contain customers")
    #
    #             self.constrs["flow_conservation"][(t, node, k)] = constr
    #
    #     return
    # def update_objective(self,solver):
    #     ud = sum(self.cal_sku_unfulfill_demand_cost(t) for t in range(self.arg.T))
    #     solver.setObjective(self.rmp_model.obj + ud, sense=self.solver_constant.MINIMIZE)

    # def cal_sku_unfulfill_demand_cost(self, t: int):
    #     unfulfill_demand_cost = 0.0
    #
    #     for node in self._iterate_no_c_nodes():
    #         if node.type == const.WAREHOUSE:
    #             if node.has_demand(t):
    #                 for k in node.get_node_sku_list(t, self.full_sku_list):
    #
    #                     if node.unfulfill_sku_unit_cost is not None:
    #                         unfulfill_sku_unit_cost = node.unfulfill_sku_unit_cost[
    #                             (t, k)
    #                         ]
    #                     else:
    #                         unfulfill_sku_unit_cost = 50000
    #
    #                     unfulfill_node_sku_cost = (
    #                             unfulfill_sku_unit_cost
    #                             * self.variables["sku_backorder"][(t, node, k)]
    #                     )
    #
    #                     unfulfill_demand_cost = (
    #                             unfulfill_demand_cost + unfulfill_node_sku_cost
    #                     )
    #
    #     return unfulfill_demand_cost

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        self.variables["column_weights"] = self.rmp_oracle.variables["column_weights"]

        cus_col_value = pd.DataFrame(
            index=range(self.iter + 1), columns=[c.idx for c in self.customer_list]
        )
        cus_col_weights = pd.DataFrame(
            index=range(self.iter + 1), columns=[c.idx for c in self.customer_list]
        )
        reduced_cost = pd.DataFrame(
            self.red_cost, columns=[c.idx for c in self.customer_list]
        )

        for customer in self.customer_list:
            # for number in range(self.num_cols):
            for number in range(self.iter - 1):
                number = number + 1
                if self.columns[customer][number] != {}:
                    cus_col_value.loc[number, customer.idx] = self.columns[customer][
                        number
                    ]["beta"]
                    cus_col_weights.loc[number, customer.idx] = self.solver.getVarValue(
                        self.variables["column_weights"][customer, number]
                    )
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
