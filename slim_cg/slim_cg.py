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
from slim_cg.slim_pricing import Pricing, PricingWorker

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
        self.obj = dict()
        self.columns_helpers = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.oracles: Dict[Customer, Optional[ray.actor.ActorHandle, Pricing]] = {}
        # @note: new,
        #   the oracle extension saves extra constraints for primal feasibility
        #   every time it is used, remove this
        # self.bool_covering = bool_covering
        self.bool_edge_lb = self.arg.edge_lb
        self.bool_node_lb = self.arg.node_lb
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
            self._logger.info("initializing ray")
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
        arg.full_sku_list = self.full_sku_list
        model_name = customer.idx + "_oracle"

        if self.init_ray:
            oracle = self.worker_cus_dict[customer]

        else:
            oracle = Pricing(
                arg,
                subgraph,
                model_name=model_name,
                customer=customer,
                solver=self.solver_name,
            )  # for initial column, set obj = 0
            oracle.modeling(customer)

        return oracle

    def solve_rmp(self):
        """
        Solve the RMP and get the dual variables to construct the subproblem
        """
        # self.rmp_model.setParam("LpMethod", 2)
        self.rmp_model.setParam("Crossover", 0)
        self.solver.solve()
        self.rmp_model.setParam(self.solver_constant.Param.Logging, 0)

    def subproblem(self, customer: Customer, col_ind, dual_vars=None, dual_index=None):
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
        _vals["unfulfilled_demand_cost"] = {t: 0.0 for t in range(self.arg.T)}
        for t in range(self.arg.T):
            if self.arg.backorder:
                # self.partial_obj[t] += self.init_sku_backlogged_demand_cost(t, customer)
                unfulfilled_cost += self.init_sku_backlogged_demand_cost(t, customer)
                _vals["unfulfilled_demand_cost"][t] += self.init_sku_backlogged_demand_cost(t, customer)
            else:
                unfulfilled_cost += self.init_sku_unfulfill_demand_cost(t, customer)
                _vals["unfulfilled_demand_cost"][t] +=  self.init_sku_unfulfill_demand_cost(t, customer)
        _vals["beta"] = unfulfilled_cost
        _vals["transportation_cost"] = 0
        return _vals

    def init_sku_unfulfill_demand_cost(self, t: int, customer: Customer):
        unfulfill_demand_cost = 0.0
        if customer.has_demand(t):
            for k in customer.demand_sku[t]:
                # if customer.unfulfill_sku_unit_cost is not None:
                #     unfulfill_sku_unit_cost = customer.unfulfill_sku_unit_cost[(t, k)]
                # else:
                unfulfill_sku_unit_cost = self.arg.unfulfill_sku_unit_cost

                unfulfill_node_sku_cost = unfulfill_sku_unit_cost * customer.demand.get(
                    (t, k), 0
                )
                unfulfill_demand_cost = unfulfill_demand_cost + unfulfill_node_sku_cost

        return unfulfill_demand_cost

    def init_sku_backlogged_demand_cost(self, t: int, customer: Customer):
        backlogged_demand_cost = 0.0
        for v in range(t+1):
            for k in self.full_sku_list:
                backlogged_demand_cost += self.arg.unfulfill_sku_unit_cost * customer.demand.get(
                    (v, k), 0
                )
        return backlogged_demand_cost

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
                        worker.construct_Pricings.remote(self.subgraph)
                        for worker in self.worker_list
                    ]
                )
                all_var_keys = ray.get(
                    [
                        worker.get_all_var_keys.remote("sku_flow")
                        for worker in self.worker_list
                    ]
                )

                var_keys = []
                for var_key in all_var_keys:
                    var_keys.extend(var_key)
            else:
                var_keys = []
                for customer in tqdm(self.customer_list):
                    self.oracles[customer] = self.construct_oracle(customer)
                    var_keys.extend(self.oracles[customer].get_var_keys("sku_flow"))

                    # init_col = self.init_column(customer)
                    # self.columns[customer] = []
                    # self.columns[customer].append(init_col)

            for customer, i in zip(self.customer_list, range(self.cus_num)):
                init_col = self.init_column(customer, var_keys[i])
                self.columns[customer] = []
                self.columns[customer].append(init_col)

        with utils.TimerContext(self.iter, f"initialize_rmp"):
            self.init_rmp()
            self.init_rmp_by_cols()
        self.rmp_model.setParam(self.solver_constant.Param.Logging, 1)
        self._logger.info("initialization of restricted master finished")
        self._logger.info("solving the first rmp")
        while True:
            try:
                bool_early_stop = False
                with utils.TimerContext(self.iter, f"solve_rmp"):
                    self.solve_rmp()
                self._logger.info(f"rmp solving finished: {self.rmp_model.status}@{iter}")
                # if self.rmp_model.status != self.solver_constant.OPTIMAL:
                #     print(self.rmp_model.status, iter)
                if self.rmp_model.status == self.solver_constant.INFEASIBLE:
                    self._logger.info("initial column of RMP is infeasible")
                    self.rmp_model.computeIIS()
                    self.rmp_model.write(f"rmp@{self.iter}.iis")
                if CG_EXTRA_VERBOSITY:
                    self._logger.info("extra verbosity lp")
                    self.rmp_model.write(f"rmp@{self.iter}.lp")
                with utils.TimerContext(self.iter, f"get_duals"):
                    ######################################
                    dual_packs = (
                        self.rmp_oracle.fetch_dual_info() if self.iter >= 1 else None
                    )
                self._logger.info("rmp dual fetch finished")

                # TODO: early stopping
                added = False
                # pre-sort dual variables,
                #   this should reduce to 1/|C| update time
                if self.iter >= 1:
                    dual, dual_ws = dual_packs
                    dual_series = pd.Series(
                        {(ee.end, ee, k, t): v for (ee, k, t), v in dual.items()}
                    )
                    dual_exists_customers = dual_series.index.get_level_values(0)
                else:
                    dual_series = None
                    dual_ws = None
                    dual_exists_customers = None
                # if self.iter >= 1 and customer in dual_exists_customers:
                #     dual_pack_this = (
                #         dual_series[customer, :].to_dict(),
                #         dual_ws[customer],
                #     )
                # else:
                #     dual_pack_this = None

                with utils.TimerContext(self.iter, f"solve_columns"):
                    # modify for parallel
                    if self.init_ray:
                        for worker in tqdm(self.worker_list, ncols=80, leave=False):
                            worker.set_scope.remote(self.skip_customers)
                            worker.model_reset_all.remote()
                            with utils.TimerContext(self.iter, f"update pricing"):
                                # worker.update_objective_all.remote(dual_pack_this)
                                worker.update_objective_all_new.remote(
                                    customer,
                                    self.iter,
                                    dual_series,
                                    dual_ws,
                                    dual_exists_customers,
                                )

                            if self.arg.pricing_relaxation:
                                worker.set_all_relaxation.remote()
                            worker.solve_all.remote()
                    else:
                        for col_ind, customer in tqdm(
                            enumerate(self.customer_list), ncols=80, leave=False
                        ):
                            oracle: Pricing = self.oracles[customer]
                            oracle.model.reset()
                            if self.iter >= 1 and customer in dual_exists_customers:
                                dual_pack_this = (
                                    dual_series[customer, :].to_dict(),
                                    dual_ws[customer],
                                )
                            else:
                                dual_pack_this = None
                            with utils.TimerContext(self.iter, f"update pricing"):
                                oracle.update_objective(
                                    customer, dual_packs=dual_pack_this
                                )
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
                                oracle.model.computeIIS()
                                oracle.model.write("oracle{}.iis".format(customer))
                                print("iis written")
                    self._logger.info("column solving finished")
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
                    self._logger.info("column generating finished")
                    for col_ind, customer in enumerate(self.customer_list):
                        self.red_cost[self.iter, col_ind] = v[col_ind]
                        added = (v[col_ind] < -1e-9 or dual_packs is None) or added
                        new_col = new_cols[col_ind]
                        self.columns[customer].append(new_col)

                        model_status = model_status_list[col_ind]
                        if model_status == self.solver_constant.INTERRUPTED:
                            bool_early_stop = True
                            self._logger.info("early terminated")
                            break
                        if model_status == self.solver_constant.INFEASIBLE:
                            self._logger.info("oracle is infeasible")
                    # modify for parallel

                self.iter += 1

                self._logger.info(
                    f"k: {self.iter:5d} / {self.max_iter:d} f: {self.rmp_model.objval:.6e}, c': {np.min(self.red_cost[self.iter - 1, :]):.4e}",
                )
                lp_objective = self.rmp_model.objval

                # if self.arg.check_rmp_mip:
                #     if int(self.iter) % self.arg.rmp_mip_iter == 0:
                if self.arg.check_rmp_mip and not self.rmp_oracle.bool_is_lp:
                    if (int(self.iter) % self.arg.rmp_mip_iter == 0) or (
                        self.iter >= self.max_iter
                    ):
                        model = self.rmp_model
                        self.rmp_oracle.switch_to_milp()
                        print("-----Solve MIP_RMP-----")
                        model.solve()
                        print(
                            self.iter,
                            "MIP_RMP",
                            model.getObjective().getValue(),
                            "GAP",
                            model.getObjective().getValue() - lp_objective,
                        )
                        ### Reset
                        self.rmp_oracle.switch_to_lp()
                        print("Reset Over")
                if self.arg.check_cost_cg:
                    holding_cost_rmp = 0.0
                    transportation_cost_rmp = 0.0
                    self.rmp_oracle.obj['holding_cost'] = {}
                    for t in self.rmp_oracle.obj['holding_cost'].keys():
                        holding_cost_rmp += self.rmp_oracle.obj['holding_cost'][t].getExpr().getValue()
                    for t in self.rmp_oracle.obj['transportation_cost'].keys():
                        if type(self.rmp_oracle.obj['transportation_cost'][t]) is not float:
                            transportation_cost_rmp += self.rmp_oracle.obj['transportation_cost'][t].getExpr().getValue()
                    print("tr_rmp",transportation_cost_rmp)
                    transportation_cost_from_customer = 0.0
                    unfulfilled_cost_from_customer = {t:0 for t in range(self.arg.T)}
                    variables = self.rmp_model.getVars()
                    for v in variables:
                        if v.getName().startswith('lambda'):
                            # print(self.iter,v.getName())
                            # print(v.getName().split('_')[1])
                            for customer in self.columns.keys():
                                if v.getName().split('_')[1] == str(customer):
                                    print(v.getName())
                                    print("lambda optimal value", v.x)
                                    col_num = int(v.getName().split('_')[2])
                                    print(self.columns[customer][col_num]['unfulfilled_demand_cost'])
                                    print(self.columns[customer][col_num]['transportation_cost'])
                                    # print("transportation_cost", self.columns[customer][0]['transportation_cost'])
                                    # print("unfulfilled_demand_cost", self.columns[customer][0]['unfulfilled_demand_cost'])
                                    transportation_cost_from_customer += v.x * self.columns[customer][col_num][
                                        'transportation_cost']
                                    for t in range(self.arg.T):
                                        unfulfilled_cost_from_customer[t] += v.x * self.columns[customer][col_num]['unfulfilled_demand_cost'][t]
                    print("tr_pricing", transportation_cost_from_customer)
                    print("transportation_cost", transportation_cost_rmp + transportation_cost_from_customer)
                    print("holding_cost", holding_cost_rmp)
                    for t in range(self.arg.T):
                        print("unfulfilled_demand_cost", t, unfulfilled_cost_from_customer[t])
                    print("unfulfilled_demand_cost", unfulfilled_cost_from_customer)
                if not added or self.iter >= self.max_iter:
                    self.red_cost = self.red_cost[: self.iter, :]
                    break

                if bool_early_stop:
                    self._logger.info("early terminated")
                    break
                self._logger.info("rmp updating started")
                with utils.TimerContext(self.iter, f"update_rmp"):
                    self.update_rmp_by_cols()
                self._logger.info("rmp updating finished")

            except KeyboardInterrupt as _unused_e:
                self._logger.info("early terminated")
                break
        utils.visualize_timers()
        self._logger.info(f"save solutions to {utils.CONF.DEFAULT_SOL_PATH}")
        # todo
        # self.get_solution(utils.CONF.DEFAULT_SOL_PATH)

    def init_rmp_by_cols(self):
        """
        update the RMP with new columns
        """
        if not self.bool_rmp_update_initialized:
            # sku_flow
            self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
            self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
            # lambda
            self.ws_cons_idx = {customer: 0 for customer in self.customer_list}
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
                new_col = self.solver.addColumn(_cons_idx, _cons_coef)
                self.rmp_oracle.variables["column_weights"][
                    c, self.columns[c].__len__() - 1
                ] = self.rmp_model.addVar(
                    obj=self.columns[c][-1]["beta"],
                    name=f"lambda_{c.idx}_{len(self.columns[c])}",
                    lb=0.0,
                    ub=1.0,
                    vtype=self.solver_constant.CONTINUOUS,
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
        if not self.bool_rmp_update_initialized:
            self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
            self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
            self.ws_cons_idx = {customer: 0 for customer in self.customer_list}
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
            try:
                new_col = self.solver.addColumn(_cons_idx, _cons_coef)
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

    def init_rmp(self):
        """
        Initialize the RMP with initial columns
        """
        self.rmp_oracle = DNPSlim(
            arg=self.arg,
            network=self.network,
            full_sku_list=self.full_sku_list,
            customer_list=self.customer_list,
            used_edge_capacity=None,
            used_warehouse_capacity=None,
            used_plant_capacity=None,
            solver=self.solver_name,
        )
        self.rmp_oracle.modeling()
        self.rmp_model = self.rmp_oracle.model
        self.rmp_oracle.create_cg_bindings()
        self.solver = self.rmp_oracle.solver

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
