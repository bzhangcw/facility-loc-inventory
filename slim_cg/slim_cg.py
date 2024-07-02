import argparse
import json
import logging
import os
from typing import *
import ray
from coptpy import COPT
from tqdm import tqdm
import slim_cg.slim_mip_heur as slp
import slim_cg.slim_rmp_alg as sla
import slim_cg.slim_col_alg as slc
from config.network import *
from entity import SKU, Customer
from slim_cg.slim_checker import check_cost_cg
from slim_cg.slim_pricing import CG_SUBP_LOGGING, Pricing, PricingWorker
from slim_cg.slim_rmp_model import DNPSlim
from solver_wrapper.CoptConstant import CoptConstant
from solver_wrapper.GurobiConstant import GurobiConstant

import utils

CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 0))


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
        self.check_number = self.arg.check_number
        self.del_col_alg = self.arg.del_col_alg
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
        self.columns_status = (
            {}
        )  # Dict[customer, List[0/1]], 1 for active, 0 for inactive
        self.columns_to_del = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.obj = dict()
        self.columns_helpers = {}  # Dict[customer, List[tuple(x, y, p)]]
        self.oracles: Dict[Customer, Optional[ray.actor.ActorHandle, Pricing]] = {}
        # @note: new,
        #   the oracle extension saves extra constraints for primal feasibility
        #   every time it is used, remove this
        self.vars_basis, self.cons_basis = None, None
        self.bool_edge_lb = self.arg.edge_lb
        self.bool_node_lb = self.arg.node_lb
        self.bool_fixed_cost = self.arg.if_fixed_cost
        self.bool_covering = self.arg.covering
        self.bool_capacity = self.arg.capacity
        self.add_in_upper = self.arg.add_in_upper
        self.dual_index = dict()
        self.variables = dict()  # variables
        self.iter = 0
        self.rmp_objval = 1e20
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
        self.column_weights_rmp = {}

        self.col_weight_history = {cus.idx: {} for cus in self.customer_list}

        # parameters for deleting columns
        self.if_del_col = self.arg.if_del_col
        self.del_col_freq = self.arg.del_col_freq
        self.del_col_stra = self.arg.del_col_stra
        self.column_pool_len = self.arg.column_pool_len
        self._logger.info(
            f"if_del_col: {self.if_del_col}, del_col_freq: {self.del_col_freq}, del_col_stra: {self.del_col_stra}"
        )

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

        sla.solve(self)

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

    # def init_column(self, customer):
    #     _vals = {}
    #     _vals["sku_flow"] = {
    #         k: 0.0 for k, v in self.oracles[customer].variables["sku_flow"].items()
    #     }
    def init_column(self, customer, sku_flow_keys, select_edge_keys, sku_select_edge_keys, open_keys):
        _vals = {}
        _vals["sku_flow"] = {k: 0.0 for k in sku_flow_keys}
        _vals["select_edge"] = {k: 0.0 for k in select_edge_keys}
        _vals["sku_select_edge"] = {k: 0.0 for k in sku_select_edge_keys}
        _vals["open"] = {k: 0.0 for k in open_keys}
        unfulfilled_cost = 0.0
        _vals["unfulfilled_demand_cost"] = {t: 0.0 for t in range(self.arg.T)}
        for t in range(self.arg.T):
            if self.arg.backorder:
                # self.partial_obj[t] += self.init_sku_backlogged_demand_cost(t, customer)
                unfulfilled_cost += self.init_sku_backlogged_demand_cost(t, customer)
                _vals["unfulfilled_demand_cost"][
                    t
                ] += self.init_sku_backlogged_demand_cost(t, customer)
            else:
                unfulfilled_cost += self.init_sku_unfulfill_demand_cost(t, customer)
                _vals["unfulfilled_demand_cost"][
                    t
                ] += self.init_sku_unfulfill_demand_cost(t, customer)
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
        for v in range(t + 1):
            for k in self.full_sku_list:
                backlogged_demand_cost += (
                    self.arg.unfulfill_sku_unit_cost * customer.demand.get((v, k), 0)
                )
        return backlogged_demand_cost

    # 初始化一个字典来记录每个customer和number在前n次iter中的值
    def filter_customers(self):
        # 初始化一个字典来记录每个customer和number在前n次iter中的值
        result = {}
        # 遍历前n次iter
        for iter in range(self.iter-self.arg.check_number,self.iter):
            for customer, numbers in self.column_weights_rmp.get(iter, {}).items():
                for number, value in numbers.items():
                    if (customer, number) not in result:
                        result[(customer, number)] = []
                    result[(customer, number)].append(value)

        # 筛选出所有前n次iter值都为0的customer和number
        filtered_result = [(customer, number) for (customer, number), values in result.items() if all(v < 1e-6 for v in values)]

        return filtered_result

    def update_col_status(self):
        if self.del_col_stra == 1:
            self.column_weights_rmp = {}
            # delete columns getting a 0 weight in any iteration
            for customer in self.customer_list:
                # for number in range(self.iter - 1):
                self.columns_to_del[customer] = []
                for number in range(self.iter):
                    if self.columns_status[customer][number] == 1:
                        lambda_c_n = self.solver.getVarValue(
                            self.rmp_oracle.variables["column_weights"][
                                customer, number
                            ]
                        )
                        if lambda_c_n < 1e-6:
                            self.columns_status[customer][number] = 0
                            self.columns_to_del[customer].append(number)
        else:
            raise NotImplementedError
        

    def update_col_status_2(self):
        if self.del_col_stra == 1:
            for customer in self.customer_list:
                # for number in range(self.iter - 1):
                self.columns_to_del[customer] = []
            filtered_result = self.filter_customers()
            print(filtered_result)
            for customer,number in filtered_result:
                self.columns_status[customer][number] = 0
                self.columns_to_del[customer].append(number)
        else:
            raise NotImplementedError
       
    def update_col_status_3(self):
        if self.del_col_stra == 1:
            for customer in self.customer_list:
                # for number in range(self.iter - 1):
                self.columns_to_del[customer] = []
            mean = self.red_cost[self.iter-1,:].mean()
            for customer in self.customer_list:
                for number in range(self.iter):
                    if self.columns[customer][number]['reduced_cost'] > mean:
                        self.columns_status[customer][number] = 0
                        self.columns_to_del[customer].append(number)
        else:
            raise NotImplementedError

    def update_col_status_4(self):
        if self.del_col_stra == 1:
            for customer in self.customer_list:
                # for number in range(self.iter - 1):
                self.columns_to_del[customer] = []
            # 收集所有 (customer, number, reduced_cost) 组合
            all_combinations = []
            for customer, numbers in self.columns.items():
                for number, values in numbers.items():
                    all_combinations.append((customer, number, values['reduced_cost']))
            # 按 reduced_cost 排序
            sorted_combinations = sorted(all_combinations, key=lambda x: x[2])
            filtered_combinations = sorted_combinations[self.arg.column_pool_len:]
            filtered_result = [(customer, number) for customer, number, reduced_cost in filtered_combinations]
            for customer, number in filtered_result:
                self.columns_status[customer][number] = 0
                self.columns_to_del[customer].append(number)
        else:
            raise NotImplementedError
        
    def run(self):
        """
        The main loop of column generation algorithm
        """
        self.get_subgraph()

        # construct oracles
        self._logger.info("initialization complete, start generating columns...")
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
                    var_keys.append(self.oracles[customer].get_var_keys("sku_flow"))

                select_keys = []
                for customer in tqdm(self.customer_list):
                    self.oracles[customer] = self.construct_oracle(customer)
                    select_keys.append(self.oracles[customer].get_var_keys("select_edge"))

                sku_select_keys = []
                for customer in tqdm(self.customer_list):
                    self.oracles[customer] = self.construct_oracle(customer)
                    sku_select_keys.append(self.oracles[customer].get_var_keys("sku_select_edge"))

                open_keys = []
                for customer in tqdm(self.customer_list):
                    self.oracles[customer] = self.construct_oracle(customer)
                    open_keys.append(self.oracles[customer].get_var_keys("open"))

            for customer, i in zip(self.customer_list, range(self.cus_num)):
                init_col = self.init_column(customer, var_keys[i],select_keys[i],sku_select_keys[i],open_keys[i])
                self.columns[customer] = []
                init_col['reduced_cost'] = init_col['beta']
                self.columns[customer].append(init_col)
                self.columns_status[customer] = [1]

        with utils.TimerContext(self.iter, f"initialize_rmp"):
            self.init_rmp()

        _obj_last_iterate = 1e20
        while True:
            try:
                bool_early_stop = False

                with utils.TimerContext(self.iter, f"update_rmp"):
                    # this is the block to update rmp after
                    #   pricing problems in the end of each iterate.
                    if self.del_col_alg == 4 and len(self.columns)>=self.arg.column_pool_len:
                        self.update_col_status_4()
                    elif (
                        self.if_del_col
                        and self.iter % self.del_col_freq == 0
                        and self.iter >= 1
                    ):
                        # update column status if deleting is used
                        if self.del_col_alg == 1:
                            self.update_col_status()
                        elif self.del_col_alg == 2:
                            self.update_col_status_2()
                        elif self.del_col_alg == 3:
                            self.update_col_status_3()

                        # self.update_col_status()
                    # update the rmp by columns
                    self.update_rmp_by_cols()

                with utils.TimerContext(self.iter, f"generate-extra-cols"):
                    # use this functional block to generate extra cols
                    # !!! this must be placed after `update_rmp`
                    # beyond those from pricing
                    # update the rmp by columns
                    # !!!`self.update_rmp_by_cols()` is contained,
                    # do not invoke again
                    # slc.add_extra_columns(self)
                    pass

                with utils.TimerContext(
                    self.iter, f"solve_rmp_{sla.RMPAlg(sla.CG_RMP_METHOD).name.lower()}"
                ):
                    if CG_EXTRA_DEBUGGING:
                        self.rmp_model.write(
                            f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.mps"
                        )
                    self.solve_rmp()
                    # add recording for weights in the rmp
                    if self.del_col_alg == 2:
                        self.column_weights_rmp[self.iter] = {}
                        for customer in self.customer_list:
                            self.column_weights_rmp[self.iter][customer] = {}
                            for number in range(self.iter):
                                if self.columns_status[customer][number] == 1:
                                    self.column_weights_rmp[self.iter][customer][number] = self.solver.getVarValue(
                                        self.rmp_oracle.variables["column_weights"][
                                            customer, number
                                        ]
                                    )
                    self._logger.info(
                        f"rmp solving finished: {self.rmp_model.status} @ iteration:{self.iter}"
                    )

                eps_fixed_point = abs(_obj_last_iterate - self.rmp_objval)

                if self.rmp_status == self.solver_constant.INFEASIBLE:
                    self._logger.info("RMP is infeasible")
                    self.rmp_model.write(
                        f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.lp"
                    )
                    self.rmp_model.computeIIS()
                    self.rmp_model.write(
                        f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.iis"
                    )
                if CG_EXTRA_VERBOSITY:
                    self._logger.info("extra verbosity lp")
                    self.rmp_model.write(
                        f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.lp"
                    )
                with utils.TimerContext(self.iter, f"get_duals"):
                    ######################################
                    dual_packs = sla.fetch_dual_info(
                        self
                    )  # if self.iter >= 1 else None

                improved = (
                    eps_fixed_point / (abs(self.rmp_objval) + 1e-3)
                ) > self.arg.terminate_condition
                added = False
                # pre-sort dual variables,
                #   this should reduce to 1/|C| update time
                if self.iter >= 1:
                    (
                        node_vals,
                        dual_ws,
                        dual_exists_customers,
                    ) = dual_packs

                    # broadcast node dual values
                    #   and compute edge dual on the remotes
                    with utils.TimerContext(self.iter, f"put node vals storage"):
                        broadcast_nodes = ray.put(node_vals)

                else:
                    dual_ws = None
                    dual_exists_customers = None
                    broadcast_nodes = None
                    broadcast_mat = None
                    broadcast_cols = None
                    broadcast_keys = None

                if self.init_ray and self.iter == 1:
                    # put broadcasting only once
                    with utils.TimerContext(self.iter, f"put obj storage"):
                        broadcast_mat = ray.put(self.rmp_oracle.broadcast_matrix)
                        broadcast_cols = ray.put(self.rmp_oracle.dual_cols)
                        broadcast_keys = ray.put(self.rmp_oracle.dual_keys)

                with utils.TimerContext(self.iter, f"solve_columns"):
                    # modify for parallel
                    if self.init_ray:
                        for worker_id, worker in tqdm(
                            enumerate(self.worker_list), ncols=80, leave=False
                        ):
                            worker.set_scope.remote(self.skip_customers)
                            worker.model_reset_all.remote()
                            with utils.TimerContext(
                                self.iter, f"update pricing", logging=worker_id < 1
                            ):
                                # worker.update_objective_all.remote(dual_pack_this)
                                worker.update_objective_all_new.remote(
                                    self.iter,
                                    broadcast_mat,
                                    broadcast_nodes,
                                    broadcast_cols,
                                    broadcast_keys,
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
                                    self.rmp_oracle.dual_keys[customer],
                                    self.rmp_oracle.dual_vals[customer],
                                    dual_ws[customer],
                                )
                            else:
                                dual_pack_this = None
                            with utils.TimerContext(
                                self.iter, f"update pricing", logging=col_ind < 1
                            ):
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
                            if CG_SUBP_LOGGING:
                                oracle.model.write(
                                    f"subp@{self.iter}@{customer.idx}.mps"
                                )
                            oracle.solve()
                            if oracle.model.status == self.solver_constant.INFEASIBLE:
                                self._logger.info("oracle is infeasible")
                                oracle.model.computeIIS()
                                oracle.model.write("oracle{}.iis".format(customer))
                                self._logger.info("iis written")

                with utils.TimerContext(self.iter, f"generating_columns"):
                    if self.init_ray:
                        # new_cols = [
                        #     cc
                        #     for worker in self.worker_list
                        #     for cc in ray.get(worker.query_all_columns.remote())
                        # ]
                        # correct version, about 2x faster
                        all_new_cols = ray.get(
                            [
                                worker.query_all_columns.remote()
                                for worker in self.worker_list
                            ]
                        )
                        new_cols = [col for new_col in all_new_cols for col in new_col]
                    else:
                        new_cols = [
                            oracle.query_columns() for oracle in self.oracles.values()
                        ]
                        # v = [oracle.model.objval for oracle in self.oracles.values()]
                        # model_status_list = [
                        #     oracle.model.status for oracle in self.oracles.values()
                        # ]
                    self._logger.info("column generating finished")
                    for col_ind, customer in enumerate(self.customer_list):
                        _redcost = new_cols[col_ind]["objval"]
                        _status = new_cols[col_ind]["status"]
                        self.red_cost[self.iter, col_ind] = _redcost
                        added = (
                            (_redcost / (new_cols[col_ind]["beta"] + 1e-1) < -1e-2)
                            or dual_packs is None
                        ) or added
                        if added:
                            new_col = new_cols[col_ind]
                            new_col['reduced_cost'] = _redcost 
                            self.columns[customer].append(new_col)
                            self.columns_status[customer].append(1)

                        if _status == self.solver_constant.INTERRUPTED:
                            bool_early_stop = True
                            self._logger.info("early terminated")
                            break
                        if _status == self.solver_constant.INFEASIBLE:
                            self._logger.info("oracle is infeasible")
                    # modify for parallel

                self.iter += 1
                _this_log_line = (
                    f"k: {self.iter:5d} / {self.max_iter:d} "
                    f"f: {self.rmp_objval:.6e}, "
                    f"eps_df: {eps_fixed_point:.2e}/{100*min(10.0, eps_fixed_point/(self.rmp_objval+1e-3)):.1f}%, "
                    f"c': {np.min(self.red_cost[self.iter - 1, :]):.4e},"
                )

                self._logger.info(_this_log_line)
                lp_objective = self.rmp_objval

                bool_terminate = self.iter > 1 and (
                    (not improved) or (not added) or self.iter >= self.max_iter
                )

                # if self.arg.cg_mip_recover and not self.rmp_oracle.bool_is_lp:
                if self.arg.cg_mip_recover:
                    if (int(self.iter) % self.arg.cg_rmp_mip_iter == 0) or (
                        bool_terminate
                    ):
                        choice = self.arg.cg_method_mip_heuristic
                        func = slp.PrimalMethod.select(choice)
                        _fname = func.__name__ if func is not None else "none"
                        with utils.TimerContext(
                            self.iter,
                            f"{_fname}",
                        ):
                            if func is None:
                                pass
                            else:
                                mip_objective = func(self)

                                self._logger.info(
                                    _this_log_line
                                    + f" f_mip: {mip_objective:.6e}, eps_int: {mip_objective - lp_objective:.4e}/{(mip_objective - lp_objective) / lp_objective * 1e2:.2f}%"
                                )
                if bool_terminate and self.arg.rounding_heuristic:
                    K = 20
                    k = 0
                    rounding_terminate = False
                    while k < K and rounding_terminate == False :
                        Q = []
                        model = self.rmp_model
                        self.rmp_oracle.rounding_method(self.columns,self.customer_list,self.iter)
                        self.rmp_model.write(f"{utils.CONF.DEFAULT_SOL_PATH}/rounding_rmp@{self.iter}.mip.mps")
                        model.optimize()
                        mip_objective = model.objval
                        variables = model.getVars()
                        # Step 4. 根据最优解获取Q（fractional column）
                        for v in variables:
                            if self.arg.backend.upper() == 'GUROBI':
                                # if v.VarName.startswith("s") and v.x != 0 and v.x != 1:
                                #     Q.append((v.VarName,v.x))
                                if v.VarName.startswith("s"):
                                    # print(v.VarName,v.x)
                                    if v.x != 0 and v.x != 1:
                                        Q.append((v.VarName,v.x))
                            else:
                                if v.getName().startswith("s") and v.x != 0 and v.x != 1:
                                    Q.append((v.getName(),v.x))
                        if len(Q) == 0:
                            rounding_terminate = True
                        print('-------Rounding Heuristic')
                        print(k,Q)
                        self.rmp_oracle.reset_to_origin()
                        self._logger.info("rounding rmp reset over")
                        with utils.TimerContext(
                                 k,
                                "rounding_heuristic",
                            ):
                                self._logger.info(
                                        _this_log_line
                                        + f" f_mip: {mip_objective:.6e}, eps_int: {mip_objective - lp_objective:.4e}/{(mip_objective - lp_objective) / lp_objective * 1e2:.2f}%"
                                    )
                        k = k + 1
                    # K = 20
                    # k = 0
                    # bool_terminate = False
                    # while k < K and bool_terminate == False :
                    #     Q = []
                    #     model = self.rmp_model
                    #     variables = model.getVars()
                    #     # for v in variables:
                    #     #     if v.getName().startswith("open"):
                    #     #         v.setType(self.solver_constant.BINARY)
                    #     # K = 20
                    #     self.rmp_oracle.rounding_method()
                    #     for t in range(self.arg.T):
                    #         ### Step 1. add constraints(或许考虑customer的也应该加上去)
                    #         self.add_rounding_constraints(t)
                    #     # Step 2. 改变目标函数 加惩罚项
                    #     self.rounding_objective()
                    #     model.optimize()
                    #     # Step 3. 获取最优解
                    #     variables = model.getVars()
                    #     for v in variables:
                    #         if v.getName().startswith("open") and (v.x == 0 or v.x == 1):
                    #             Q.append((v.getName(),v.x))
                    #     if len(Q) == 0:
                    #         bool_terminate = True
                    #     print('-------Rounding Heuristic')
                    #     print(k,Q)
                    #     # Step 4. 根据最优解获取Q（fractional column）
                    #     k = k + 1
                    
                # for debugging
                # self.get_col_weight("./out/", self.iter)
                # self.get_col_weight_his()

                if self.arg.check_cost_cg:
                    # cost checker
                    check_cost_cg(self)
                if bool_terminate:
                    self.red_cost = self.red_cost[: self.iter, :]
                    break

                if bool_early_stop:
                    self._logger.info("early terminated")
                    break

                _obj_last_iterate = self.rmp_objval

            except KeyboardInterrupt as _unused_e:
                self._logger.info("early terminated")
                break
        utils.visualize_timers()
        self._logger.info(f"save solutions to {utils.CONF.DEFAULT_SOL_PATH}")
        # todo
        # self.get_solution(utils.CONF.DEFAULT_SOL_PATH)

    # def rounding_objective(self):
    #     obj = self.original_obj + self.rounding_penalty_objective()

    #     # self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)
    #     self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)

    # def rounding_penalty_objective(self):
    #     obj = 0.0
    #     p1,p2,p3,p4 = 10,20,15,25
    #     for t in range(self.arg.T):
    #         sku_list = edge.get_edge_sku_list(t, self.full_sku_list)
    #         for e, edge in self._iterate_no_c_edges():
    #             obj += np.random.uniform(0, p1)*self.variables["rounding_slack_edge"][t, edge]*(1-self.variables["rounding_slack_edge"][t, edge])
    #             for k in sku_list:
    #                 obj += np.random.uniform(0, p2)*self.variables["rounding_slack_edge_k"][t, edge, k]*(1-self.variables["rounding_slack_edge_k"][t, edge, k])
    #         for node in self._iterate_no_c_nodes():
    #             obj += np.random.uniform(0, p3)*self.variables["rounding_slack_node"][t, node]*(1-self.variables["rounding_slack_node"][t, node])
    #             for k in sku_list:
    #                 obj += np.random.uniform(0, p4)*self.variables["rounding_slack_node_k"][t, node, k]*(1-self.variables["rounding_slack_node_k"][t, node,k])
    #     return obj

    # def add_rounding_constraints(self,t):
    #     for e, edge in self._iterate_no_c_edges():
    #         sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

    #         constr = self.solver.addConstr(
    #             self.variables["select_edge"][t, edge] + self.variables["rounding_slack_edge"][t, edge]
    #             == 1
    #         )
    #         self.constrs["rounding_relationship"]["slack_edge"][
    #                 (t, edge)
    #             ] = constr

    #         for k in sku_list:
    #             constr = self.solver.addConstr(
    #                 self.variables["sku_select_edge"][t, edge, k] + self.variables["rounding_slack_edge_k"][t, edge,k]
    #                 == 1
    #             )
    #             self.constrs["rounding_relationship"]["slack_edge_k"][
    #                 (t, edge, k)
    #             ] = constr

    #     for node in self._iterate_no_c_nodes():
    #         sku_list = node.get_node_sku_list(t, self.full_sku_list)
    #         constr = self.solver.addConstr(
    #                 self.variables["open"][t, node] + self.variables["rounding_slack_node"][t, node] == 1
    #             )
    #         self.constrs["rounding_relationship"]["slack_node"][(t, node)] = constr
    #         for k in sku_list:
    #             constr = self.solver.addConstr(
    #                     self.variables["sku_open"][t, node, k] + self.variables["rounding_slack_node_k"][t, node,k] == 1
    #                 )
    #             self.constrs["rounding_relationship"]["slack_node_k"][(t, node, k)] = constr

# def init_rmp_by_cols(self):
    #     """
    #     update the RMP with new columns
    #     """
    #     if not self.bool_rmp_update_initialized:
    #         # sku_flow
    #         self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
    #         self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
    #         # lambda
    #         self.ws_cons_idx = {customer: 0 for customer in self.customer_list}
    #         for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
    #             edges = self.rmp_oracle.cg_downstream[node]
    #             for ee in edges:
    #                 c = ee.end
    #                 self.delievery_cons_idx[c].append(v)
    #
    #         for c, v in self.rmp_oracle.cg_binding_constrs_ws.items():
    #             self.ws_cons_idx[c] = v
    #             # v.lb = 1.0
    #             # v.ub = 1.0
    #             self.solver.setEqualConstr(v, 1.0)
    #
    #     for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
    #         edges = self.rmp_oracle.cg_downstream[node]
    #         for ee in edges:
    #             c = ee.end
    #             this_col = self.columns[c][-1]
    #             self.delievery_cons_coef[c].append(
    #                 -this_col["sku_flow"].get((t, ee, k), 0)
    #             )
    #
    #     for c in self.customer_list:
    #         _cons_idx = [*self.delievery_cons_idx[c], self.ws_cons_idx[c]]
    #         _cons_coef = [*self.delievery_cons_coef[c], 1]
    #         col_idxs = list(zip(_cons_idx, _cons_coef))
    #         try:
    #             new_col = self.solver.addColumn(_cons_idx, _cons_coef)
    #             self.rmp_oracle.variables["column_weights"][
    #                 c, self.columns[c].__len__() - 1
    #             ] = self.rmp_model.addVar(
    #                 obj=self.columns[c][-1]["beta"],
    #                 name=f"lambda_{c.idx}_{len(self.columns[c])}",
    #                 lb=0.0,
    #                 ub=1.0,
    #                 vtype=self.solver_constant.CONTINUOUS,
    #                 column=new_col,
    #             )
    #         except Exception as e:
    #             print(f"failed at {c}\n\t{col_idxs}")
    #             raise e

    def update_rmp_by_cols(self):
        if not self.bool_rmp_update_initialized:
            # now the bindings have been created
            # do the clean-ups
            sla.cleanup(self)

        sla.update(self)

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
                    if self.columns_status[customer][number] == 1:
                        cus_col_weights.loc[
                            number, customer.idx
                        ] = self.solver.getVarValue(
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

    def get_col_weight(self, data_dir: str = "./", iter: int = 0):
        self.variables["column_weights"] = self.rmp_oracle.variables["column_weights"]
        cus_col_weights = pd.DataFrame(
            index=range(self.iter + 1), columns=[c.idx for c in self.customer_list]
        )

        for customer in self.customer_list:
            # for number in range(self.num_cols):
            for number in range(self.iter - 1):
                number = number + 1
                if self.columns[customer][number] != {}:
                    if self.columns_status[customer][number] == 1:
                        cus_col_weights.loc[
                            number, customer.idx
                        ] = self.solver.getVarValue(
                            self.variables["column_weights"][customer, number]
                        )
                else:
                    cus_col_weights.loc[number, customer.idx] = 0
        num_cus = len(self.customer_list)
        cus_col_weights.to_csv(
            os.path.join(
                data_dir,
                "cus" + str(num_cus) + "_col_weight_iter_" + str(iter) + ".csv",
            ),
            index=False,
        )

    def get_col_weight_his(self):
        """
        I have an assumption that columns may have preference level between each other
        Specifically, those columns set as 0 weight may not be used in the future,
        i.e. with 0 weight, or at least not high weight
        So I want to check the columns' weight to see if it is true
        """
        for customer in self.customer_list:
            # for number in range(self.num_cols):
            for number in range(self.iter - 1):
                # number = number + 1
                if number not in self.col_weight_history[customer.idx]:
                    self.col_weight_history[customer.idx][number] = []
                if self.columns_status[customer][number] == 1:
                    self.col_weight_history[customer.idx][number].append(
                        self.solver.getVarValue(
                            self.rmp_oracle.variables["column_weights"][
                                customer, number
                            ]
                        )
                    )

    def watch_col_weight(self):
        zero_weight_col_used = {cus.idx: 0 for cus in self.customer_list}
        for cus in self.customer_list:
            for col_idx, col_weight in self.col_weight_history[cus.idx].items():
                if np.mean(col_weight) > 0 and np.min(col_weight) == 0:
                    zero_weight_col_used[cus.idx] += 1

        # write zero_weight_col_used to csv file
        df = pd.DataFrame(zero_weight_col_used, index=[0])
        df.to_csv("./out/zero_weight_col_used.csv", index=False)


if __name__ == "__main__":
    pass
