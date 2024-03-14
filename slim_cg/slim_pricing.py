import argparse
import os

import networkx as nx
import ray
from coptpy import COPT
from gurobipy import GRB

from entity import *
from solver_wrapper import CoptWrapper, GurobiWrapper
from solver_wrapper.CoptConstant import CoptConstant
from solver_wrapper.GurobiConstant import GurobiConstant
from utils import logger

ATTR_IN_RMPSLIM = ["sku_flow"]
# macro for debugging
CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 0))
CG_SUBP_LOGGING = int(os.environ.get("CG_SUBP_LOGGING", 0))
CG_SUBP_THREADS = int(os.environ.get("CG_SUBP_THREADS", 2))
CG_SUBP_GAP = float(os.environ.get("CG_SUBP_GAP", 0.05))
CG_SUBP_TIMELIMIT = float(os.environ.get("CG_SUBP_TIMELIMIT", 100))
# cg big M
BIG_M_SELECT_EDGE = 1e5


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
        self.bool_covering = self.arg.covering
        self.bool_edge_lb = bool_edge_lb
        self.bool_node_lb = bool_node_lb
        self.solver = solver
        self.columns_dict = {}

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
                customer=customer,
                solver=self.solver,
            )
            oracle.modeling(customer)
            self.DNP_dict[customer] = oracle
            self.columns_dict[customer] = oracle.column

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

    # def query_all_columns(self):
    #     columns = []
    #     for customer in self.cus_list:
    #         columns.append(self.DNP_dict[customer].query_columns())
    #     return columns

    def query_all_columns(self):
        return [self.DNP_dict[customer].query_columns() for customer in self.cus_list]

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

    def update_objective_all_new(
            self,
            iter,
            broadcast_mat,
            broadcast_nodes,
            broadcast_cols,
            broadcast_keys,
            dual_ws,
            dual_exists_customers,
    ):
        for customer in self.cus_list:
            if customer in self.skipped:
                continue
            if iter >= 1 and customer in dual_exists_customers:
                ccols = broadcast_cols[customer]
                dual_pack_this = (
                    broadcast_keys[customer],
                    broadcast_mat[ccols, :] @ broadcast_nodes,
                    dual_ws[customer],
                )
            else:
                dual_pack_this = None

            self.DNP_dict[customer].update_objective(customer, dual_pack_this)

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

    def get_all_var_keys(self, var_type):
        var_keys = []
        for customer in self.cus_list:
            var_keys.append(self.DNP_dict[customer].get_var_keys(var_type))
        return var_keys

    def set_all_relaxation(self):
        for customer in self.cus_list:
            self.DNP_dict[customer].relaxation()


class Pricing(object):
    """
    this is a class for dynamic network flow (DNP)
    """

    def __init__(
            self,
            arg: argparse.Namespace,
            network: nx.DiGraph,
            model_name: str = "PricingDelivery",
            bool_dp: bool = False,
            gap: float = 1e-4,
            threads: int = None,
            limit: int = 3600,
            customer: Customer = None,
            solver: str = "COPT",
    ) -> None:
        self.backend = solver.upper()
        self.solver_name = solver.upper()
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
        # self.sku_list = self.network.graph["sku_list"]
        self.sku_list = self.arg.sku_list
        # self.bool_capacity = bool_capacity
        # self.bool_fixed_cost = bool_fixed_cost
        # self.bool_covering = bool_covering
        self.bool_fixed_cost = self.arg.fixed_cost
        self.bool_covering = self.arg.covering
        self.bool_capacity = self.arg.capacity
        self.add_in_upper = self.arg.add_in_upper
        self.add_distance = self.arg.add_distance
        self.add_cardinality = self.arg.add_cardinality
        self.bool_edge_lb = False

        self.sku_flow_keys = None
        self.sku_flow_vars = None
        # self.bool_edge_lb = self.arg.edgelb
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

        self.var_idx = None
        self.dual_index_for_RMP = {
            "transportation_capacity": dict(),
            "node_capacity": dict(),
        }
        self.index_for_dual_var = 0  # void bugs of index out of range

        # for remote
        self.columns_helpers = None
        self.primal_vars_sorted = None
        self.args_modeling = (bool_dp, model_name, gap, limit, threads, CG_SUBP_LOGGING)

    def model_reset(self):
        self.model.reset()

    def get_model_objval(self):
        return self.model.objval

    def get_model_status(self):
        return self.model.status

    def modeling(self, customer):
        bool_dp, model_name, gap, limit, threads, logging, *_ = self.args_modeling
        if bool_dp:
            raise ValueError("not implemented")
        else:
            self.modeling_milp(model_name, gap, limit, threads, logging, customer)

    #####################

    def modeling_milp(self, model_name, gap, limit, threads, logging, customer):
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
        if self.solver_name == "GUROBI":
            self.model.setParam("OutputFlag", logging)
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
        self.add_constraints(customer)

        # print("set objective ...")
        self.set_objective()
        # for remote
        self.init_col_helpers()

    def _iterate_edges(self):
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            if edge.end.type == const.CUSTOMER:
                yield edge

    # 更改点1 iterate node约束
    def _iterate_nodes(self):
        for n in self.network.nodes:
            if n.type != const.CUSTOMER:
                yield n

    def add_vars(self):
        """
        add variables
        """
        if self.bool_covering:
            self.var_types = {
                "sku_flow": {
                    "lb": 0,
                    "ub": self.solver_constant.INFINITY,
                    # "lb": [],
                    # "ub": [],
                    # "ub": 0,
                    "vtype": self.solver_constant.CONTINUOUS,
                    "nameprefix": "w",
                    "index": "(t, edge, k)",
                },
                "select_edge": {
                    "lb": 0,
                    "ub": 1,
                    "vtype": self.solver_constant.BINARY,
                    "nameprefix": "p",
                    "index": "(t, edge)",
                },
                "sku_select_edge": {
                    "lb": 0,
                    "ub": 1,
                    "vtype": self.solver_constant.BINARY,
                    "nameprefix": "p",
                    "index": "(t, edge,k)",
                },
                "open": {
                    "lb": 0,
                    "ub": 1,
                    "vtype": self.solver_constant.BINARY,
                    "nameprefix": "p",
                    "index": "(t, node)",
                },
            }
        else:
            self.var_types = {
                "sku_flow": {
                    "lb": 0,
                    "ub": self.solver_constant.INFINITY,
                    # "lb": [],
                    # "ub": [],
                    # "ub": 0,
                    "vtype": self.solver_constant.CONTINUOUS,
                    "nameprefix": "w",
                    "index": "(t, edge, k)",
                },
            }
        if self.arg.backorder:
            self.var_types["sku_backorder"] = {
                "lb": 0,
                # "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "s",
                "index": "(t, k)",
            }
        else:
            self.var_types["sku_slack"] = {
                "lb": 0,
                "ub": [],  # TBD
                # "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "s",
                "index": "(t, k)",
            }
        idx = dict()
        for vt in self.var_types.keys():
            idx[vt] = list()
        if self.bool_covering:
            for node in self._iterate_nodes():
                for t in range(self.T):
                    # select node i at t
                    idx["open"].append((t, node))
            for edge in self._iterate_edges():
                for t in range(self.T):
                    # select edge (i,j) at t
                    idx["select_edge"].append((t, edge))
                    for k in self.sku_list:
                        idx["sku_select_edge"].append((t, edge, k))

        for t in range(self.T):
            for k in self.sku_list:
                if self.arg.backorder:
                    idx["sku_backorder"].append((t, k))
                else:
                    idx["sku_slack"].append((t, k))
                    self.var_types["sku_slack"]["ub"].append(
                        self.customer.demand.get((t, k), 0)
                    )

        for edge in self._iterate_edges():
            for t in range(self.T):
                for k in self.sku_list:
                    # flow of sku k on edge (i,j) at t
                    idx["sku_flow"].append((t, edge, k))

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
                ub=param["ub"] if "ub" in param else self.solver_constant.INFINITY,
                vtype=param["vtype"],
                nameprefix=f"{param['nameprefix']}_",
            )

        # to avoid repeatedly open new mem for column
        self.column = {}
        self.column["sku_flow"] = {k: 0.0 for k in self.var_idx["sku_flow"].keys()}

        self.column["beta"] = 0.0
        self.column["transportation_cost"] = 0
        self.column["unfulfilled_demand_cost"] = 0.0

    def get_var_keys(self, var_type):
        return self.var_idx[var_type].keys()

    def relaxation(self):
        variables = self.model.getVars()
        for v in variables:
            if v.getType() == COPT.BINARY:
                v.setType(COPT.CONTINUOUS)

    def add_constraints(self, customer):
        if self.bool_covering:
            self.constr_types = {
                "flow_conservation": {"index": "(t, node, k)"},
                "open_relationship": {
                    "select_edge": {"index": "(t, edge, node)"},
                    "sku_select_edge": {"index": "(t, edge, k)"},
                    "sku_flow_select": {"index": "(t, edge, k)"},
                    "open": {"index": "(t, warehouse with demand / customer)"},
                    "sku_open": {"index": "(t, node, k)"},
                },
                # "transportation_capacity": {"index": "(t, edge)"},
            }
        else:
            self.constr_types = {"flow_conservation": {"index": "(t, node, k)"}}
        if self.bool_capacity:
            self.constr_types["transportation_capacity"] = {"index": "(t, edge)"}
        if self.bool_edge_lb:
            self.constr_types["transportation_variable_lb"] = {"index": "(t, edge)"}

        for constr in self.constr_types.keys():
            self.constrs[constr] = dict()
        if self.bool_covering:
            for constr in self.constr_types["open_relationship"].keys():
                self.constrs["open_relationship"][constr] = dict()

        # for t in tqdm(range(self.T)):
        for t in range(self.T):
            # initial status and flow conservation
            self.add_constr_flow_conservation(t)
            if self.bool_covering:
                # node status and open relationship
                self.add_constr_open_relationship(t)
                # if self.arg.add_cardinality:
                #     self.add_constr_cardinality(t)
            if self.bool_capacity:
                # transportation/production/holding capacity
                self.add_constr_transportation_capacity(t)
            if self.bool_edge_lb:
                self.add_constr_transportation_lb(t)
            if self.add_distance:
                self.add_constr_distance(t, customer)
            if self.add_cardinality:
                self.add_constr_cardinality(t, customer)

    def add_constr_flow_conservation(self, t: int):
        """flow constraint in the pricing problem
        for each exclusive customer
        """
        edges = list(self._iterate_edges())
        for k in self.sku_list:
            constr_name = f"flow_conservation_{t}_{k.idx}"
            if self.arg.backorder:
                if t == 0:
                    constr = self.solver.addConstr(
                        self.variables["sku_flow"].sum(t, edges, k)
                        + self.variables["sku_backorder"][(t, k)]
                        == self.customer.demand.get((t, k), 0),
                        name=constr_name,
                    )
                else:
                    constr = self.solver.addConstr(
                        self.variables["sku_flow"].sum(t, edges, k)
                        + self.variables["sku_backorder"][(t, k)]
                        == self.customer.demand.get((t, k), 0)
                        + self.variables["sku_backorder"][(t - 1, k)],
                        name=constr_name,
                    )
            else:
                constr = self.solver.addConstr(
                    self.variables["sku_flow"].sum(t, edges, k)
                    + self.variables["sku_slack"][(t, k)]
                    == self.customer.demand.get((t, k), 0),
                    name=constr_name,
                )
            self.constrs["flow_conservation"][(t, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):
        for edge in self._iterate_edges():
            # sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

            constr = self.model.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.start]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.start)
            ] = constr

            # self.constrs["open_relationship"]["select_edge"][
            #     (t, edge, edge.end)
            # ] = constr = self.model.addConstr(
            #     self.variables["select_edge"][t, edge]
            #     <= self.variables["open"][t, edge.end]
            # )

            for k in self.sku_list:
                constr = self.model.addConstr(
                    self.variables["sku_select_edge"][t, edge, k]
                    <= self.variables["select_edge"][t, edge]
                )
                self.constrs["open_relationship"]["sku_select_edge"][
                    (t, edge, k)
                ] = constr
            for k in self.sku_list:
                # constr = self.model.addConstr(
                constr = self.model.addConstr(
                    self.variables["sku_select_edge"][t, edge, k]
                    <= self.variables["select_edge"][t, edge]
                )
                self.constrs["open_relationship"]["sku_select_edge"][
                    (t, edge, k)
                ] = constr

                constr = self.model.addConstr(
                    self.variables["sku_flow"][t, edge, k]
                    <= BIG_M_SELECT_EDGE * self.variables["sku_select_edge"][t, edge, k]
                )
                self.constrs["open_relationship"]["sku_flow_select"][
                    (t, edge, k)
                ] = constr
        # todo
        # for node in self._iterate_no_c_nodes():
        #     sku_list = node.get_node_sku_list(t, self.full_sku_list)

        #     if (
        #             node.type == const.WAREHOUSE
        #             and node.has_demand(t)
        #             and len(node.demand_sku[t]) > 0
        #     ):
        #         # constr = self.model.addConstr(self.variables["open"][t, node] == 1)
        #         constr = self.solver.addConstr(self.variables["open"][t, node] == 1)
        #         self.constrs["open_relationship"]["open"][(t, node)] = constr
        #     elif node.type == const.CUSTOMER:
        #         # constr = self.model.addConstr(self.variables["open"][t, node] == 1)
        #         constr = self.solver.addConstr(self.variables["open"][t, node] == 1)
        #         self.constrs["open_relationship"]["open"][(t, node)] = constr

        #     for k in sku_list:
        #         if node.type == const.PLANT:
        #             # constr = self.model.addConstr(
        #             constr = self.solver.addConstr(
        #                 self.variables["sku_open"][t, node, k]
        #                 <= self.variables["open"][t, node]
        #             )

        #         self.constrs["open_relationship"]["sku_open"][(t, node, k)] = constr
        return

    def add_constr_transportation_lb(self, t: int, verbose=False):
        for edge in self._iterate_edges():
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")

            # variable lower bound
            if edge.variable_lb < np.inf:
                self.constrs["transportation_variable_lb"][
                    (t, edge)
                    # ] = self.model.addConstr(
                ] = self.solver.addConstr(
                    flow_sum
                    >= edge.variable_lb * self.variables["select_edge"][t, edge],
                    name=f"edge_lb_{t}_{edge.start}_{edge.end}",
                )

                self.index_for_dual_var += 1

        return

    def add_constr_transportation_capacity(self, t: int, verbose=False):
        for edge in self._iterate_edges():
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")

            if edge.capacity < np.inf:
                if self.arg.covering:
                    bound = self.variables["select_edge"][t, edge]
                else:
                    bound = 1
                if type(flow_sum) is not float:
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

    def add_constr_cardinality(self, t: int, customer: Customer):
        used_edge = 0
        for edge in self._iterate_edges():
            used_edge += self.variables["select_edge"][t, edge]
        constr = self.model.addConstr(used_edge <= self.arg.cardinality_limit)
        self.constrs["cardinality"][(t, customer)] = constr
        self.index_for_dual_var += 1

    def add_constr_distance(self, t: int, customer: Customer):
        used_distance = 0
        for edge in self._iterate_edges():
            used_distance += self.variables["select_edge"][t, edge] * edge.distance
        constr = self.model.addConstr(used_distance <= self.arg.distance_limit)
        self.constrs["distance"][(t, customer)] = constr
        self.index_for_dual_var += 1

    def get_original_objective(self):
        """
        Get the original objective value
        """
        obj = 0.0
        for t in range(self.T):
            obj = obj + self.cal_sku_transportation_cost(t)
            if self.arg.backorder:
                obj = obj + self.cal_sku_backlogged_demand_cost(t)
            else:
                obj = obj + self.cal_sku_unfulfilled_demand_cost(t)
        if self.bool_fixed_cost:
            obj = obj + self.cal_fixed_node_cost()
        else:
            self.obj["fixed_node_cost"] = 0

        return obj

    def extra_objective(self, customer, dual_packs=None):
        if dual_packs is None:
            return 0.0
        dual_keys, dual_vals, dual_ws = dual_packs
        # dual_series, dual_ws = dual_packs

        obj = sum(
            self.variables["sku_flow"].get((t, ee, k), 0) * v
            for (ee, k, t), v in zip(dual_keys, dual_vals)
        )

        return obj - dual_ws

    def update_objective(self, customer, dual_packs):
        """
        Use dual variables to calculate the reduced cost
        """

        obj = self.original_obj + self.extra_objective(customer, dual_packs)

        self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)

    def set_objective(self):
        self.obj_types = {
            "transportation_cost": {"index": "(t, edge)"},
            "unfulfilled_demand_cost": {"index": "(t, c, k)"},
            "backlogged_demand_cost": {"index": "(t, c, k)"},
            "fixed_node_cost": {"index": "(t)"},
        }
        for obj in self.obj_types.keys():
            self.obj[obj] = dict()

        self.original_obj = self.get_original_objective()

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

        self.obj["transportation_cost"][t] = transportation_cost

        return transportation_cost

    def cal_sku_backlogged_demand_cost(self, t: int):
        backlogged_demand_cost = 0.0
        for k in self.sku_list:
            backlogged_demand_cost += (
                    self.arg.unfulfill_sku_unit_cost
                    * self.variables["sku_backorder"][(t, k)]
            )
        self.obj["backlogged_demand_cost"][t] = backlogged_demand_cost
        return backlogged_demand_cost

    def cal_sku_unfulfilled_demand_cost(self, t: int):
        unfulfilled_node_cost = 0.0
        # if self.customer.has_demand(t):
        for k in self.sku_list:
            unfulfilled_sku_unit_cost = self.arg.unfulfill_sku_unit_cost
            unfulfilled_node_cost += unfulfilled_sku_unit_cost * self.variables[
                "sku_slack"
            ].get((t, k), 0)
        self.obj["unfulfilled_demand_cost"][t] = unfulfilled_node_cost

        return unfulfilled_node_cost

    def cal_fixed_node_cost(self):
        fixed_node_cost = 0.0

        if not self.bool_covering:
            return fixed_node_cost

        for node in self._iterate_nodes():
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
                node_fixed_node_cost += (
                        this_node_fixed_cost * self.variables["open"][(t, node)]
                )

            fixed_node_cost += node_fixed_node_cost
            self.obj["fixed_node_cost"][node] = node_fixed_node_cost

        return fixed_node_cost

    def cal_fixed_edge_cost(self, t):
        fixed_edge_cost = 0.0
        for edge in self._iterate_edges():
            fixed_edge_cost += (
                    edge.transportation_fixed_cost
                    * self.variables["select_edge"][(t, edge)]
            )
        self.obj["edge_fixed_cost"][(t)] = fixed_edge_cost
        return fixed_edge_cost

    def solve(self):
        self.solver.solve()
        if self.sku_flow_keys is None:
            tmp = dict(self.variables["sku_flow"])
            self.sku_flow_keys = [v.index for k, v in tmp.items()]
            self.sku_flow_vars = [v for k, v in tmp.items()]

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

    def _query_a_expr_or_float_or_constraints(v):
        if type(v) is not float:
            return v.getExpr().getValue()
        else:
            return v

    def eval_helper(self):
        # TOCHeck：column的时候只有sku_flow和beta
        _vals = {}
        _vals["beta"] = self._query_a_expr_or_float_or_variable(
            self.columns_helpers["beta"]
        )
        _vals["objval"] = self.model.objval
        _vals["status"] = self.model.status
        _vals["sku_flow"] = {}
        if len(self.sku_flow_keys) > 0:
            if self.arg.backend.upper() == "COPT":
                _vals["sku_flow"] = dict(
                    self.model.getInfo(COPT.Info.Value, self.variables["sku_flow"])
                )
            else:
                _vals["sku_flow"] = dict(
                    self.model.getAttr(GRB.Attr.X, self.variables["sku_flow"])
                )

        if CG_EXTRA_DEBUGGING:
            if type(self.columns_helpers["transportation_cost"]) != float:
                transportation_cost = 0
                for t in self.columns_helpers["transportation_cost"].keys():
                    z = self.columns_helpers["transportation_cost"][t]
                    if type(z) is not float:
                        period_transportation_cost = z.getExpr().getValue()
                    else:
                        period_transportation_cost = z
                    transportation_cost = (
                            period_transportation_cost + transportation_cost
                    )
                _vals["transportation_cost"] = transportation_cost
            else:
                # print("tr",self.columns_helpers["transportation_cost"])
                _vals["transportation_cost"] = 0
            _vals["unfulfilled_demand_cost"] = {t: 0 for t in range(self.T)}
            if type(self.columns_helpers["unfulfilled_demand_cost"]) != float:
                unfulfilled_demand_cost = 0
                for t in self.columns_helpers["unfulfilled_demand_cost"].keys():
                    z = self.columns_helpers["unfulfilled_demand_cost"][t]
                    if type(z) is not float:
                        period_unfulfilled_demand_cost = z.getExpr().getValue()
                    else:
                        period_unfulfilled_demand_cost = z
                    # print(t,z)
                    unfulfilled_demand_cost = (
                            unfulfilled_demand_cost + period_unfulfilled_demand_cost
                    )
                    _vals["unfulfilled_demand_cost"][t] = period_unfulfilled_demand_cost
            else:
                for t in self.columns_helpers["unfulfilled_demand_cost"].keys():
                    _vals["unfulfilled_demand_cost"][t] = 0

        return _vals

    def init_col_helpers(self):
        """
        Initialize the column helpers to extract frequently
            needed quantities for the subproblems
        """
        col_helper = {}
        try:
            col_helper["beta"] = (
                self.original_obj.getExpr()
                if self.solver_name == "COPT"
                else self.original_obj
            )
        except:
            logger.warning(f"customer {self.customer} has null objective")
            col_helper["beta"] = 0.0
        try:
            col_helper["transportation_cost"] = self.obj["transportation_cost"]
        except:
            col_helper["transportation_cost"] = 0.0
        if self.arg.backorder:
            label = "backlogged_demand_cost"
        else:
            label = "unfulfilled_demand_cost"
        try:
            col_helper["unfulfilled_demand_cost"] = self.obj[label]
        except:
            col_helper["unfulfilled_demand_cost"] = 0.0

        # col_helper["sku_flow"] = pd.Series(
        #     {k: 0 for k, v in self.variables["sku_flow"].items()}
        # )

        self.columns_helpers = col_helper
        self.columns = []

    def query_columns(self):
        new_col = self.eval_helper()

        # new to avoid repeatedly open new mem for column
        # for k, v in self.variables["sku_flow"].items():
        #     self.column["sku_flow"][k] = v.x
        # self.column["beta"] = self._query_a_expr_or_float_or_variable(
        #     self.columns_helpers["beta"]
        # )
        self.column = new_col

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

    def add_in_upper(self, node, t):
        in_inventory_sum = 0
        for e in self.network.edges:
            edge = self.network.edges[e]["object"]
            if edge.end == node:
                in_inventory_sum += self.variables["sku_flow"].sum(t, edge, "*")
        self.model.addConstr(in_inventory_sum <= node.inventory_capacity * 0.4)
        self.index_for_dual_var += 1

    def add_distance(self, t):
        # TODO: add_distance 的部分加上去
        return
