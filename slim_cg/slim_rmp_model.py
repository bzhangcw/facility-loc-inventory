import utils
import datetime

from dnp_model import *
from entity import *
from solver_wrapper import GurobiWrapper, CoptWrapper
from solver_wrapper.CoptConstant import CoptConstant
from solver_wrapper.GurobiConstant import GurobiConstant

CG_RMP_LOGGING = int(os.environ.get("CG_RMP_LOGGING", 1))
CG_RMP_METHOD = int(os.environ.get("CG_RMP_METHOD", 4))


class DNPSlim(DNP):
    """
    the modeling class for dynamic network flow (DNP),
        customer free version.
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
        threads: int = 12,
        limit: int = 7200,
        cg: bool = True,
        customer_list: List[Customer] = None,
        env=None,
        solver="COPT",
    ) -> None:
        if solver == "COPT":
            self.solver = CoptWrapper.CoptWrapper(model_name)
            self.solver_constant = CoptConstant
        elif solver == "GUROBI":
            self.solver = GurobiWrapper.GurobiWrapper(model_name)
            self.solver_constant = GurobiConstant
        else:
            raise ValueError("solver must be either COPT or GUROBI")

        self.arg = arg
        self.T = arg.T
        self.network = network
        self.full_sku_list = (
            full_sku_list
            if full_sku_list is not None
            else self.network.graph["sku_list"]
        )
        self.customer_list = customer_list

        self.env = self.solver.ENVR
        self.model = self.solver.model
        self.cg = cg

        self.model.setParam("Logging", CG_RMP_LOGGING)
        self.model.setParam("LogToConsole", CG_RMP_LOGGING)
        self.model.setParam("Crossover", 0)
        self.model.setParam(self.solver_constant.Param.RelGap, 0.015)
        self.model.setParam(self.solver_constant.Param.TimeLimit, limit)
        self.model.setLogFile(
            f"{utils.CONF.DEFAULT_TMP_PATH}/rmp-{datetime.datetime.now()}.log"
        )
        self.model.setParam(self.solver_constant.Param.LpMethod, 4)
        if threads is not None:
            self.model.setParam(self.solver_constant.Param.Threads, threads)

        self.variables = dict()  # variables
        self.constrs = dict()  # constraints
        self.obj = dict()  # objective
        self.bool_fixed_cost = self.arg.fixed_cost
        self.bool_covering = self.arg.covering
        self.bool_capacity = self.arg.capacity
        self.add_in_upper = self.arg.add_in_upper
        self.bool_edge_lb = False
        self.bool_node_lb = False
        # self.add_distance = self.arg.add_distance
        # self.add_cardinality = self.arg.add_cardinality
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
        self.var_idx = None
        self.dual_index_for_RMP = {
            "transportation_capacity": dict(),
            "node_capacity": dict(),
            # "weights_sum": dict(),
        }
        self.index_for_dual_var = 0  # void bugs of index out of range

        # for remote
        self.columns_helpers = None

        self.bool_is_lp = False
        self.binaries = []

    #####################

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

        # for remote
        # self.init_col_helpers()

    def init_col_helpers(self):
        """
        Initialize the column helpers to extract frequently
            needed quantities for the subproblems

        """
        # utils.logger.info("generating column helpers for %s" % self.cus_list[0].idx)

        # col_helper = {attr: {t: {} for t in range(self.oracles[customer].T)} for attr in ATTR_IN_RMP}
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

                col_helper["sku_flow_sum"][t][edge] = self.variables["sku_flow"].sum(
                    t, edge, "*"
                )

            for node in self.network.nodes:
                if node.type == const.PLANT:
                    # if node.production_capacity == np.inf:
                    #     continue
                    # col_helper["sku_production_sum"][t][node] = self.variables[
                    # col_helper["sku_production_sum"][t][node.idx] = self.variables[
                    col_helper["sku_production_sum"][t][node] = self.variables[
                        "sku_production"
                    ].sum(t, node, "*")
                elif node.type == const.WAREHOUSE:
                    # node holding capacity
                    # if node.inventory_capacity == np.inf:
                    #     continue
                    # col_helper["sku_inventory_sum"][t][node] = self.variables[
                    # col_helper["sku_inventory_sum"][t][node.idx] = self.variables[
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

    def _iterate_no_c_edges(self):
        if self.cg:
            for e in self.network.edges:
                edge = self.network.edges[e]["object"]

                if edge.end.type == const.CUSTOMER or edge.start.type == const.CUSTOMER:
                    continue
                yield e, edge
        else:
            for e in self.network.edges:
                edge = self.network.edges[e]["object"]
                yield e, edge

    def _iterate_no_c_nodes(self):
        if self.cg:
            for node in self.network.nodes:
                if node.type == const.CUSTOMER:
                    continue
                yield node
        else:
            for node in self.network.nodes:
                yield node

    def add_vars(self):
        """
        add variables
        """
        self.var_types = {
            "sku_flow": {
                "lb": 0,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "w",
                "index": "(t, edge, k)",
            },
            "sku_production": {
                "lb": 0,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "x",
                "index": "(t, plant, k)",
            },
            "sku_delivery": {
                "lb": 0,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "z",
                "index": "(t, warehouse, k)",
            },
            "sku_inventory": {
                # "lb": -self.solver_constant.INFINITY if self.arg.backorder is True else 0,
                "lb": 0,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "I",
                "index": "(t, warehouse, k)",
            },
            # "sku_demand_slack": {
            #     "lb": 0,
            #     "ub": [],  # TBD
            #     # "ub": self.solver_constant.INFINITY,
            #     "vtype": self.solver_constant.CONTINUOUS,
            #     "nameprefix": "s",
            #     "index": "(t, customer, k)",
            # },
        }

        if self.bool_covering:
            self.var_types["select_edge"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.BINARY,
                "nameprefix": "p",
                "index": "(t, edge)",
            }
            self.var_types["sku_select_edge"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.BINARY,
                "nameprefix": "pk",
                "index": "(t, edge, k)",
            }
            self.var_types["open"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.BINARY,
                "nameprefix": "y",
                "index": "(t, node)",
            }
            self.var_types["sku_open"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.BINARY,
                "nameprefix": "yk",
                "index": "(t, plant, k)",
            }

        # generate index tuple
        idx = dict()
        for vt in self.var_types.keys():
            idx[vt] = list()

        # periods
        for t in range(self.T):
            for e, edge in self._iterate_no_c_edges():
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
            for node in self._iterate_no_c_nodes():
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
                        idx["sku_delivery"].append((t, node, k))

        # for initializaiton in CG
        self.var_idx = {}
        for var in idx.keys():
            self.var_idx[var] = {key: 0 for key in idx[var]}

        ##########################
        # add variables
        for vt, param in self.var_types.items():
            self.variables[vt] = self.solver.addVars(
                idx[vt],
                lb=param["lb"],
                ub=param["ub"] if "ub" in param else self.solver_constant.INFINITY,
                vtype=param["vtype"],
                nameprefix=f"{param['nameprefix']}_",
            )
        # self.variables["cg_temporary"] = self.model.addVars(
        self.variables["cg_temporary"] = self.solver.addVars(
            [c.idx for c in self.customer_list], nameprefix="lbdtempo"
        )
        self.variables["column_weights"] = {}

        # record binary variables
        variables = self.model.getVars()

        self.binaries = [v for v in variables if v.getType() == COPT.BINARY]
        # preset this to continuous
        # if only the lagrangian bound is concerned, i.e., rmp is a relaxation
        # it is never set back
        if self.binaries.__len__() == 0:
            self.bool_is_lp = True
        self.switch_to_lp()

    def switch_to_milp(self):
        for v in self.binaries:
            v.setType(COPT.BINARY)
        variables = self.model.getVars()
        for v in variables:
            if v.getName().startswith("lambda"):
                v.setType(COPT.BINARY)
        self.model.setParam("CutLevel", 0)
        self.model.setParam("HeurLevel", 3)

    def switch_to_lp(self):
        for v in self.binaries:
            v.setType(COPT.CONTINUOUS)
        variables = self.model.getVars()
        for v in variables:
            if v.getName().startswith("lambda"):
                v.setType(COPT.CONTINUOUS)

    def add_constraints(self):
        if self.bool_capacity:
            self.constr_types = {
                "flow_conservation": {"index": "(t, node, k)"},
                "transportation_capacity": {"index": "(t, edge)"},
                "production_capacity": {"index": "(t, node)"},
                "holding_capacity": {"index": "(t, node)"},
            }
        else:
            self.constr_types = {
                "flow_conservation": {"index": "(t, node, k)"},
            }
        if self.bool_covering:
            covering_constr_types = {
                "select_edge": {"index": "(t, edge, node)"},
                "sku_select_edge": {"index": "(t, edge, k)"},
                "open": {"index": "(t, warehouse with demand / customer)"},
                "sku_open": {"index": "(t, node, k)"},
                "sku_flow_select": {"index": "(t, edge, k)"},
            }
            self.constr_types["open_relationship"] = covering_constr_types
        if self.bool_edge_lb:
            self.constr_types["transportation_variable_lb"] = {"index": "(t, edge)"}
        if self.bool_node_lb:
            self.constr_types["production_variable_lb"] = {"index": "(t, node)"}
            self.constr_types["holding_variable_lb"] = {"index": "(t, node)"}
        if self.add_in_upper:
            self.constr_types["in_upper"] = {"index": "(t, node)"}

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
            # if self.arg.capacity_for_c is False:
            if self.bool_capacity:
                # transportation/production/holding capacity
                self.add_constr_transportation_capacity(t)
                self.add_constr_production_capacity(t)
                self.add_constr_holding_capacity(t)
            if self.bool_edge_lb:
                self.add_constr_transportation_lb(t)
            if self.bool_node_lb:
                self.add_constr_node_lb(t)
            if self.add_in_upper:
                self.add_constr_flow_in_upper(t)

    def add_constr_flow_conservation(self, t: int):
        for node in self._iterate_no_c_nodes():
            in_edges = get_in_edges(self.network, node)
            out_edges = get_out_edges(self.network, node)
            out_edges_master = [e for e in out_edges if e.end.type != const.CUSTOMER]

            sku_list = node.get_node_sku_list(t, self.full_sku_list)
            for k in sku_list:
                constr_name = f"flow_conservation_{t}_{node.idx}_{k.idx}"

                if node.type == const.PLANT:
                    constr = self.solver.addConstr(
                        self.variables["sku_production"][t, node, k]
                        - self.variables["sku_flow"].sum(t, out_edges_master, k)
                        == 0,
                        name=constr_name,
                    )

                elif node.type == const.WAREHOUSE:
                    fulfilled_demand = 0
                    last_period_inventory = 0.0

                    if t == 0:
                        if node.initial_inventory is not None:
                            # if self.open_relationship:
                            # todo
                            # self.solver.addConstr(
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

                    constr = self.solver.addConstr(
                        self.variables["sku_flow"].sum(t, in_edges, k)
                        + last_period_inventory
                        - self.variables["sku_flow"].sum(t, out_edges_master, k)
                        - self.variables["sku_delivery"][t, node, k]
                        == self.variables["sku_inventory"][t, node, k],
                        name=constr_name,
                    )
                elif node.type == const.CUSTOMER:
                    if self.cg:
                        raise ValueError("flow in RMP do not contain customers")

                self.constrs["flow_conservation"][(t, node, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):
        for e, edge in self._iterate_no_c_edges():
            sku_list = edge.get_edge_sku_list(t, self.full_sku_list)

            constr = self.solver.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.start]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.start)
            ] = constr

            constr = self.solver.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.end]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.end)
            ] = constr

            for k in sku_list:
                constr = self.solver.addConstr(
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

        for node in self._iterate_no_c_nodes():
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            # if (
            #         node.type == const.WAREHOUSE
            #         and node.has_demand(t)
            #         and len(node.demand_sku[t]) > 0
            # ):
            #     # constr = self.model.addConstr(self.variables["open"][t, node] == 1)
            #     constr = self.solver.addConstr(self.variables["open"][t, node] == 1)
            #     self.constrs["open_relationship"]["open"][(t, node)] = constr
            # todo
            if node.type == const.CUSTOMER:
                # constr = self.model.addConstr(self.variables["open"][t, node] == 1)
                constr = self.solver.addConstr(self.variables["open"][t, node] == 1)
                self.constrs["open_relationship"]["open"][(t, node)] = constr

            for k in sku_list:
                if node.type == const.PLANT:
                    # constr = self.model.addConstr(
                    constr = self.solver.addConstr(
                        self.variables["sku_open"][t, node, k]
                        <= self.variables["open"][t, node]
                    )

                self.constrs["open_relationship"]["sku_open"][(t, node, k)] = constr

        return

    def add_constr_transportation_lb(self, t: int, verbose=False):
        # if self.arg.rmp_relaxation:
        #     return
        # else:
        for e, edge in self._iterate_no_c_edges():
            edge = self.network.edges[e]["object"]
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")
            # variable lower bound
            if edge.variable_lb < np.inf:
                self.constrs["transportation_variable_lb"][
                    (t, edge)
                ] = self.solver.addConstr(
                    flow_sum
                    >= edge.variable_lb * self.variables["select_edge"][t, edge]
                )

                self.index_for_dual_var += 1

        return

    def add_constr_transportation_capacity(self, t: int, verbose=False):
        for e, edge in self._iterate_no_c_edges():
            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")
            if edge.capacity < np.inf:
                left_capacity = edge.capacity - self.used_edge_capacity.get(t).get(
                    edge, 0
                )

                if self.bool_covering:
                    bound = self.variables["select_edge"][t, edge]
                else:
                    bound = 1

                if type(flow_sum) is not float:
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

    def add_constr_production_capacity(self, t: int):
        for node in self._iterate_no_c_nodes():
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
        for node in self._iterate_no_c_nodes():
            if node.type == const.WAREHOUSE:
                node_sum = self.variables["sku_inventory"].sum(t, node, "*")

                if node.inventory_capacity < np.inf:
                    left_capacity = (
                        node.inventory_capacity
                        - self.used_warehouse_capacity.get(t).get(node, 0)
                    )

                    bound = (
                        self.variables["open"][(t, node)] if self.bool_covering else 1.0
                    )

                    constr = self.solver.addConstr(
                        self.variables["sku_inventory"].sum(t, node, "*")
                        <= left_capacity * bound
                    )
                    self.constrs["holding_capacity"][(t, node)] = constr

                    self.dual_index_for_RMP["node_capacity"][
                        node
                    ] = self.index_for_dual_var
                    self.index_for_dual_var += 1
        return

    def add_constr_node_lb(self, t: int):
        for node in self._iterate_no_c_nodes():
            if node.type == const.PLANT:
                node_sum = self.variables["sku_production"].sum(t, node, "*")
                if node.production_lb < np.inf:
                    self.constrs["production_variable_lb"][
                        (t, node)
                    ] = self.model.addConstr(
                        node_sum
                        >= node.production_lb * self.variables["open"][t, node],
                        name=f"node_plant_lb{t, node}",
                    )
                    self.index_for_dual_var += 1
            if node.type == const.WAREHOUSE:
                node_sum = self.variables["sku_inventory"].sum(t, node, "*")
                if node.inventory_lb < np.inf:
                    self.constrs["holding_variable_lb"][
                        (t, node)
                    ] = self.solver.addConstr(
                        node_sum
                        >= node.inventory_lb * self.variables["open"][(t, node)],
                        name=f"node_warehouse_lb{t, node}",
                    )
                    self.index_for_dual_var += 1
        return

    def add_constr_flow_in_upper(self, t: int):
        for node in self._iterate_no_c_nodes():
            if node.type == const.WAREHOUSE:
                in_edges = get_in_edges(self.network, node)
                inbound_sum = self.variables["sku_flow"].sum(t, in_edges, "*")
                self.constrs["in_upper"][(t, node)] = self.model.addConstr(
                    inbound_sum <= node.inventory_capacity * self.arg.in_upper_ratio
                )
                self.index_for_dual_var += 1
        return

    def get_original_objective(self):
        """
        get the original objective value
        """

        obj = 0.0
        for t in range(self.T):
            # obj = obj + self.cal_sku_producing_cost(t)
            obj = obj + self.cal_sku_holding_cost(t)
            obj = obj + self.cal_sku_transportation_cost(t)
            # obj = obj + self.cal_sku_unfulfilled_demand_cost(t)

        if self.bool_fixed_cost:
            obj += self.cal_fixed_node_cost()

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

        # self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)
        self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)

    def set_objective(self):
        self.obj_types = {
            "producing_cost": {"index": "(t, plant)"},
            "holding_cost": {"index": "(t, warehouse)"},
            "sku_backorder_cost": {"index": "(t, warehouse, k)"},
            "transportation_cost": {"index": "(t, edge)"},
            "unfulfilled_demand_cost": {"index": "(t, customer)"},
            "fixed_node_cost": {"index": "(t, plant / warehouse, k)"},
        }

        for obj in self.obj_types.keys():
            self.obj[obj] = dict()

        self.original_obj = self.get_original_objective()

        # self.solver.setObjective(self.original_obj, sense=self.solver_constant.MINIMIZE)
        self.solver.setObjective(self.original_obj, sense=self.solver_constant.MINIMIZE)

        return

    def cal_sku_producing_cost(self, t: int):
        producing_cost = 0.0
        for node in self._iterate_no_c_nodes():
            if node.type == const.PLANT:
                node_producing_cost = 0
                sku_list = node.get_node_sku_list(t, self.full_sku_list)
                for k in sku_list:
                    if (
                        node.production_sku_unit_cost is not None
                        and k in node.production_sku_unit_cost.index.to_list()
                    ):
                        node_producing_cost += (
                            node.production_sku_unit_cost[k]
                            * self.variables["sku_production"][t, node, k]
                        )
                    else:
                        node_producing_cost += (
                            self.arg.production_sku_unit_cost
                            * self.variables["sku_production"][t, node, k]
                        )

                producing_cost = producing_cost + node_producing_cost
        self.obj["producing_cost"][t] = producing_cost

        return producing_cost

    def cal_sku_holding_cost(self, t: int):
        holding_cost = 0.0

        for node in self._iterate_no_c_nodes():
            if node.type == const.WAREHOUSE:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)
                node_holding_cost = 0.0
                for k in sku_list:
                    if node.holding_sku_unit_cost is not None:
                        holding_sku_unit_cost = node.holding_sku_unit_cost[k]
                    else:
                        holding_sku_unit_cost = self.arg.holding_sku_unit_cost

                    node_holding_cost += (
                        holding_sku_unit_cost
                        * self.variables["sku_inventory"][t, node, k]
                    )

                holding_cost = holding_cost + node_holding_cost

        self.obj["holding_cost"][t] = holding_cost

        return holding_cost

    def cal_sku_transportation_cost(self, t: int):
        transportation_cost = 0.0

        for e, edge in self._iterate_no_c_edges():
            edge_transportation_cost = 0.0

            (
                sku_list_with_fixed_transportation_cost,
                sku_list_with_unit_transportation_cost,
            ) = edge.get_edge_sku_list_with_transportation_cost(t, self.full_sku_list)

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

        self.obj["transportation_cost"][t] = transportation_cost

        return transportation_cost

    def cal_fixed_node_cost(self):
        fixed_node_cost = 0.0

        if not self.bool_covering:
            return fixed_node_cost

        for node in self._iterate_no_c_nodes():
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
            this_node_fixed_cost = (
                0.0 if np.isnan(this_node_fixed_cost) else this_node_fixed_cost
            )
            node_fixed_node_cost = 0.0
            for t in range(self.T):
                node_fixed_node_cost += (
                    this_node_fixed_cost * self.variables["open"][(t, node)]
                )

            fixed_node_cost += node_fixed_node_cost
        self.obj["fixed_node_cost"] = fixed_node_cost

        return fixed_node_cost

    def cal_fixed_edge_cost(self):
        fixed_edge_cost = 0.0
        if not self.bool_covering:
            return fixed_edge_cost

        for e, edge in self._iterate_no_c_edges():
            edge_fixed_edge_cost = 0.0
            if self.arg.edge_cost:
                edge.transportation_fixed_cost = 10
            for t in range(self.T):
                edge_fixed_edge_cost = (
                    edge.transportation_fixed_cost
                    * self.variables["select_edge"][(t, edge)]
                )

            fixed_edge_cost = fixed_edge_cost + edge_fixed_edge_cost

        self.obj["fixed_edge_cost"][edge] = edge_fixed_edge_cost

        return fixed_edge_cost

    def create_cg_bindings(self):
        self.cg_binding_constrs = {}
        self.cg_binding_constrs_ws = {}
        self.cg_downstream = {}
        for node in self._iterate_no_c_nodes():
            if node.type != const.WAREHOUSE:
                continue
            out_edges = get_out_edges(self.network, node)
            out_edges_downstream = [
                e for e in out_edges if e.end.type == const.CUSTOMER
            ]
            self.cg_downstream[node] = out_edges_downstream
            for t in range(self.T):
                sku_list = node.get_node_sku_list(t, self.full_sku_list)
                for k in sku_list:
                    # self.cg_binding_constrs[node, k, t] = self.solver.addConstr(
                    self.cg_binding_constrs[node, k, t] = self.solver.addConstr(
                        self.variables["sku_delivery"][t, node, k] == 0,
                        name=f"cg_binding{t, node, k}",
                    )
        for c in self.customer_list:
            # self.cg_binding_constrs_ws[c] = self.solver.addConstr(
            self.cg_binding_constrs_ws[c] = self.solver.addConstr(
                self.variables["cg_temporary"][c.idx] == 0.0, name=f"cg_binding_ws{c}"
            )

        return

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        super().get_solution(data_dir, preserve_zeros)

    def fetch_dual_info(self):
        node_dual = {
            k: v.pi if v is not None else 0 for k, v in self.cg_binding_constrs.items()
        }
        # broadcast to edge
        edge_dual = {
            (ee, k, t): v
            for (node, k, t), v in node_dual.items()
            for ee in self.cg_downstream[node]
        }
        # the dual of weight sum = 1
        ws_dual = {
            k: v.pi if v is not None else 0
            for k, v in self.cg_binding_constrs_ws.items()
        }
        return edge_dual, ws_dual
