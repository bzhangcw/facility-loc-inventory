from dnp_model import *
from solver_wrapper import GurobiWrapper, CoptWrapper
from solver_wrapper.CoptConstant import CoptConstant
from solver_wrapper.GurobiConstant import GurobiConstant

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
            bool_covering: bool = True,
            bool_capacity: bool = True,
            bool_edge_lb: bool = True,
            bool_node_lb: bool = True,
            bool_fixed_cost: bool = True,
            used_edge_capacity: dict = None,
            used_warehouse_capacity: dict = None,
            used_plant_capacity: dict = None,
            logging: int = 0,
            gap: float = 1e-4,
            threads: int = None,
            limit: int = 3600,
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

        # if env is None:
        #     self.env = cp.Envr(env_name)
        # else:
        #     self.env = env
        # self.model = self.env.createModel(model_name)
        self.env = self.solver.ENVR
        self.model = self.solver.model
        self.cg = cg

        self.model.setParam(self.solver_constant.Param.Logging, logging)
        self.model.setParam(self.solver_constant.Param.RelGap, gap)
        self.model.setParam(self.solver_constant.Param.TimeLimit, limit)
        if threads is not None:
            self.model.setParam(self.solver_constant.Param.Threads, threads)

        self.variables = dict()  # variables
        self.constrs = dict()  # constraints
        self.obj = dict()  # objective
        self.bool_fixed_cost = bool_fixed_cost
        self.bool_covering = bool_covering
        self.bool_capacity = bool_capacity
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
            # "weights_sum": dict(),
        }
        self.index_for_dual_var = 0  # void bugs of index out of range

        # for remote
        self.columns_helpers = None

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
        # lk 疑问：不知道这个init要干嘛 对RMP初始化一堆列 关于RMP这个子网络里怎么运输转运的
        # for remote
        self.init_col_helpers()

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
                #lk：没必要加这个限制
                # if edge.capacity == np.inf:
                #     continue
                # can we do better?
                # col_helper["sku_flow_sum"][t][edge] = self.variables["sku_flow"].sum(
                # col_helper["sku_flow_sum"][t][edge.idx] = self.variables[
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
                "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "w",
                "index": "(t, edge, k)",
            },
            "sku_production": {
                "lb": 0,
                "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "x",
                "index": "(t, plant, k)",
            },
            "sku_delivery": {
                "lb": 0,
                "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "z",
                "index": "(t, warehouse, k)",
            },
            "sku_inventory": {
                # "lb": -self.solver_constant.INFINITY if self.arg.backorder is True else 0,
                "lb": -self.solver_constant.INFINITY,
                "ub": self.solver_constant.INFINITY,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "I",
                "index": "(t, warehouse, k)",
            },
            "sku_demand_slack": {
                "lb": 0,
                "ub": [],  # TBD
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "s",
                "index": "(t, warehouse with demand / customer, k)",
            },
            "sku_backorder": {
                    "lb": 0,
                    # "ub": [],  # TBD
                    "ub": self.solver_constant.INFINITY,
                    "vtype": self.solver_constant.CONTINUOUS,
                    "nameprefix": "bs",
                    "index": "(t,warehouse, k)",
                },
        }

        if self.bool_covering:
            self.var_types["select_edge"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "p",
                "index": "(t, edge)",
            }
            self.var_types["sku_select_edge"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "pk",
                "index": "(t, edge, k)",
            }
            self.var_types["open"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.CONTINUOUS,
                "nameprefix": "y",
                "index": "(t, node)",
            }
            self.var_types["sku_open"] = {
                "lb": 0,
                "ub": 1,
                "vtype": self.solver_constant.CONTINUOUS,
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
                        # lk：增加了warehouse的open情况
                        if self.bool_covering:
                            idx["sku_open"].append((t, node, k))
                        # amount of sku k stored on node i at t
                        idx["sku_inventory"].append((t, node, k))
                        idx["sku_delivery"].append((t, node, k))
                        idx["sku_backorder"].append((t, node, k))
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            # print(node.demand)
                            idx["sku_demand_slack"].append((t, node, k))
                            self.var_types["sku_demand_slack"]["ub"].append(
                                node.demand[(t, k)]
                            )

                    elif node.type == const.CUSTOMER:
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            idx["sku_demand_slack"].append((t, node, k))
                            self.var_types["sku_demand_slack"]["ub"].append(
                                node.demand[t, k]
                            )

        # for initializaiton in CG
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
        # self.variables["cg_temporary"] = self.model.addVars(
        self.variables["cg_temporary"] = self.solver.addVars(
            [c.idx for c in self.customer_list], nameprefix="lbdtempo"
        )
        self.variables["column_weights"] = {}

    def add_constraints(self):
        # lk: 这部分的约束都写完了
        #lk update: 该加的约束都加一下
        #lk: bool_covering的更改
        if self.bool_capacity:
            self.constr_types = {
                "flow_conservation": {"index": "(t, node, k)"},
                "transportation_capacity": {"index": "(t, edge)"},
                # "transportation_variable_lb": {"index": "(t, edge)"},
                "production_capacity": {"index": "(t, node)"},
                # "production_variable_lb": {"index": "(t, node)"},
                "holding_capacity": {"index": "(t, node)"},
                # "holding_variable_lb": {"index": "(t, node)"},
                "holding_capacity_back_order": {"index": "(t, node)"},
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
                }
            self.constr_types["open_relationship"] = covering_constr_types
        if self.bool_edge_lb:
            self.constr_types["transportation_variable_lb"] = {"index": "(t, edge)"}
        if self.bool_node_lb:
            self.constr_types["production_variable_lb"] = {"index": "(t, node)"}
            self.constr_types["holding_variable_lb"] = {"index": "(t, node)"}
        if self.arg.add_distance:
            self.constr_types["distance"] = {"index": "(t, node)"}

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
                self.add_constr_production_capacity(t)
                self.add_constr_holding_capacity(t)
            # if self.arg.add_distance:
            #     self.add_constr_distance(t)

    def add_constr_flow_conservation(self, t: int):
        for node in self._iterate_no_c_nodes():
            in_edges = get_in_edges(self.network, node)
            out_edges = get_out_edges(self.network, node)
            out_edges_master = [e for e in out_edges if e.end.type != const.CUSTOMER]

            sku_list = node.get_node_sku_list(t, self.full_sku_list)
            for k in sku_list:
                constr_name = f"flow_conservation_{t}_{node.idx}_{k.idx}"

                if node.type == const.PLANT:
                    # constr = self.model.addConstr(
                    constr = self.solver.addConstr(
                        self.variables["sku_production"][t, node, k]
                        - self.variables["sku_flow"].sum(t, out_edges_master, k)
                        == 0,
                        name=constr_name,
                    )

                elif node.type == const.WAREHOUSE:
                    fulfilled_demand = 0
                    if node.has_demand(t, k):
                        fulfilled_demand = (
                                node.demand[t, k]
                                - self.variables["sku_demand_slack"][t, node, k]
                        )

                    last_period_inventory = 0.0

                    if t == 0:
                        if self.bool_covering:
                            if node.initial_inventory is not None:
                                # if self.open_relationship:
                                # self.model.addConstr(
                                self.solver.addConstr(
                                    self.variables["open"][self.T - 1, node] == 1
                                )
                                last_period_inventory = (
                                    node.initial_inventory[k]
                                    if k in node.initial_inventory
                                    else 0.0
                                )
                            else:
                                last_period_inventory = 0.0
                        else:
                            last_period_inventory = 0.0
                    else:
                        last_period_inventory = self.variables["sku_inventory"][
                            t - 1, node, k
                        ]
                    # last_period_inventory *= self.cus_ratio
                    if self.arg.backorder:
                        constr = self.solver.addConstr(
                            self.variables["sku_flow"].sum(t, in_edges, k)
                            + last_period_inventory
                            + self.variables["sku_backorder"][t,node, k]
                            - fulfilled_demand
                            - self.variables["sku_flow"].sum(t, out_edges_master, k)
                            - self.variables["sku_delivery"][t, node, k]
                            - self.variables["sku_backorder"].get((t - 1,node, k), 0)
                            == self.variables["sku_inventory"][t, node, k],
                            name=constr_name,
                        )
                    else:
                        constr = self.solver.addConstr(
                            self.variables["sku_flow"].sum(t, in_edges, k)
                            + last_period_inventory
                            - fulfilled_demand
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

            # constr = self.model.addConstr(
            constr = self.solver.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.start]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.start)
            ] = constr


            # constr = self.model.addConstr(
            constr = self.solver.addConstr(
                self.variables["select_edge"][t, edge]
                <= self.variables["open"][t, edge.end]
            )

            self.constrs["open_relationship"]["select_edge"][
                (t, edge, edge.end)
            ] = constr

            for k in sku_list:
                # constr = self.model.addConstr(
                constr = self.solver.addConstr(
                    self.variables["sku_select_edge"][t, edge, k]
                    <= self.variables["select_edge"][t, edge]
                )
                self.constrs["open_relationship"]["sku_select_edge"][
                    (t, edge, k)
                ] = constr

        for node in self._iterate_no_c_nodes():
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            if (
                    node.type == const.WAREHOUSE
                    and node.has_demand(t)
                    and len(node.demand_sku[t]) > 0
            ):
                # constr = self.model.addConstr(self.variables["open"][t, node] == 1)
                constr = self.solver.addConstr(self.variables["open"][t, node] == 1)
                self.constrs["open_relationship"]["open"][(t, node)] = constr
            elif node.type == const.CUSTOMER:
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

    def add_constr_transportation_capacity(self, t: int, verbose=False):

        for e, edge in self._iterate_no_c_edges():
            if self.arg.partial_fixed:
                if edge.start.type == const.PLANT and edge.end.idx == "T0015":
                    # if edge.start.type == const.PLANT and edge.end.type == const.WAREHOUSE:
                    if (
                            len(utils.get_in_edges(self.network, edge.end))
                            + len(utils.get_out_edges(self.network, edge.end))
                            <= 23
                    ):
                        # self.variables["select_edge"][t, edge] = 1
                        # self.model.addConstr(
                        self.solver.addConstr(
                            self.variables["select_edge"][t, edge] == 1
                        )
                        print("Fixed select_edge", t, edge)

            flow_sum = self.variables["sku_flow"].sum(t, edge, "*")

            # variable lower bound
            if self.bool_edge_lb and edge.variable_lb < np.inf:
                self.constrs["transportation_variable_lb"][
                    (t, edge)
                # ] = self.model.addConstr(
                ] = self.solver.addConstr(
                    flow_sum
                    >= edge.variable_lb * self.variables["select_edge"][t, edge]
                )

                self.index_for_dual_var += 1

            # capacity constraint
            if edge.capacity < np.inf:
                left_capacity = edge.capacity - self.used_edge_capacity.get(t).get(
                    edge, 0
                )
                # if self.used_edge_capacity.get(t) !={}:
                #     i = i+1
                #     print("edge",edge,"left_capacity", left_capacity,i)
                if left_capacity < 0:
                    print("t", t, "edge", edge, "left_capacity", left_capacity)
                bound = (
                    self.variables["select_edge"][t, edge]
                    if self.bool_covering
                    else 1.0
                )

                self.constrs["transportation_capacity"][
                    (t, edge)
                # ] = self.model.addConstr(
                ] = self.solver.addConstr(
                    flow_sum <= left_capacity * bound*1000,
                    name=f"edge_capacity{t, edge}",
                )
                # self.constrs["transportation_capacity"][
                #     (t, edge)
                # # ] = self.model.addConstr(
                # ] = self.solver.addConstr(
                #     flow_sum <= 1e10,
                #     name=f"edge_capacity{t, edge}",
                # )
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

            # lower bound constraint
            if self.bool_node_lb and node.production_lb < np.inf:
                self.constrs["production_variable_lb"][
                    (t, node)
                # ] = self.model.addConstr(
                ] = self.solver.addConstr(
                    node_sum >= node.production_lb * self.variables["open"][t, node],
                    name=f"node_lb{t, node}"
                )
                # index += 1

            # capacity constraint
            if node.production_capacity < np.inf:
                left_capacity = node.production_capacity - self.used_plant_capacity.get(
                    node, 0
                )
                # if self.used_plant_capacity.get(t) != {}:
                #     left_capacity = node.production_capacity - self.used_plant_capacity.get(t)[node]
                # else:
                #     left_capacity = node.production_capacity
                bound = self.variables["open"][t, node] if self.bool_covering else 1.0
                if left_capacity < 0:
                    print("t", t, "node", node, "left_capacity", left_capacity)

                # self.constrs["production_capacity"][(t, node)] = self.model.addConstr(
                self.constrs["production_capacity"][(t, node)] = self.solver.addConstr(
                    node_sum <= bound * left_capacity,
                    name=f"node_capacity{t, node}",
                )

                # self.dual_index_for_RMP["node_capacity"][node] = index
                # index += 1

                self.dual_index_for_RMP["node_capacity"][node] = self.index_for_dual_var
                self.index_for_dual_var += 1

        return

    def add_constr_holding_capacity(self, t: int):
        for node in self._iterate_no_c_nodes():
            if node.type != const.WAREHOUSE:
                continue

            node_sum = self.variables["sku_inventory"].sum(t, node, "*")

            # lower bound constraint
            if self.bool_node_lb and node.inventory_lb < np.inf:
                # self.constrs["holding_variable_lb"][(t, node)] = self.model.addConstr(
                self.constrs["holding_variable_lb"][(t, node)] = self.solver.addConstr(
                    node_sum >= node.inventory_lb * self.variables["open"][(t, node)]
                )

                self.index_for_dual_var += 1

            # capacity constraint
            if node.inventory_capacity < np.inf:
                left_capacity = (
                        node.inventory_capacity
                        - self.used_warehouse_capacity.get(t).get(node, 0)
                )
                if left_capacity < 0:
                    print("t", t, "node", node, "left_capacity", left_capacity)
                bound = self.variables["open"][(t, node)] if self.bool_covering else 1.0

                # constr = self.model.addConstr(
                constr = self.solver.addConstr(
                    self.variables["sku_inventory"].sum(t, node, "*")
                    <= left_capacity * bound
                )
                self.constrs["holding_capacity"][(t, node)] = constr

                self.dual_index_for_RMP["node_capacity"][node] = self.index_for_dual_var
                self.index_for_dual_var += 1

            if self.arg.backorder is True:
                if self.bool_covering:
                    # self.model.addConstr(
                    self.solver.addConstr(
                        self.variables["sku_inventory"].sum(t, node, "*")
                        >= -self.arg.M * self.variables["open"][t, node]
                    )
                else:
                    # self.model.addConstr(
                    self.solver.addConstr(
                        self.variables["sku_inventory"].sum(t, node, "*") >= -self.arg.M
                    )
                self.index_for_dual_var += 1

            if self.arg.add_in_upper == 1:
                in_inventory_sum = 0
                for e, edge in self._iterate_no_c_edges():
                    if edge.end == node:
                        in_inventory_sum += self.variables["sku_flow"].sum(t, edge, "*")
                # self.model.addConstr(in_inventory_sum <= node.inventory_capacity * 0.4)
                self.solver.addConstr(in_inventory_sum <= node.inventory_capacity * 0.4)
                self.index_for_dual_var += 1

        return
    def add_constr_cardinality(self, t: int):
        for node in self._iterate_no_c_nodes():
            if node.type == const.CUSTOMER:
                used_edge = 0
                for e in self.network.edges:
                    edge = self.network.edges[e]["object"]
                    if edge.end == node:
                        used_edge += self.variables["select_edge"][t, edge]
                constr = self.model.addConstr(used_edge <= 2)
                self.constrs["cardinality"][(t, node)] = constr
                self.index_for_dual_var += 1

    def add_constr_distance(self, t: int):
        for node in self._iterate_no_c_nodes():
            if node.type == const.CUSTOMER:
                used_distance = 0
                for e in self.network.edges:
                    edge = self.network.edges[e]["object"]
                    if edge.end == node:
                        used_distance += self.variables["select_edge"][t, edge]*edge.distance
                constr = self.model.addConstr(used_distance <= 1000)
                self.constrs["distance"][(t, node)] = constr
                self.index_for_dual_var += 1
    def get_original_objective(self):
        """
        get the original objective value
        """

        nf = 0.0
        ef = 0.0
        #TODO：但是目前的obj是没考虑未履约的需求的 所以init_RMP的objective要改一下
        # 仓库是固定的持货成本
        hc = sum(self.cal_sku_holding_cost(t) for t in range(self.T))
        pc = sum(self.cal_sku_producing_cost(t) for t in range(self.T))
        tc = sum(self.cal_sku_transportation_cost(t) for t in range(self.T))
        ud = sum(self.cal_sku_unfulfill_demand_cost(t) for t in range(self.T))

        # @note: cannot be computed in the master model (RMP)
        # ud = ud + self.cal_sku_unfulfill_demand_cost(t)

        if self.bool_fixed_cost:
            nf = self.cal_fixed_node_cost()
            ef = self.cal_fixed_edge_cost()


        obj = hc + pc + tc + nf + ef
        return obj, hc, pc, tc, nf, ef

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

        # self.model.setObjective(obj, sense=self.solver_constant.MINIMIZE)
        self.solver.setObjective(obj, sense=self.solver_constant.MINIMIZE)

    def set_objective(self):
        self.obj_types = {
            "sku_producing_cost": {"index": "(t, plant, k)"},
            "sku_holding_cost": {"index": "(t, warehouse, k)"},
            "sku_backorder_cost": {"index": "(t, warehouse, k)"},
            "sku_transportation_cost": {"index": "(t, edge)"},
            "unfulfill_demand_cost": {
                "index": "(t, warehouse with demand / customer, k)"
            },
            "fixed_node_cost": {"index": "(t, plant / warehouse, k)"},
            "fixed_edge_cost": {"index": "(t, edge, k)"},
            "end_inventory_cost": {"index": "(node, k)"},
        }

        for obj in self.obj_types.keys():
            self.obj[obj] = dict()
        # 去除了ud
        (
            self.original_obj,
            self.hc,
            self.pc,
            self.tc,
            self.nf,
            self.ef,
        ) = self.get_original_objective()

        # self.model.setObjective(self.original_obj, sense=self.solver_constant.MINIMIZE)
        self.solver.setObjective(self.original_obj, sense=self.solver_constant.MINIMIZE)

        return

    def cal_sku_producing_cost(self, t: int):
        producing_cost = 0.0

        for node in self._iterate_no_c_nodes():
            if node.type == const.PLANT:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    node_sku_producing_cost = 0.0

                    if (
                            node.production_sku_fixed_cost is not None
                            and self.bool_covering
                    ):
                        node_sku_producing_cost = (
                                node_sku_producing_cost
                                + node.production_sku_fixed_cost[k]
                                * self.variables["sku_open"][t, node, k]
                        )
                    if node.production_sku_unit_cost is not None:
                        node_sku_producing_cost = (
                                node_sku_producing_cost
                                + node.production_sku_unit_cost[k]
                                * self.variables["sku_production"][t, node, k]
                        )

                    producing_cost = producing_cost + node_sku_producing_cost

                    self.obj["sku_producing_cost"][
                        (t, node, k)
                    ] = node_sku_producing_cost

        return producing_cost

    def cal_sku_holding_cost(self, t: int):
        holding_cost = 0.0

        for node in self._iterate_no_c_nodes():
            if node.type == const.WAREHOUSE:
                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    node_sku_holding_cost = 0.0

                    if node.holding_sku_unit_cost is not None:
                        holding_sku_unit_cost = node.holding_sku_unit_cost[k]
                    else:
                        holding_sku_unit_cost = self.arg.holding_sku_unit_cost

                    # I_hat = max(I, 0)
                    I_hat = self.model.addVar(name=f"I_hat_({t},_{node},_{k})")
                    # lk：不是很懂啊 这个约束加上去干嘛 也不好计算dual
                    self.model.addConstr(
                        I_hat >= self.variables["sku_inventory"][t, node, k]
                    )
                    # lk：把Ihat改成了sku_inventory
                    # node_sku_holding_cost = (
                    #         node_sku_holding_cost + holding_sku_unit_cost * self.variables["sku_inventory"][t, node, k]
                    # )
                    node_sku_holding_cost = (
                            node_sku_holding_cost + holding_sku_unit_cost *  I_hat
                    )

                    holding_cost = holding_cost + node_sku_holding_cost

                    self.obj["sku_holding_cost"][(t, node, k)] = node_sku_holding_cost

        return holding_cost

    def cal_sku_transportation_cost(self, t: int):
        transportation_cost = 0.0

        for e, edge in self._iterate_no_c_edges():
            # edge = self.network.edges[e]["object"]

            edge_transportation_cost = 0.0

            (
                sku_list_with_fixed_transportation_cost,
                sku_list_with_unit_transportation_cost,
            ) = edge.get_edge_sku_list_with_transportation_cost(t, self.full_sku_list)

            if self.bool_covering:
                for k in sku_list_with_fixed_transportation_cost:
                    if (
                            edge.transportation_sku_fixed_cost is not None
                            and k in edge.transportation_sku_fixed_cost
                    ):
                        edge_transportation_cost = (
                                edge_transportation_cost
                                + edge.transportation_sku_fixed_cost[k]
                                * self.variables["sku_select_edge"][t, edge, k]
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
                        * self.variables["sku_flow"][t, edge, k]
                )

            transportation_cost = transportation_cost + edge_transportation_cost

            self.obj["sku_transportation_cost"][(t, edge)] = edge_transportation_cost

        return transportation_cost

        # 咱就是说这部分应该也没有 因为unfulfill cost 也是customer的

    def cal_fixed_node_cost(self):
        # 计算fixed_cost的部分要重新写一下（lk）
        fixed_node_cost = 0.0
        if not self.bool_covering:
            return fixed_node_cost

        for node in self._iterate_no_c_nodes():
            node_fixed_node_cost = 0.0
            if self.arg.node_cost:
                if node.type == const.PLANT:
                    # this_node_fixed_cost = node.production_fixed_cost
                    this_node_fixed_cost = 100
                elif node.type == const.WAREHOUSE:
                    # this_node_fixed_cost = node.holding_fixed_cost
                    this_node_fixed_cost = 500
                else:
                    continue
            else:
                if node.type == const.PLANT:
                    this_node_fixed_cost = node.production_fixed_cost
                elif node.type == const.WAREHOUSE:
                    this_node_fixed_cost = node.holding_fixed_cost
                else:
                    continue
            # 更改：这块改成每期都计算fixed cost了
            # y = self.model.addVar(vtype=COPT.BINARY, name=f"y_{node}")
            for t in range(self.T):
                # self.model.addConstr(self.variables["open"][(t, node)] <= y)

                node_fixed_node_cost = this_node_fixed_cost * self.variables["open"][(t, node)]

            fixed_node_cost = fixed_node_cost + node_fixed_node_cost

            self.obj["fixed_node_cost"][node] = node_fixed_node_cost

        return fixed_node_cost
    # def cal_fixed_node_cost(self):
    #     # 加完了
    #     return 0.0

    # def cal_fixed_edge_cost(self):
    #     # 加完了
    #     return 0.0

    def cal_fixed_edge_cost(self):
        fixed_edge_cost = 0.0
        if not self.bool_covering:
            return fixed_edge_cost

        for e, edge in self._iterate_no_c_edges():
            edge_fixed_edge_cost = 0.0
           # 更改lk： 同样的改成每一期的了
           #  p = self.model.addVar(vtype=COPT.BINARY, name=f"p_{edge}")
           #
           #  for t in range(self.T):
           #      self.model.addConstr(self.variables["select_edge"][(t, edge)] <= p)

            # edge_fixed_edge_cost = edge.transportation_fixed_cost * p
            if self.arg.edge_cost:
                edge.transportation_fixed_cost = 10
            for t in range(self.T):
                edge_fixed_edge_cost = edge.transportation_fixed_cost * self.variables["select_edge"][(t, edge)]

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
            out_edges_downstream = [e for e in out_edges if e.end.type == const.CUSTOMER]
            self.cg_downstream[node] = out_edges_downstream
            for t in range(self.T):
                sku_list = node.get_node_sku_list(t, self.full_sku_list)
                for k in sku_list:
                    # self.cg_binding_constrs[node, k, t] = self.model.addConstr(
                    self.cg_binding_constrs[node, k, t] = self.solver.addConstr(
                        self.variables["sku_delivery"][t, node, k] == 0, name=f"cg_binding{t, node, k}"
                    )
        for c in self.customer_list:
            # self.cg_binding_constrs_ws[c] = self.model.addConstr(
            self.cg_binding_constrs_ws[c] = self.solver.addConstr(
                self.variables["cg_temporary"][c.idx] == 0.0, name=f"cg_binding_ws{c}"
            )

    def cal_sku_unfulfill_demand_cost(self, t: int):
        unfulfill_demand_cost = 0.0

        for node in self._iterate_no_c_nodes():
            if node.type == const.WAREHOUSE:
                if node.has_demand(t):
                    for k in node.get_node_sku_list(t, self.full_sku_list):

                        if node.unfulfill_sku_unit_cost is not None:
                            unfulfill_sku_unit_cost = node.unfulfill_sku_unit_cost[
                                (t, k)
                            ]
                        else:
                            unfulfill_sku_unit_cost = 50000

                        unfulfill_node_sku_cost = (
                            unfulfill_sku_unit_cost
                            * self.variables["sku_backorder"][(t, node, k)]
                        )

                        unfulfill_demand_cost = (
                            unfulfill_demand_cost + unfulfill_node_sku_cost
                        )

                        self.obj["unfulfill_demand_cost"][
                            (t, node, k)
                        ] = unfulfill_node_sku_cost

        return unfulfill_demand_cost

    def get_solution(self, data_dir: str = "./", preserve_zeros: bool = False):
        super().get_solution(data_dir, preserve_zeros)

    def fetch_dual_info(self):
        # TODO：RMP的dual要都重新写一下或者check一下不知道这种分成三类的对不对'

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
            k: v.pi if v is not None else 0 for k, v in self.cg_binding_constrs_ws.items()
        }
        return edge_dual, ws_dual


if __name__ == "__main__":
    import datetime

    import pandas as pd

    import utils
    from network import construct_network
    from param import Param

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