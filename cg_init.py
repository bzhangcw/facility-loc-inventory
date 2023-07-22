"""
utilities for initialize first columns in CG framework
"""
from coptpy import COPT

import dnp_model
from entity import Customer, Plant
from utils import get_in_edges, get_out_edges


def init_cols_from_dual_feas_sol(self, dual_vars):
    full_lp_relaxation = dnp_model.DNP(self.arg, self.full_network, cus_num=472)
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

    # dual_index["weights_sum"] = self.dual_index["weights_sum"]

    # for customer in self.customer_list:
    #     index = dual_index["weights_sum"][customer]
    #     lp_dual[index] = dual_vars[index]

    # return lp_dual, dual_index
    init_dual = dual_vars.copy()
    for edge in self.dual_index["transportation_capacity"].keys():
        init_dual[self.dual_index["transportation_capacity"][edge]] = lp_dual[
            dual_index["transportation_capacity"][edge]
        ]

    for node in self.dual_index["node_capacity"].keys():
        init_dual[self.dual_index["node_capacity"][node]] = lp_dual[
            dual_index["node_capacity"][node]
        ]

    init_dual_index = self.dual_index.copy()
    return init_dual, init_dual_index


def init_cols_from_primal_feas_sol(self):
    full_model = dnp_model.DNP(self.arg, self.full_network, cus_num=472)
    full_model.modeling()
    # get the LP relaxation
    # vars = model.model.getVars()
    # binary_vars_index = []
    # for v in vars:
    #     if v.getType() == COPT.BINARY:
    #         binary_vars_index.append(v.getIdx())
    #         v.setType(COPT.CONTINUOUS)
    ######################
    full_model.model.setParam("Logging", 1)
    full_model.model.setParam("RelGap", 1.3)
    full_model.solve()

    t = 0  # one time period
    init_col = {}
    # for customer in tqdm(self.customer_list, desc="Customer"):
    for customer in self.customer_list:
        init_col[customer] = self.oracles[customer].var_idx
        init_col[customer]["sku_flow_ratio"] = {}

        self.fix_val_constr[customer] = []

        # get all paths from customer to each plant node
        # all_paths = {}
        # for node in self.subgraph[customer].nodes:
        #     if self.subgraph[customer].in_degree(node) == 0:  # source node
        #         all_paths[node] = nx.all_simple_paths(
        #             self.subgraph[customer],
        #             node,
        #             customer,
        #             cutoff=20,
        #             # self.subgraph[customer].reverse(), customer, node # find all paths start from customer to source node
        #         )

        customer_in_edges = get_in_edges(self.subgraph[customer], customer)
        customer_pre_node_egdes = []
        for edge in customer_in_edges:
            customer_pre_node = edge.start
            customer_pre_node_egdes += get_out_edges(self.network, customer_pre_node)

        # for sku in tqdm(self.subgraph[customer].graph["sku_list"], desc="SKU"):
        for sku in self.subgraph[customer].graph["sku_list"]:
            total_sku_flow_sum = full_model.variables["sku_flow"].sum(
                t, customer_pre_node_egdes, sku
            )
            customer_sku_flow = full_model.variables["sku_flow"].sum(
                t, customer_in_edges, sku
            )

            if type(total_sku_flow_sum) != float:
                total_sku_flow_sum = total_sku_flow_sum.getValue()
            if type(customer_sku_flow) != float:
                customer_sku_flow = customer_sku_flow.getValue()

            sku_flow_ratio = (
                customer_sku_flow / total_sku_flow_sum if total_sku_flow_sum != 0 else 0
            )

            init_col[customer]["sku_flow_ratio"][sku] = sku_flow_ratio

            # for node in tqdm(self.subgraph[customer].nodes, desc="Nodes"):
            for node in self.subgraph[customer].nodes:
                node_sku_list = node.get_node_sku_list(
                    t, self.subgraph[customer].graph["sku_list"]
                )

                if isinstance(node, Customer):
                    if sku in node_sku_list:
                        init_col[customer]["sku_demand_slack"][
                            (t, node, sku)
                        ] = full_model.variables["sku_demand_slack"][t, node, sku].x

                        constr = self.oracles[customer].model.addConstr(
                            self.oracles[customer].variables["sku_demand_slack"][
                                t, node, sku
                            ]
                            == full_model.variables["sku_demand_slack"][t, node, sku].x
                        )

                        self.fix_val_constr[customer].append(constr)

                elif isinstance(node, Plant):
                    if sku in node_sku_list:
                        init_col[customer]["sku_production"][(t, node, sku)] = (
                            full_model.variables["sku_production"][t, node, sku].x
                            * sku_flow_ratio
                        )

                        constr = self.oracles[customer].model.addConstr(
                            self.oracles[customer].variables["sku_production"][
                                t, node, sku
                            ]
                            == full_model.variables["sku_production"][t, node, sku].x
                            * sku_flow_ratio
                        )

                        self.fix_val_constr[customer].append(constr)

                else:
                    if sku in node_sku_list:
                        init_col[customer]["sku_inventory"][(t, node, sku)] = (
                            full_model.variables["sku_inventory"][t, node, sku].x
                            * sku_flow_ratio
                        )

                        constr = self.oracles[customer].model.addConstr(
                            self.oracles[customer].variables["sku_inventory"][
                                t, node, sku
                            ]
                            == full_model.variables["sku_inventory"][t, node, sku].x
                            * sku_flow_ratio
                        )

                        self.fix_val_constr[customer].append(constr)

            # for e in tqdm(self.subgraph[customer].edges, desc="Edges"):
            for e in self.subgraph[customer].edges:
                edge = self.subgraph[customer].edges[e]["object"]
                edge_sku_list = edge.get_edge_sku_list(
                    t, self.subgraph[customer].graph["sku_list"]
                )

                if sku in edge_sku_list:
                    if isinstance(edge.end, Customer):
                        init_col[customer]["sku_flow"][
                            (t, edge, sku)
                        ] = full_model.variables["sku_flow"][t, edge, sku].x

                        constr = self.oracles[customer].model.addConstr(
                            self.oracles[customer].variables["sku_flow"][t, edge, sku]
                            == full_model.variables["sku_flow"][t, edge, sku].x
                        )

                        self.fix_val_constr[customer].append(constr)

                    else:
                        init_col[customer]["sku_flow"][(t, edge, sku)] = (
                            full_model.variables["sku_flow"][t, edge, sku].x
                            * sku_flow_ratio
                        )

                        constr = self.oracles[customer].model.addConstr(
                            self.oracles[customer].variables["sku_flow"][t, edge, sku]
                            == full_model.variables["sku_flow"][t, edge, sku].x
                            * sku_flow_ratio
                        )

                        self.fix_val_constr[customer].append(constr)

    return init_col
