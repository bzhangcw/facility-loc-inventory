"""
utilities for initialize first columns in CG framework
"""
from coptpy import COPT

from dnp_model import DNP
from entity import Customer, Plant
from utils import get_in_edges, get_out_edges


def init_cols_from_primal_feas_sol(_cg_instance):
    """
    _cg_instance: an object in np_cg
    """
    arg, full_network, customer_list, oracles, fix_val_constr, subgraph, network = (
        _cg_instance.arg,
        _cg_instance.full_network,
        _cg_instance.customer_list,
        _cg_instance.oracles,
        _cg_instance.fix_val_constr,
        _cg_instance.subgraph,
        _cg_instance.network,
    )
    full_model = DNP(arg, full_network, cus_num=472)
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
    # for customer in tqdm(_cg_instance.customer_list, desc="Customer"):
    for customer in customer_list:
        init_col[customer] = oracles[customer].var_idx
        init_col[customer]["sku_flow_ratio"] = {}

        fix_val_constr[customer] = []

        # get all paths from customer to each plant node
        # all_paths = {}
        # for node in _cg_instance.subgraph[customer].nodes:
        #     if _cg_instance.subgraph[customer].in_degree(node) == 0:  # source node
        #         all_paths[node] = nx.all_simple_paths(
        #             _cg_instance.subgraph[customer],
        #             node,
        #             customer,
        #             cutoff=20,
        #             # _cg_instance.subgraph[customer].reverse(), customer, node # find all paths start from customer to source node
        #         )

        customer_in_edges = get_in_edges(subgraph[customer], customer)
        customer_pre_node_egdes = []
        for edge in customer_in_edges:
            customer_pre_node = edge.start
            customer_pre_node_egdes += get_out_edges(network, customer_pre_node)

        # for sku in tqdm(_cg_instance.subgraph[customer].graph["sku_list"], desc="SKU"):
        for sku in subgraph[customer].graph["sku_list"]:
            total_sku_flow_sum = full_model.vars["sku_flow"].sum(
                t, customer_pre_node_egdes, sku
            )
            customer_sku_flow = full_model.vars["sku_flow"].sum(
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

            # for node in tqdm(_cg_instance.subgraph[customer].nodes, desc="Nodes"):
            for node in subgraph[customer].nodes:
                node_sku_list = node.get_node_sku_list(
                    t, subgraph[customer].graph["sku_list"]
                )

                if isinstance(node, Customer):
                    if sku in node_sku_list:
                        init_col[customer]["sku_demand_slack"][
                            (t, node, sku)
                        ] = full_model.vars["sku_demand_slack"][t, node, sku].x

                        constr = oracles[customer].model.addConstr(
                            oracles[customer].vars["sku_demand_slack"][t, node, sku]
                            == full_model.vars["sku_demand_slack"][t, node, sku].x
                        )

                        fix_val_constr[customer].append(constr)

                elif isinstance(node, Plant):
                    if sku in node_sku_list:
                        init_col[customer]["sku_production"][(t, node, sku)] = (
                                full_model.vars["sku_production"][t, node, sku].x
                                * sku_flow_ratio
                        )

                        constr = oracles[customer].model.addConstr(
                            oracles[customer].vars["sku_production"][t, node, sku]
                            == full_model.vars["sku_production"][t, node, sku].x
                            * sku_flow_ratio
                        )

                        fix_val_constr[customer].append(constr)

                else:
                    if sku in node_sku_list:
                        init_col[customer]["sku_inventory"][(t, node, sku)] = (
                                full_model.vars["sku_inventory"][t, node, sku].x
                                * sku_flow_ratio
                        )

                        constr = oracles[customer].model.addConstr(
                            oracles[customer].vars["sku_inventory"][t, node, sku]
                            == full_model.vars["sku_inventory"][t, node, sku].x
                            * sku_flow_ratio
                        )

                        fix_val_constr[customer].append(constr)

            # for e in tqdm(_cg_instance.subgraph[customer].edges, desc="Edges"):
            for e in subgraph[customer].edges:
                edge = subgraph[customer].edges[e]["object"]
                edge_sku_list = edge.get_edge_sku_list(
                    t, subgraph[customer].graph["sku_list"]
                )

                if sku in edge_sku_list:
                    if isinstance(edge.end, Customer):
                        init_col[customer]["sku_flow"][
                            (t, edge, sku)
                        ] = full_model.vars["sku_flow"][t, edge, sku].x

                        constr = oracles[customer].model.addConstr(
                            oracles[customer].vars["sku_flow"][t, edge, sku]
                            == full_model.vars["sku_flow"][t, edge, sku].x
                        )

                        fix_val_constr[customer].append(constr)

                    else:
                        init_col[customer]["sku_flow"][(t, edge, sku)] = (
                                full_model.vars["sku_flow"][t, edge, sku].x * sku_flow_ratio
                        )

                        constr = oracles[customer].model.addConstr(
                            oracles[customer].vars["sku_flow"][t, edge, sku]
                            == full_model.vars["sku_flow"][t, edge, sku].x
                            * sku_flow_ratio
                        )

                        fix_val_constr[customer].append(constr)

    num_infeas_col = 0
    for customer in _cg_instance.customer_list:
        ##### check if the columns generated by full feasible solution is feasible #####
        _cg_instance.oracles[customer].model.reset()
        _cg_instance.oracles[customer].model.solve()
        if _cg_instance.oracles[customer].model.status == COPT.INFEASIBLE:
            num_infeas_col += 1
            print(num_infeas_col, ": ", customer.idx, " column is infeasible")

        # remove the constraints for the fixed variables
        for constr in _cg_instance.fix_val_constr[customer]:
            constr.remove()

        init_col = {
            "beta": 0,
            "sku_flow_sum": {},
            "sku_production_sum": {},
            "sku_inventory_sum": {},
        }
        _cg_instance.columns[customer] = [init_col]
        ###############################################################################
    _cg_instance.init_RMP()
