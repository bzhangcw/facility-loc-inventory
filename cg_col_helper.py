# external friend functions for np_cg
import os

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

import const
import utils

ATTR_IN_RMP = ["sku_flow_sum", "sku_production_sum", "sku_inventory_sum"]
# macro for debugging
CG_EXTRA_VERBOSITY = int(os.environ.get("CG_EXTRA_VERBOSITY", 0))
CG_EXTRA_DEBUGGING = int(os.environ.get("CG_EXTRA_DEBUGGING", 1))


def init_col_helpers(cg_object):
    """
    Initialize the column helpers to extract frequently
        needed quantities for the subproblems

    """
    utils.logger.info("generating column helpers")

    # for t in range(cg_object.oracles[0].T):
    for customer in tqdm(cg_object.customer_list):
        # col_helper = {attr: {t: {} for t in range(cg_object.oracles[customer].T)} for attr in ATTR_IN_RMP}
        col_helper = {
            attr: {
                # t: {} for t in range(ray.get(cg_object.oracles[customer].getT.remote()))
                t: {}
                for t in range(cg_object.arg.T)
            }
            for attr in ATTR_IN_RMP
        }

        # saving column LinExpr
        # for t in range(cg_object.oracles[customer].T):
        # for t in range(ray.get(cg_object.oracles[customer].getT.remote())):
        for t in range(cg_object.arg.T):
            for e in cg_object.subgraph[customer].edges:
                edge = cg_object.network.edges[e]["object"]
                if edge.capacity == np.inf:
                    continue
                # can we do better?
                col_helper["sku_flow_sum"][t][edge] = (
                    # cg_object.oracles[customer].variables["sku_flow"].sum(t, edge, "*")
                    # ray.get(cg_object.oracles[customer].getvariables.remote())[
                    ray.get(cg_object.oracles[customer].getvariables.remote())[
                        "sku_flow"
                    ].sum(t, edge, "*")
                )

            for node in cg_object.subgraph[customer].nodes:
                if node.type == const.PLANT:
                    if node.production_capacity == np.inf:
                        continue
                    col_helper["sku_production_sum"][t][node] = (
                        cg_object.oracles[customer]
                        .variables["sku_production"]
                        .sum(t, node, "*")
                    )
                elif node.type == const.WAREHOUSE:
                    # node holding capacity
                    if node.inventory_capacity == np.inf:
                        continue
                    col_helper["sku_inventory_sum"][t][node] = (
                        cg_object.oracles[customer]
                        .variables["sku_inventory"]
                        .sum(t, node, "*")
                    )

        col_helper["beta"] = cg_object.oracles[customer].original_obj.getExpr()

        cg_object.columns_helpers[customer] = col_helper
        cg_object.columns[customer] = []


def eval_helper(col_helper, T):
    _vals = {
        attr: {
            t: {
                k: v.getValue() if type(v) is not float else 0
                for k, v in col_helper[attr][t].items()
            }
            for t in range(T)
        }
        for attr in ATTR_IN_RMP
    }
    # for t in range(T):
    #     for attr in ATTR_IN_RMP:
    #         if col_helper[attr][t] != {}:
    #             for k, v in col_helper[attr][t].items():
    #                 if type(v) is not float:
    #                     if v.getValue() > 0:
    #                         _vals[attr][t][k] = v.getValue()
    # _vals[attr][t][k] = v.getValue()
    # _vals = {
    #     t: {attr: {k: v.getValue() for k, v in col_helper[attr][t].items()}
    #     for attr in ATTR_IN_RMP} for t in range(7)}
    _vals["beta"] = col_helper["beta"].getValue()
    return _vals


def query_columns(cg_object, customer):
    new_col = eval_helper(cg_object.columns_helpers[customer], cg_object.arg.T)

    # visualize this column
    oracle = cg_object.oracles[customer]
    if CG_EXTRA_DEBUGGING:
        flow_records = []
        for e in cg_object.network.edges:
            edge = cg_object.network.edges[e]["object"]
            for t in range(cg_object.oracles[customer].T):
                edge_sku_list = edge.get_edge_sku_list(t, cg_object.full_sku_list)
                for k in edge_sku_list:
                    try:
                        if oracle.variables["sku_flow"][(t, edge, k)].x != 0:
                            flow_records.append(
                                {
                                    "c": customer.idx,
                                    "start": edge.start.idx,
                                    "end": edge.end.idx,
                                    "sku": k.idx,
                                    "t": t,
                                    "qty": oracle.variables["sku_flow"][(t, edge, k)].x,
                                }
                            )
                    except:
                        pass

            new_col["records"] = flow_records
            if CG_EXTRA_VERBOSITY:
                df = pd.DataFrame.from_records(flow_records).set_index(["c", "col_id"])
    return new_col
