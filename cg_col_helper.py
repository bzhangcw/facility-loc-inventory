# external friend functions for np_cg


import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import const
import utils

ATTR_IN_RMP = ["sku_flow_sum", "sku_production_sum", "sku_inventory_sum"]

CG_EXTRA_VERBOSITY = os.environ.get("CG_EXTRA_VERBOSITY", 0)
CG_EXTRA_DEBUGGING = os.environ.get("CG_EXTRA_DEBUGGING", 1)


def init_col_helpers(cg_object):
    """
    Initialize the column helpers to extract frequently
        needed quantities for the subproblems

    """
    utils.logger.info("generating column helpers")
    for customer in tqdm(cg_object.customer_list):
        col_helper = {attr: {} for attr in ATTR_IN_RMP}
        # saving column LinExpr
        for e in cg_object.subgraph[customer].edges:
            edge = cg_object.network.edges[e]["object"]
            if edge.capacity == np.inf:
                continue
            # can we do better?
            col_helper["sku_flow_sum"][edge] = (
                cg_object.oracles[customer].variables["sku_flow"].sum(0, edge, "*")
            )

        for node in cg_object.subgraph[customer].nodes:
            if node.type == const.PLANT:
                if node.production_capacity == np.inf:
                    continue
                col_helper["sku_production_sum"][node] = (
                    cg_object.oracles[customer]
                    .variables["sku_production"]
                    .sum(0, node, "*")
                )
            elif node.type == const.WAREHOUSE:
                # node holding capacity
                if node.inventory_capacity == np.inf:
                    continue
                col_helper["sku_inventory_sum"][node] = (
                    cg_object.oracles[customer]
                    .variables["sku_inventory"]
                    .sum(0, node, "*")
                )
        col_helper["beta"] = cg_object.oracles[customer].original_obj.getExpr()

        cg_object.columns_helpers[customer] = col_helper
        cg_object.columns[customer] = []


def eval_helper(col_helper):
    _vals = {
        attr: {k: v.getValue() for k, v in col_helper[attr].items()}
        for attr in ATTR_IN_RMP
    }
    _vals["beta"] = col_helper["beta"].getValue()
    return _vals


def query_columns(cg_object, customer):
    new_col = eval_helper(cg_object.columns_helpers[customer])

    # visualize this column
    oracle = cg_object.oracles[customer]
    if CG_EXTRA_DEBUGGING:
        flow_records = []
        for e in cg_object.network.edges:
            edge = cg_object.network.edges[e]["object"]
            for t in range(1):
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
                print(df)
    return new_col
