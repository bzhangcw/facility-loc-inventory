import numpy as np
import pandas as pd

from config.param import Param
from entity import *

loc = np.array([0, 0])


def read_data(
    data_dir,
    sku_num=np.inf,
    plant_num=np.inf,
    warehouse_num=np.inf,
    customer_num=np.inf,
    one_period=False,
):
    """
    > The function reads the data from the excel file and constructs the corresponding objects

    :param data_dir: the directory of the data file
    :param sku_num: the number of SKUs to be used in the model
    :param plant_num: the number of plants to be used in the model
    :param warehouse_num: the number of warehouses to be used in the model
    :param customer_num: the number of customers
    """

    if min([sku_num, plant_num, warehouse_num, customer_num]) < 0:
        raise ValueError("num should be non-negative")
    if sku_num == 0:
        raise ValueError("sku_num should be positive")
    if max([plant_num, warehouse_num, customer_num]) == 0:
        raise ValueError("at least one type of node should be positive")

    sku_list = []
    plant_list = []
    warehouse_list = []
    customer_list = []
    edge_list = []
    nodes_dict = {}
    # data_dir = data_dir + "data_0401_0inv.xlsx"
    # ==================== load all data ============================
    sku_df = pd.read_excel(data_dir, sheet_name="0-sku")
    node_df = pd.read_excel(data_dir, sheet_name="1-node")
    edge_df = pd.read_excel(data_dir, sheet_name="2-edge")
    node_sku_df = pd.read_excel(data_dir, sheet_name="3-node-sku-info")
    edge_sku_df = pd.read_excel(data_dir, sheet_name="4-edge-sku")
    # node_time_df = pd.read_excel(data_dir, sheet_name='5-node-time')
    # edge_time_df = pd.read_excel(data_dir, sheet_name='6-edge-time')
    node_sku_time_df = pd.read_excel(data_dir, sheet_name="7-node-sku-time")
    # edge_sku_time_df = pd.read_excel(data_dir, sheet_name='8-edge-sku-time')

    sku_df.fillna(0)
    node_df.fillna(0)
    edge_df.fillna(0)
    node_sku_df.fillna(0)
    edge_sku_df.fillna(0)
    node_sku_time_df.fillna(0)

    # ==================== control data size ============================
    sku_num = min(sku_num, len(sku_df))
    plant_num = min(
        plant_num, len(node_df.query("id.str.startswith('P')", engine="python"))
    )
    warehouse_num = min(
        warehouse_num, len(node_df.query("id.str.startswith('T')", engine="python"))
    )
    customer_num = min(
        customer_num, len(node_df.query("id.str.startswith('C')", engine="python"))
    )

    # TODO: add other ways to choose nodes and skus
    sku_df = sku_df.iloc[:sku_num]
    node_df = pd.concat(
        [
            node_df.query("id.str.startswith('P')", engine="python").iloc[:plant_num],
            node_df.query("id.str.startswith('T')", engine="python").iloc[
                :warehouse_num
            ],
            node_df.query("id.str.startswith('C')", engine="python").iloc[
                :customer_num
            ],
        ]
    )

    # TODO: can do better to drop nodes and skus
    edge_df = edge_df[
        edge_df["from"].isin(node_df["id"]) & edge_df["to"].isin(node_df["id"])
    ]  # empty
    node_sku_df = node_sku_df[
        node_sku_df["id"].isin(node_df["id"]) & node_sku_df["sku"].isin(sku_df["id"])
    ]
    edge_sku_df = edge_sku_df[
        edge_sku_df["start_id"].isin(node_df["id"])
        & edge_sku_df["end_id"].isin(node_df["id"])
        & edge_sku_df["sku"].isin(sku_df["id"])
    ]
    node_sku_time_df = node_sku_time_df[
        node_sku_time_df["id"].isin(node_df["id"])
        & node_sku_time_df["sku"].isin(sku_df["id"])
    ]
    # ==================== construct sku ============================
    sku_dict = {}
    for i in list(sku_df.index):
        irow = sku_df.loc[i]
        SKUi = SKU(irow["id"], irow["value"])
        sku_list.append(SKUi)
        sku_dict[irow["id"]] = SKUi

    # ==================== construct plant ============================
    # plant_df = node_df.query("id.str.startswith('P')", engine="python")[
    #     ["id", "total_capacity"]
    # ]
    plant_df = node_df.query("id.str.startswith('P')", engine="python")[
        ["x", "y", "id", "total_capacity"]
    ]
    plant_sku_df_dict = dict(
        list(node_sku_df.query("id.str.startswith('P')", engine="python").groupby("id"))
    )

    for p in plant_df.index:
        items = plant_df.loc[p].values.tolist()
        # plant_id = items[0]
        # capacity = items[1]
        plant_x = items[0]
        plant_y = items[1]
        plant_id = items[2]
        capacity = items[3]

        if plant_id in plant_sku_df_dict:
            plant_sku_df = plant_sku_df_dict[plant_id].replace({"sku": sku_dict})
            producible_sku = plant_sku_df["sku"].drop_duplicates().tolist()
            sku_rate = plant_sku_df[["sku", "rate"]].set_index("sku")["rate"].dropna()
            sku_unit_cost = (
                plant_sku_df[["sku", "unit_cost"]]
                .set_index("sku")["unit_cost"]
                .dropna()
            )
        else:
            producible_sku = None
            sku_rate = None
            sku_unit_cost = None
        loc = np.array([plant_x, plant_y])
        this_plant = Plant(
            idx=plant_id,
            location=loc,
            production_capacity=capacity / 1,
            producible_sku=producible_sku,
            production_sku_rate=sku_rate,
            production_sku_unit_cost=sku_unit_cost,
        )
        if (producible_sku is not None and len(producible_sku) > 0) or not one_period:
            plant_list.append(this_plant)
            nodes_dict[plant_id] = this_plant
    # ==================== construct warehouse ============================
    # warehouse_df = node_df.query("id.str.startswith('T')", engine="python")[
    #     ["id", "total_capacity", "fixed_cost", "if_current"]
    # ]
    warehouse_df = node_df.query("id.str.startswith('T')", engine="python")[
        ["x", "y", "id", "total_capacity", "fixed_cost", "if_current"]
    ]
    warehouse_sku_df_dict = dict(
        list(node_sku_df.query("id.str.startswith('T')", engine="python").groupby("id"))
    )
    warehouse_sku_time_df_dict = dict(
        list(
            node_sku_time_df.query("id.str.startswith('T')", engine="python").groupby(
                "id"
            )
        )
    )

    for w in warehouse_df.index:
        items = warehouse_df.loc[w].values.tolist()
        ws_x = items[0]
        ws_y = items[1]
        # ws_id = items[0]
        # capacity = items[1]
        # fixed_cost = items[2] if not pd.isna(items[2]) else 0
        # if_current = bool(items[3])
        ws_id = items[2]
        capacity = items[3]
        fixed_cost = items[4] if not (pd.isna(items[2]) or pd.isna(items[4])) else 0
        # if fixed_cost.isna():
        #     print("NAN", ws_id,fixed_cost)
        if_current = bool(items[5])

        if ws_id in warehouse_sku_df_dict.keys():
            ws_df = warehouse_sku_df_dict[ws_id].replace({"sku": sku_dict})
            begin_inventory = (
                ws_df[["sku", "int_inv"]].set_index("sku")["int_inv"].dropna()
            )
            end_inventory = (
                ws_df[["sku", "end_inv"]].set_index("sku")["end_inv"].dropna()
            )
        else:
            begin_inventory = None
            end_inventory = None

        if ws_id in warehouse_sku_time_df_dict.keys():
            wst_df = warehouse_sku_time_df_dict[ws_id].replace({"sku": sku_dict})
            demand = (
                wst_df[["time", "sku", "demand"]]
                .set_index(["time", "sku"])["demand"]
                .dropna()
            )
            demand_sku = {}
            for t in list(wst_df.groupby("time")):
                demand_sku[t[0]] = t[1]["sku"].tolist()
            demand_sku = pd.Series(demand_sku)
        else:
            demand = None
            demand_sku = None

        loc = np.array([ws_x, ws_y])
        this_warehouse = Warehouse(
            idx=ws_id,
            location=loc,
            inventory_capacity=capacity / 1,
            holding_fixed_cost=fixed_cost,
            if_current=if_current,
            initial_inventory=begin_inventory,
            end_inventory=end_inventory,
            demand=demand,
            demand_sku=demand_sku,
        )
        warehouse_list.append(this_warehouse)
        nodes_dict[ws_id] = this_warehouse
    # ==================== construct customer ============================
    customer_df = node_df.query("id.str.startswith('C')", engine="python")[
        ["x", "y", "id"]
    ]
    customer_sku_time_df_list = dict(
        list(
            node_sku_time_df.query("id.str.startswith('C')", engine="python").groupby(
                "id"
            )
        )
    )

    for c in customer_df.index:
        items = customer_df.loc[c].values.tolist()
        cus_x = items[0]
        cus_y = items[1]
        cst_id = items[2]
        # cst_id = customer_df.loc[c]

        if cst_id in customer_sku_time_df_list.keys():
            cst_df = customer_sku_time_df_list[cst_id].replace({"sku": sku_dict})
            demand = (
                cst_df[["time", "sku", "demand"]]
                .set_index(["time", "sku"])["demand"]
                .dropna()
            )
            demand.sort_index(level="time")

            demand_sku = {}
            for t in list(cst_df.groupby("time")):
                demand_sku[t[0]] = t[1]["sku"].tolist()
            demand_sku = pd.Series(demand_sku)
        else:
            demand = None
            demand_sku = None
        loc = np.array([cus_x, cus_y])
        this_customer = Customer(
            idx=cst_id, location=loc, demand=demand, demand_sku=demand_sku
        )
        if (demand_sku is not None and 0 in demand_sku.index) or not one_period:
            customer_list.append(this_customer)
            nodes_dict[cst_id] = this_customer
    # ==================== construct edge ============================
    edge_sku_df_list = list(edge_sku_df.groupby(["start_id", "end_id"]))

    for edge in edge_sku_df_list:
        if edge[0][0] not in nodes_dict.keys() or edge[0][1] not in nodes_dict.keys():
            continue

        start = nodes_dict[edge[0][0]]
        end = nodes_dict[edge[0][1]]
        edge_id = edge[0][0] + "_" + edge[0][1]

        edge_df = edge[1].replace({"sku": sku_dict})
        unit_cost = edge_df[["sku", "unit_cost"]].set_index("sku")["unit_cost"].dropna()
        capacity = np.inf

        this_edge = Edge(
            idx=edge_id,
            start=start,
            end=end,
            capacity=capacity,
            variable_lb=np.inf,
            transportation_sku_unit_cost=unit_cost,
        )
        edge_list.append(this_edge)
    return sku_list, plant_list, warehouse_list, customer_list, edge_list


if __name__ == "__main__":
    datapath = "./data_0401_V3.xlsx"

    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=datapath, sku_num=5, plant_num=5, warehouse_num=5, customer_num=5
    )
