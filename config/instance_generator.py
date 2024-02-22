import pandas as pd
from entity import *
from config.param import Param
import numpy as np
from geopy.distance import geodesic
import random
import const
from more_itertools import chunked

loc = np.array([0, 0])
import pandas as pd


def generate_instance(
        data_dir,
        sku_num=np.inf,
        plant_num=np.inf,
        warehouse_num=np.inf,
        customer_num=np.inf,
        one_period=False,
):
    if min([sku_num, plant_num, warehouse_num, customer_num]) < 0:
        raise ValueError("num should be non-negative")
    if sku_num == 0:
        raise ValueError("sku_num should be positive")
    if max([plant_num, warehouse_num, customer_num]) == 0:
        raise ValueError("at least one type of node should be positive")
    np.random.seed(10)
    plant_list = []
    warehouse_list = []
    customer_list = []
    sku_list = []
    edge_list = []
    nodes_dict = {}
    # ==================== load all data ============================
    data_demand = data_dir + 'demand_sku_time.csv'
    data_node = data_dir + 'node_info/'
    data_sku = data_dir + 'sku.csv'

    node_df = pd.read_csv(data_node + 'facility.csv')
    node_df = node_df.dropna(subset=['id'])
    plant_df = node_df.query("id.str.startswith('P')", engine="python")
    warehouse_df = node_df.query("id.str.startswith('T')", engine="python")
    customer_df = pd.read_csv(data_node + 'customer.csv')
    customer_sku_demand_df = pd.read_csv(data_demand)

    sku_df = pd.read_csv(data_sku)

    sku_num = min(sku_num, len(sku_df))
    plant_num = min(
        plant_num, len(plant_df)
    )
    warehouse_num = min(
        warehouse_num, len(warehouse_df)
    )
    customer_num = min(
        customer_num, len(customer_df)
    )

    sku_df = sku_df.iloc[:sku_num]
    plant_df = plant_df.iloc[:plant_num]
    warehouse_df = warehouse_df.iloc[:warehouse_num]
    customer_df = customer_df.iloc[:customer_num]
    node_df = pd.concat([
        plant_df, warehouse_df, customer_df
    ]).fillna(0)
    customer_sku_demand_df = customer_sku_demand_df[
        customer_sku_demand_df["id"].isin(customer_df["id"]) & customer_sku_demand_df["sku"].isin(sku_df["id"])]

    customer_sku_time_df_list = dict(
        list(
            customer_sku_demand_df.groupby(
                "id"
            )
        )
    )
    # ==================== construct sku ============================
    sku_dict = {}
    for i in list(sku_df.index):
        irow = sku_df.loc[i]
        SKUi = SKU(irow["id"])
        sku_list.append(SKUi)
        sku_dict[irow["id"]] = SKUi

    for w in warehouse_df.index:
        items = warehouse_df.loc[w].values.tolist()
        dc_id = items[0]
        dc_latitude = items[1]
        dc_longitude = items[2]
        capacity = items[3]
        open_cost = items[4]
        temp_w = {sku: items[5] for sku in sku_list}
        temp_df_w = pd.DataFrame.from_dict(temp_w, orient='index', columns=['Unit_cost']).rename_axis(
            'sku').reset_index()
        holding_cost = temp_df_w.set_index("sku")["Unit_cost"].dropna()
        loc = np.array([dc_latitude, dc_longitude])
        this_warehouse = Warehouse(
            idx=dc_id,
            location=loc,
            inventory_capacity=capacity,
            holding_sku_unit_cost=holding_cost,
            open_fixed_cost=open_cost,
        )
        warehouse_list.append(this_warehouse)
        nodes_dict[dc_id] = this_warehouse

    for c in customer_df.index:
        items = customer_df.loc[c].values.tolist()
        customer_id = items[0]
        customer_latitude = items[1]
        customer_longitude = items[2]
        if customer_id in customer_sku_time_df_list.keys():
            cst_df = customer_sku_time_df_list[customer_id].replace({"sku": sku_dict})
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
        loc = np.array([customer_latitude, customer_longitude])
        this_customer = Customer(
            idx=customer_id,
            location=loc,
            demand=demand,
            demand_sku=demand_sku
        )
        customer_list.append(this_customer)
        nodes_dict[customer_id] = this_customer

    for p in plant_df.index:
        items = plant_df.loc[p].values.tolist()
        plant_id = items[0]
        plant_latitude = items[1]
        plant_longitude = items[2]
        capacity = items[3]
        open_cost = items[4]
        temp = {sku: items[5] for sku in sku_list}
        temp_df = pd.DataFrame.from_dict(temp, orient='index', columns=['Unit_cost']).rename_axis('sku').reset_index()
        production_cost = temp_df.set_index("sku")["Unit_cost"].dropna()

        loc = np.array([plant_latitude, plant_longitude])
        this_plant = Plant(
            idx=plant_id,
            location=loc,
            production_capacity=capacity,
            production_sku_unit_cost=production_cost,
            open_fixed_cost=open_cost,
            producible_sku=sku_list
        )
        plant_list.append(this_plant)
        nodes_dict[plant_id] = this_plant

    if ('random' in data_dir) or ('generate' in data_dir):
        edges_w_c = []
        for customer in customer_list:
            if warehouse_num > 1000:
                customer_distance = {}
                for warehouse in warehouse_list:
                    distance = geodesic(warehouse.location, customer.location).kilometers
                    customer_distance[warehouse] = distance
                sorted_dict = dict(sorted(customer_distance.items(), key=lambda item: item[1]))
                new_list = list(sorted_dict.keys())[:500]
                for warehouse in new_list:
                    rand = random.randint(0, 480)
                    lengg = random.randint(0, 20)
                    for sku in sku_list[rand:rand + lengg]:
                        unit_cost = random.random() * customer_distance[warehouse] / 100000
                        edges_w_c.append(
                            {'start_id': warehouse, 'end_id': customer, 'sku': sku, 'unit_cost': unit_cost})
            else:
                # 全链接
                for warehouse in warehouse_list:
                    for sku in sku_list:
                        unit_cost = random.random() * geodesic(warehouse.location,
                                                               customer.location).kilometers / 100000
                        edges_w_c.append(
                            {'start_id': warehouse.idx, 'end_id': customer.idx, 'sku': sku.idx, 'unit_cost': unit_cost})

        edges_w_c_df = pd.DataFrame(edges_w_c)
        edges_p_w = []
        for plant in plant_list:
            if warehouse_num > 1000:
                plant_distance = {}
                for warehouse in warehouse_list:
                    distance = geodesic(warehouse.location, plant.location).kilometers
                    plant_distance[warehouse] = distance
                sorted_dict = dict(sorted(plant_distance.items(), key=lambda item: item[1]))
                new_list = list(sorted_dict.keys())[:500]
                for warehouse in new_list:
                    rand = random.randint(0, 480)
                    leng = random.randint(0, 10)
                    for sku in sku_list[rand:rand + leng]:
                        unit_cost = random.random() * plant_distance[warehouse] / 100000
                        edges_p_w.append({'start_id': plant, 'end_id': warehouse, 'sku': sku, 'unit_cost': unit_cost})
            else:
                for warehouse in warehouse_list:
                    for sku in sku_list:
                        unit_cost = random.random() * geodesic(warehouse.location, plant.location).kilometers / 100000
                        edges_p_w.append(
                            {'start_id': plant.idx, 'end_id': warehouse.idx, 'sku': sku.idx, 'unit_cost': unit_cost})
        edges_p_w_df = pd.DataFrame(edges_p_w)
        edges_t_t = []
        result = list(chunked(warehouse_list, 2))
        # rand_result = random.randint(0,2000)
        if warehouse_num > 1000:
            index = random.randint(0, 2000)
        else:
            index = 10
        for i in result[0:index]:
            count = 0
            for item in i:
                count = count + 1
            if count == 2:
                distance = geodesic(i[0].location, i[1].location).kilometers
                for sku in sku_list:
                    unit_cost = random.randint(1, 2) * distance / 1000
                    edges_t_t.append({'start_id': i[0].idx, 'end_id': i[1].idx, 'sku': sku.idx, 'unit_cost': unit_cost})
        edges_w_w_df = pd.DataFrame(edges_t_t)
        data_w_c = data_dir + 'edge_sku_info/edges_w_c.csv'
        data_w_w = data_dir + 'edge_sku_info/edges_w_w.csv'
        data_p_w = data_dir + 'edge_sku_info/edges_p_w.csv'
        edges_w_c_df.to_csv(data_w_c, index=False)
        edges_w_w_df.to_csv(data_w_w, index=False)
        edges_p_w_df.to_csv(data_p_w, index=False)
        # data_w_c = data_dir + 'edge_sku_info/edges_w_c.csv'
        # edges_w_c_df = pd.read_csv(data_w_c)
        # data_w_w = data_dir + 'edge_sku_info/edges_w_w.csv'
        # edges_w_w_df = pd.read_csv(data_w_w)
        # data_p_w = data_dir + 'edge_sku_info/edges_p_w.csv'
        # edges_p_w_df = pd.read_csv(data_p_w)
        edge_sku_df = pd.concat([edges_w_c_df, edges_w_w_df, edges_p_w_df]).dropna()
        print('capacity')
        generate_attr(data_dir)
    else:
        data_edge = data_dir + 'edge_sku_info.csv'
        edge_sku_df = pd.read_csv(data_edge)
    edge_sku_df = edge_sku_df[
        edge_sku_df["start_id"].isin(node_df["id"]) & edge_sku_df["end_id"].isin(node_df["id"]) & edge_sku_df[
            "sku"].isin(sku_df["id"])
        ]
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


def generate_attr(data_dir):
    def generate_id(df):
        df['id'] = df['start_id'] + '_' + df['end_id']
        df.drop(columns=['start_id', 'end_id', 'sku'], axis=0, inplace=True)
        df.set_index('id', inplace=True)
        duplicated_index = df.index.duplicated()
        df = df[~duplicated_index]
        df.reset_index(inplace=True)
        return df

    data_w_c = data_dir + 'edge_sku_info/edges_w_c.csv'
    edges_w_c_df = pd.read_csv(data_w_c)
    data_w_w = data_dir + 'edge_sku_info/edges_w_w.csv'
    edges_w_w_df = pd.read_csv(data_w_w)
    data_p_w = data_dir + 'edge_sku_info/edges_p_w.csv'
    edges_p_w_df = pd.read_csv(data_p_w)
    edge_sku_df = pd.concat([edges_w_c_df, edges_w_w_df, edges_p_w_df]).dropna()
    df = generate_id(edge_sku_df)
    df['qty'] = df['unit_cost'] * random.randint(10000000, 20000000)
    df.drop(columns=['unit_cost'], axis=0, inplace=True)
    capacity_dir = data_dir + 'capacity.csv'
    df.to_csv(capacity_dir, index=False)
    _w_c_df = generate_id(edges_w_c_df)
    _w_c_df.loc[:, 'lb'] = _w_c_df.loc[:, 'unit_cost'] * random.randint(100, 200)
    _w_c_df.drop(columns=['unit_cost'], axis=0, inplace=True)
    _w_c_dir = data_dir + 'lb_end.csv'
    _w_c_df.to_csv(_w_c_dir, index=False)
    _w_w_df = generate_id(edges_w_w_df)
    _w_w_df.loc[:, 'lb'] = _w_w_df.loc[:, 'unit_cost'] * random.randint(200, 400)
    _w_w_df.drop(columns=['unit_cost'], axis=0, inplace=True)
    _w_w_dir = data_dir + 'lb_inter.csv'
    _w_w_df.to_csv(_w_w_dir, index=False)
    print('Generate over')
