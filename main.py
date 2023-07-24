from read_data import read_data
from network import construct_network
from dnp_model import DNP
from param import Param
import os
import utils
import pandas as pd
import numpy as np
import datetime
from np_cg import *

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    param = Param()
    arg = param.arg

    datapath = "data/data_0401_V3.xlsx"
    arg.T = 7
    # cfg = dict(
    #     data_dir=datapath,
    #     sku_num=140,
    #     plant_num=23,
    #     warehouse_num=28,
    #     customer_num=519,
    #     one_period=False,
    #     # sku_num=10,
    #     # plant_num=3,
    #     # warehouse_num=8,
    #     # customer_num=9,
    #     # one_period=True,
    # )

    cfg = dict(
        data_dir=datapath,
        sku_num=20,
        plant_num=5,
        warehouse_num=5,
        customer_num=100,
        one_period=False,
        # sku_num=10,
        # plant_num=3,
        # warehouse_num=8,
        # customer_num=9,
        # one_period=True,
    )

    (
        sku_list,
        plant_list,
        warehouse_list,
        customer_list,
        edge_list,
        network,
        node_list,
        *_,
    ) = utils.get_data_from_cfg(cfg)
    # node_list = plant_list + warehouse_list + customer_list
    cap = pd.read_csv("./data/random_capacity_updated.csv").set_index("id")
    for e in edge_list:
        e.capacity = cap["qty"].get(e.idx, np.inf)
        e.variable_lb = cap["lb"].get(e.idx, np.inf)

    lb_df = pd.read_csv("./data/node_lb_V3.csv").set_index("id")
    for n in node_list:
        if n.type == const.PLANT:
            n.production_lb = lb_df["lb"].get(n.idx, np.inf)
        # if n.type == const.WAREHOUSE:
        #     n.warehouse_lb = lb_df["lb"].get(n.idx, np.inf)
    network = construct_network(node_list, edge_list, sku_list)
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.solve()

    model.get_solution(data_dir=utils.CONF.DEFAULT_SOL_PATH)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    model.write("mps/test_7_f.mps")
