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
    # -----------------DNP Model-----------------#
    param = Param()
    arg = param.arg
    arg.T = 4
    arg.backorder = False
    # arg.bool_capacity = False # True
    datapath = "data/data_0401_0inv.xlsx"

    # for comparing speed
    # arg.cus_num = 50
    # cfg = dict(
    #     data_dir=datapath,
    #     sku_num=50,
    #     plant_num=30,
    #     warehouse_num=30,
    #     customer_num=arg.cus_num,
    #     one_period=False if arg.T > 1 else True,
    # )

    # for debug
    arg.cus_num = 5
    cfg = dict(
        data_dir=datapath,
        sku_num=10,
        plant_num=5,
        warehouse_num=5,
        customer_num=arg.cus_num,
        one_period=False if arg.T > 1 else True,
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
    #
    if arg.capacity == 1:
        cap = pd.read_csv("./data/random_capacity_updated.csv").set_index("id")
        for e in edge_list:
            e.capacity = cap["qty"].get(e.idx, np.inf)
    if arg.lowerbound == 1:
        cap = pd.read_csv("./data/lb_cons.csv").set_index("id")
        for e in edge_list:
            e.variable_lb = cap["lb"].get(e.idx, np.inf)

        # lb_df = pd.read_csv("./data/node_lb_V3.csv").set_index("id")
        # for n in node_list:
        #     if n.type == const.PLANT:
        #         n.production_lb = lb_df["lb"].get(n.idx, np.inf)
    network = construct_network(node_list, edge_list, sku_list)
    # model = DNP(arg, network)
    # model.modeling()
    # model.model.setParam("Logging", 1)
    # model.solve()
    # model.get_solution("New_sol/")

    # model = DNP.remote(arg, network)
    # model.modeling.remote()
    # model.model.setParam.remote("Logging", 1)
    # model.solve.remote()
    # model.get_solution.remote("New_sol/")

    # #-----------------CG Model-----------------#
    print("----------DCG Model------------")
    max_iter = 2
    init_primal = None
    init_dual = None
    init_ray = False
    # init_ray = True

    np_cg = NetworkColumnGeneration(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        bool_covering=True,
        init_primal=init_primal,
        init_dual=init_dual,
        bool_edge_lb=True,
        init_ray=init_ray,
    )
    np_cg.run()
    np_cg.get_solution("New_sol/")
