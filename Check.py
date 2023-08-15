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
from np_cg import *
import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import construct_network
from param import Param


if __name__ == "__main__":
    # -----------------DNP Model-----------------#
    param = Param()
    arg = param.arg
    arg.T = 4
    arg.cus_num = 100
    arg.backorder = False
    # arg.bool_capacity = False # True
    datapath = "data/data_0401_0inv.xlsx"
    cfg = dict(
        data_dir=datapath,
        sku_num=140,
        plant_num=23,
        warehouse_num=28,
        # sku_num=50,
        # plant_num=10,
        # warehouse_num=10,
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
        cap = pd.read_csv("./data/random_capacity_updated_2.csv").set_index("id")
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
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.solve()
    # model.Test_cost()
    model.get_solution("New_sol/")

    # #-----------------CG Model-----------------#
    print("----------DCG Model------------")
    max_iter = 100
    init_primal = None
    init_dual = None

    np_cg = NetworkColumnGeneration(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        bool_covering=True,
        init_primal=init_primal,
        init_dual=init_dual,
        bool_edge_lb = True,
    )
    np_cg.run()
    np_cg.get_solution("New_sol/")
