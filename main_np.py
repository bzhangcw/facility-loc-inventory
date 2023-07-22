import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import construct_network
from param import Param
from coptpy import COPT

if __name__ == "__main__":
    param = Param()
    arg = param.arg

    datapath = "data/data_0401_V3.xlsx"
    pick_instance = 3
    if pick_instance == 1:
        cfg = dict(
            data_dir=datapath,
            sku_num=2,
            plant_num=2,
            warehouse_num=13,
            customer_num=5,
            one_period=True,
        )
    elif pick_instance == 2:
        # smallest instance causing bug
        cfg = dict(
            data_dir=datapath,
            sku_num=1,
            plant_num=1,
            warehouse_num=25,
            customer_num=3,
            one_period=True,
        )
    elif pick_instance == 3:
        cfg = dict(
            data_dir=datapath,
            sku_num=140,
            plant_num=23,
            warehouse_num=28,
            customer_num=100,
            one_period=True,
        )
    else:
        cfg = dict(data_dir=datapath, one_period=True)
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
    # # use external capacity, todo, move internals
    cap = pd.read_csv("./data/random_capacity.csv").set_index("id")
    for e in edge_list:
        # e.bool_capacity = cap["qty"].get(e.idx, np.inf)
        e.capacity = cap["qty"].get(e.idx, np.inf)
        # e.variable_lb = cap["lb"].get(e.idx, np.inf)
        # pass
    network = construct_network(node_list, edge_list, sku_list)
    ###############################################################

    model = DNP(arg, network)
    model.modeling()
    # get the LP relaxation
    # variables = model.model.getVars()
    # binary_vars_index = []
    # for v in variables:
    #     if v.getType() == COPT.BINARY:
    #         binary_vars_index.append(v.getIdx())
    #         v.setType(COPT.CONTINUOUS)
    ######################
    model.model.setParam("Logging", 1)
    # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method

    model.solve()

    # starting from the LP relaxation to find a feasible MIP solution
    # for idx in binary_vars_index:
    #     vars[idx].setType(COPT.BINARY)
    # # model.model.setObjective(0.0, sense=COPT.MINIMIZE)
    # model.model.solve()
    #############################################################

    model.get_solution(data_dir=utils.CONF.DEFAULT_SOL_PATH)
