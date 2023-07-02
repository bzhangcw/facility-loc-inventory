from np_cg import *
import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import constuct_network
from param import Param

if __name__ == "__main__":
    datapath = "data/data_0401_V3.xlsx"
    cfg = dict(
        data_dir=datapath,
        sku_num=2,
        plant_num=2,
        warehouse_num=13,
        customer_num=2,
        one_period=True,
    )

    # cfg = dict(data_dir=datapath, one_period=True)

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

    # use external capacity, todo, move internals
    cap = pd.read_csv("./data/random_capacity.csv").set_index("id")
    # for e in edge_list:
    #     e.capacity = cap["qty"].get(e.idx, np.inf)
    #     e.variable_lb = cap["lb"].get(e.idx, np.inf)
    network = constuct_network(node_list, edge_list, sku_list)
    ###############################################################

    param = Param()
    arg = param.arg
    arg.T = 1
    # arg.backorder = False
    # max_iter = 15
    max_iter = 15
    init_primal = None
    init_dual = None  # 'dual'

    np_cg = NP_CG(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        open_relationship=True,
        init_primal=init_primal,
        init_dual=init_dual,
    )

    np_cg.run()
