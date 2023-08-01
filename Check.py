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
import gurobipy as gp
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
    arg.T = 1
    arg.backorder = False
    # arg.bool_capacity = False # True
    datapath = "data/data_0401_0inv.xlsx"
    cfg = dict(
        data_dir=datapath,
        sku_num=140,
        plant_num=23,
        warehouse_num=28,
        customer_num=arg.total_cus_num,
        one_period=True,
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
    network = construct_network(node_list, edge_list, sku_list)
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.solve()


    model.get_solution("out/")

    #-----------------CG Model-----------------#
    max_iter = 10
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
        init_dual=init_dual
    )
    np_cg.run()