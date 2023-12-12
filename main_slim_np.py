import numpy as np
import pandas as pd
from coptpy import COPT

import const
import utils
from slim.slim_rmp_model import DNPSlim
from network import construct_network
from param import Param

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    conf_label = 1

    utils.configuration(conf_label, arg)
    # datapath = "data/data_0401_0inv.xlsx"
    datapath = "data/data_0401_V4.xlsx"
    pick_instance = 3
    (
        sku_list,
        plant_list,
        warehouse_list,
        customer_list,
        edge_list,
        network,
        node_list,
        *_,
    ) = utils.scale(pick_instance, datapath, arg)
    utils.add_attr(edge_list, node_list, arg, const)
    network = construct_network(node_list, edge_list, sku_list)
    ###############################################################

    model = DNPSlim(arg, network,customer_list= customer_list, cg = False)
    # 解出来会是0
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    model.solve()
    # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method

    ###############################################################
    # get the LP relaxation
    # variables = model.model.getVars()
    # binary_vars_index = []
    # for v in variables:
    #     if v.getType() == COPT.BINARY:
    #         binary_vars_index.append(v.getIdx())
    #         v.setType(COPT.CONTINUOUS)
    # model.solve()
    #
    # lpval = model.get_model_objval()
    # lpsol = model.get_solution
    # #############################################################
    # # starting from the LP relaxation to find a feasible MIP solution
    # for idx in binary_vars_index:
    #     variables[idx].setType(COPT.BINARY)
    # model.model.write("mm.lp")
    # model.model.solve()
    # mipval = model.get_model_objval()
    # #############################################################
    # model.get_solution(data_dir=utils.CONF.DEFAULT_SOL_PATH)
