import numpy as np
import pandas as pd

import utils as utils
from dnp_model import DNP
from config.network import construct_network
from ncg.np_cg import *
from config.param import Param
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

if __name__ == "__main__":
    param = Param()
    arg = param.arg

    datapath = "data/data_0401_V4_1219.xlsx"
    arg.conf_label = 9
    arg.distance = 0
    utils.configuration(arg.conf_label, arg)
    # datapath = "data/data_0401_V4.xlsx"
    arg.pick_instance = 7
    arg.customer_backorder = 0
    arg.T = 432
    # arg.rmp_relaxation = 1
    # arg.pricing_relaxation = 1
    (
        sku_list,
        plant_list,
        warehouse_list,
        customer_list,
        edge_list,
        network,
        node_list,
        *_,
    ) = utils.scale(arg.pick_instance, datapath, arg)
    utils.add_attr(edge_list, node_list, arg, const)
    ###############################################################
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    # model.model.setParam("TimeLimit", 3600)
    # model.model.setParam("RelGap", 1.3)
    model.model.setParam("LpMethod", 2)  # interior point method

    ###############################################################
    # get the LP relaxation
    variables = model.model.getVars()
    binary_vars_index = []
    for v in variables:
        if v.getType() == COPT.BINARY:
            binary_vars_index.append(v.getIdx())
            v.setType(COPT.CONTINUOUS)
    # model.write("model.mps")
    model.solve()

    # lpval = model.get_model_objval()
    #############################################################
    # starting from the LP relaxation to find a feasible MIP solution
    # for idx in binary_vars_index:
    #     variables[idx].setType(COPT.BINARY)
    # model.model.solve()
    # mipval = model.get_model_objval()
    # print(
    #     f"""
    # --- summary ---------------------------
    # lp relaxation: {lpval},
    # mip          : {mipval},
    # cg           : {np_cg.RMP_model.objval}
    # """
    # )
