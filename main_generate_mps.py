import gurobipy as gp
import numpy as np
import pandas as pd
from coptpy import COPT
from gurobipy import GRB

import const
import utils
from config.network import construct_network
from config.param import Param
from dnp_model import DNP
from slim_cg.slim_cg import NetworkColumnGenerationSlim as NCS
from slim_cg.slim_rmp_model import DNPSlim

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    # arg.conf_label = 3
    # arg.pick_instance = 5
    # arg.backorder = 0
    utils.configuration(arg.conf_label, arg)
    # arg.fpath = "data/data_random/"
    # arg.fpath = "data/data_1219/"
    # arg.fpath = "data/data_0inv/"
    # arg.fpath = 'data/_history_/'
    # arg.fpath = 'data/_history_/data_0401_0inv.xlsx'
    datapath = arg.fpath
    if "history" in datapath:
        arg.new_data = 0
        dnp_mps_name = f"history_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    else:
        arg.new_data = 1
        dnp_mps_name = f"new_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    print(f"save mps name {dnp_mps_name}")
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
    network = construct_network(node_list, edge_list, sku_list)

    ##################### DNP #######################################
    with utils.TimerContext(1, "COPT generates a MILP model"):
        solver = "COPT"
        model = DNP(arg, network)
        model.modeling()
        model.model.setParam("Logging", 1)
        model.model.setParam("Threads", 8)
        model.model.setParam("TimeLimit", 3600)
        model.model.setParam("Crossover", 0)

    with utils.TimerContext(1, "COPT saves a MILP model"):
        model.model.write(dnp_mps_name)
    #
    # with utils.TimerContext(0, "Gurobi generates a MILP model"):
    #     solver = "Gurobi"
    #     model = DNP(arg, network)
    #     model.modeling()
    #     model.model.setParam("Logging", 1)
    #     model.model.setParam("Threads", 8)
    #     model.model.setParam("TimeLimit", 3600)
    #     model.model.setParam("Crossover", 0)

    utils.visualize_timers()
