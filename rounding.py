import numpy as np
import pandas as pd
from coptpy import COPT
from dnp_model import DNP
import gurobipy as gp
from gurobipy import GRB
import const
import utils
from slim_cg.slim_rmp_model import DNPSlim
from slim_cg.slim_cg import NetworkColumnGenerationSlim as NCS
from config.network import construct_network
from config.param import Param

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    # 1-8
    # arg.conf_label = 2
    arg.conf_label = 2
    utils.configuration(arg.conf_label, arg)
    # datapath = "data/data_0401_V4_1219.xlsx"
    datapath = "data/data_0401_0inv.xlsx"
    # datapath = "data/data_0401_V4.xlsx"
    arg.rmp_relaxation = 1
    # arg.pricing_relaxation = 1
    arg.pricing_relaxation = 0
    arg.backorder = 1
    arg.T = 7
    arg.rmp_mip_iter = 2
    arg.check_rmp_mip = 1
    arg.pick_instance = 5
    dnp_mps_name = f"allinone_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}.mps"
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

    solver = "COPT"
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    model.model.setParam("LpMethod", 2)
    model.model.setParam("Crossover", 0)
    model.model.write(dnp_mps_name)
    # dnp_mps_lp_name = f"allinone_lp_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}.mps"
    # model.model.write(dnp_mps_lp_name)
    m = gp.read(dnp_mps_name)
    print("----------DNP Model(MIP)------------")
    m.optimize()
    r = m.relax()
    print("----------DNP Model(LP)------------")
    r.optimize()

    # ###############################################################
    print("----------DCS Model------------")
    max_iter = 200
    init_primal = None
    init_dual = None
    init_ray = False
    # init_ray = True
    # num_workers = 4
    # num_cpus = 8
    np_cg = NCS(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        # bool_covering= True,
        init_primal=init_primal,
        init_dual=init_dual,
        init_ray=init_ray,
        # num_workers=num_workers,
        # num_cpus=num_cpus,
        solver=solver,
    )
    np_cg.run()
