import numpy as np
import pandas as pd
from coptpy import COPT
from dnp_model import DNP
import const as const
import utils as utils
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
    arg.conf_label = 9
    utils.configuration(arg.conf_label, arg)
    # datapath = "data/data_0401_V4_1219.xlsx"
    datapath = "data/data_0401_0inv.xlsx"
    # datapath = "data/data_0401_V4.xlsx"

    arg.pick_instance = 4
    arg.rmp_relaxation = 0
    arg.T = 7
    arg.distance = 0
    arg.customer_backorder = 0
    arg.nodelb = 0
    arg.pricing_relaxation = 0

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

    # # 3.825860e+05
    # np_cg.get_solution("New_sol/")
    ##################### DNP #######################################

    # solver = "COPT"
    # # arg.cardinality_limit = 3  # # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method
    # 1207234.130513193
    print("----------DNP Model------------")
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    model.model.setParam("RelGap", 1.3)
    model.model.setParam("LpMethod", 2)
    # model.model.write(dnp_mps_name)
    model.model.solve()

    ###############################################################
    print("----------DCS Model------------")
    max_iter = 200
    init_primal = None
    init_dual = None
    init_ray = False
    # init_ray = True
    # num_workers = 4
    # num_cpus = 8
    solver = "COPT"
    # solver = "GUROBI"
    # 4.010644e+05

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