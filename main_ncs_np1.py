import numpy as np
import pandas as pd
from coptpy import COPT
from dnp_model import DNP
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
    arg.conf_label = 1
    # arg.conf_label = 4
    utils.configuration(arg.conf_label, arg)
    # datapath = "data/data_0401_V4_1219.xlsx"
    datapath = "data/data_0401_0inv.xlsx"
    # datapath = "data/data_0401_V4.xlsx"
    arg.rmp_relaxation = 0
    arg.pricing_relaxation = 0
    arg.backorder = 1
    arg.T = 7
    arg.rmp_mip_iter = 20
    arg.check_rmp_mip = 1
    # 7: full scale
    arg.pick_instance = 4
    # arg.pick_instance = 7
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
    # solver = "GUROBI"
    # print("----------DNP Model------------")
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    model.model.setParam("LpMethod", 2)
    model.model.setParam("Crossover", 0)
    model.model.write(dnp_mps_name)
    dnp_mps_lp_name = f"allinone_lp_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}.mps"
    variables = model.model.getVars()
    for v in variables:
        if v.getType() == COPT.BINARY:
            v.setType(COPT.CONTINUOUS)
    model.model.write(dnp_mps_lp_name)

    # model,model.solve()

    # ###############################################################
    print("----------DCS Model------------")
    max_iter = 20
    init_primal = None
    init_dual = None
    # init_ray = False
    init_ray = True
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
    # print(np_cg.rmp_model.getObjective())
    # for i,k in np_cg.rmp_model.obj.items():
    #     print(i)
    #     cost = 0
    #     if k is not None:
    #         for j,l in k.items():
    #             cost += l.getExpr().getValue()
    #     print(cost)
