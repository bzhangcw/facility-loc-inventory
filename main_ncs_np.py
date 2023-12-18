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
    arg.conf_label = 8
    utils.configuration(arg.conf_label, arg)
    datapath = "data/data_0401_0inv_V2.xlsx"
    # datapath = "data/data_0401_V4.xlsx"
    arg.pick_instance = 6
    arg.rmp_relaxation = 1
    arg.pricing_relaxation = 1
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
    ###############################################################
    print("----------DCG Model------------")
    max_iter = 100
    init_primal = None
    init_dual = None
    init_ray = False
    # init_ray = True
    # num_workers = 4
    # num_cpus = 8
    solver = "COPT"
    # solver = "GUROBI"

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
    np_cg.get_solution("New_sol/")
    #####################DNP#######################################
    # solver = "COPT"
    # arg.cardinality_limit = 3
    # model = DNP(arg, network)
    # model.modeling()
    # model.model.setParam("Logging", 1)
    # model.model.setParam("Threads", 8)
    # model.model.setParam("TimeLimit", 3600)
    # model.model.solve()
    # if model.model.status == COPT.INFEASIBLE:
    #         model.model.computeIIS()
    #         model.model.write("iis/dnp.iis")

    # model = DNPSlim(arg, network,customer_list= customer_list, cg = False)
    # # 解出来会是0
    # model.modeling()
    # model.model.setParam("Logging", 1)
    # model.model.setParam("Threads", 8)
    # model.model.setParam("TimeLimit", 3600)
    # model.solve()