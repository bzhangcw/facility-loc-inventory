import numpy as np
import pandas as pd
from coptpy import COPT

import const
import utils
from slim.slim_rmp_model import DNPSlim
from slim.slim_cg import NetworkColumnGenerationSlim as NCS
from network import construct_network
from param import Param

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    conf_label = 1

    utils.configuration(conf_label, arg)
    datapath = "data/data_0401_0inv.xlsx"
    # datapath = "data/data_0401_V4.xlsx"
    pick_instance = 5
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
    # get the LP relaxation
    print("----------DCG Model------------")
    max_iter = 50
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
        bool_covering= True,
        init_primal=init_primal,
        init_dual=init_dual,
        bool_edge_lb=False,
        init_ray=init_ray,
        # num_workers=num_workers,
        # num_cpus=num_cpus,
        solver=solver,
    )
    np_cg.run()
    np_cg.get_solution("New_sol/")