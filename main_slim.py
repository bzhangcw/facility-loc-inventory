import json
import os

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
from ncg.np_cg import *
from slim_cg.slim_cg import NetworkColumnGenerationSlim as NCS
from slim_cg.slim_rmp_model import DNPSlim

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    arg.conf_label = 1
    arg.pick_instance = 12
    arg.backorder = 0
    utils.configuration(arg.conf_label, arg)
    # arg.fpath = "data/data_random/"
    arg.fpath = "data/data_generate/"
    # arg.fpath = "data/data_1219/"
    # arg.fpath = "data/data_0inv/"
    # arg.fpath = 'data/_history_/'
    # arg.fpath = 'data/_history_/data_0401_0inv.xlsx'

    datapath = arg.fpath

    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
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
    ) = utils.scale(arg.pick_instance, datapath, arg)

    utils.add_attr(edge_list, node_list, arg, const)
    network = construct_network(node_list, edge_list, sku_list)
    arg.DNP = 0
    solver = arg.backend.upper()
    print("----------NCS------------")
    init_ray = True
    num_workers = min(os.cpu_count(), 24)
    num_cpus = min(os.cpu_count(), 24)
    utils.logger.info(f"detecting up to {os.cpu_count()} cores")
    utils.logger.info(f"using     up to {num_cpus} cores")
    np_cg = NCS(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=arg.cg_itermax,
        init_ray=init_ray,
        num_workers=num_workers,
        num_cpus=num_cpus,
        solver=solver,
    )
    with utils.TimerContext(0, "column generation main routine"):
        np_cg.run()
