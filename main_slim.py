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
    # arg.conf_label = 1
    # arg.pick_instance = 12
    # arg.backorder = 0
    arg.backorder_sku_unit_cost=5000
    arg.capacity_node_ratio=100
    arg.capacity_ratio= 100
    arg.cardinality_limit= 30
    arg.distance_limit=5000
    arg.holding_sku_unit_cost=1
    arg.in_upper_ratio= 0.24
    arg.lb_end_ratio=0.1
    arg.lb_inter_ratio=0.1
    arg.node_lb_ratio= 0.1
    arg.unfulfill_sku_unit_cost= 5000
    arg.conf_label = 1
    arg.pick_instance = 2
    arg.backorder = 0
    arg.terminate_condition = 0.01
    arg.transportation_sku_unit_cost = 10
    arg.T = 7
    utils.configuration(arg.conf_label, arg)
    arg.fpath = 'data/cases/data_0inv/'

    # arg.fpath = "data/data_generate/"
    # arg.fpath = "data/cases/data_1219/"
    # arg.fpath = "data/cases/data_1118/"
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
    if "history" in datapath:
        arg.new_data = 0
        dnp_mps_name = f"history_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    else:
        arg.new_data = 1
        dnp_mps_name = f"new_guro_V4_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
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
    # pickle.dump(network, open(f"data_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.pickle", 'wb'))
    solver = arg.backend.upper()
    # print("----------DNP Model------------")
    #
    # arg.DNP = 1
    # arg.sku_list = sku_list
    # model = DNP(arg, network)
    # model.modeling()
    # model.model.setParam("Logging", 1)
    # model.model.setParam("Threads", 8)
    # model.model.setParam("TimeLimit", 7200)
    # model.model.setParam("LpMethod", 2)
    # model.model.setParam("Crossover", 0)
    # print(f"save mps name {dnp_mps_name}")
    # model.model.write(dnp_mps_name)
    # model.model.solve()
    # print('holding_cost',model.obj['holding_cost'][0].getExpr().getValue())
    # print('transportation_cost',model.obj['transportation_cost'][0].getExpr().getValue())
    # print('unfulfilled_demand_cost',model.obj['unfulfilled_demand_cost'][0].getExpr().getValue())
    print("----------NCS------------")
    init_ray = True
    num_workers = min(os.cpu_count(), 24)
    num_cpus = min(os.cpu_count(), 24)
    utils.logger.info(f"detecting up to {os.cpu_count()} cores")
    utils.logger.info(f"using     up to {num_cpus} cores")
    arg.DNP = 0
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
