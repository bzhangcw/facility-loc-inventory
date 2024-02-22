import json
from ncg.np_cg import *
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
from config.instance_generator import generate_instance

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    arg.conf_label = 3
    arg.pick_instance = 5
    arg.backorder = 0
    utils.configuration(arg.conf_label, arg)
    # arg.fpath = "data/data_random/"
    # arg.fpath = "data/data_1219/"
    # arg.fpath = "data/data_0inv/"
    # arg.fpath = 'data/_history_/'
    arg.fpath = 'data/_history_/data_0401_0inv.xlsx'
    datapath = arg.fpath
    arg.new_data = 0
    arg.pricing_relaxation = 0
    arg.T = 7
    arg.cg_mip_recover = 1
    # arg.backorder = 0
    # 随机生成的新数据从pick_instance = 12 开始 全规模的是15
    # arg.pick_instance = 15
    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
    )
    # dnp_mps_name = f"allinone_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    if arg.new_data:
        dnp_mps_name = f"new_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    else:
        dnp_mps_name = f"history_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
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
    print("----------DNP Model------------")
    arg.DNP = 1
    arg.sku_list = sku_list
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 7200)
    model.model.setParam("LpMethod", 2)
    model.model.setParam("Crossover", 0)
    print(f"save mps name {dnp_mps_name}")
    model.model.write(dnp_mps_name)
    variables = model.model.getVars()
    print('-----------MIP------------')
    model.model.solve()
    # # print('------Cost Information--------')
    # # print('holding_cost',model.obj['holding_cost'][0].getExpr().getValue())
    # # print('transportation_cost',model.obj['transportation_cost'][0].getExpr().getValue())
    # # print('unfulfilled_demand_cost',model.obj['unfulfilled_demand_cost'][0].getExpr().getValue())
    # for v in variables:
    #     if v.getType() == COPT.BINARY:
    #         v.setType(COPT.CONTINUOUS)
    # print('-----------LP------------')
    # model.model.solve()
    # model.get_solution('sol/')
    # print('write')

    # print("----------NCG------------")
    # arg.DNP = 0
    # max_iter = 100
    # init_primal = None
    # init_dual = None  # 'dual'
    # init_sweeping = True
    # np_cg = NetworkColumnGeneration(
    #     arg,
    #     network,
    #     customer_list,
    #     sku_list,
    #     max_iter=max_iter,
    #     init_primal=init_primal,
    #     init_dual=init_dual,
    #     init_sweeping=init_sweeping,
    #     init_ray=False,
    # )

    # np_cg.run()
    # np_cg.get_solution("new_sol_1/")
    print("----------NCS------------")
    # init_ray = arg.use_ray
    arg.DNP = 0
    init_ray = True
    num_workers = 22
    num_cpus = 22
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
