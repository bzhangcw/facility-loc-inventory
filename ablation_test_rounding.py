import json
import os

# import gurobipy as gp
# import numpy as np
import pandas as pd
# from coptpy import COPT
# from gurobipy import GRB
from template_generate import *
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
    arg.backorder_sku_unit_cost = 5000
    arg.capacity_node_ratio = 1
    arg.capacity_ratio = 1
    arg.cardinality_limit = 30
    arg.distance_limit = 5000
    arg.holding_sku_unit_cost = 1
    arg.in_upper_ratio = 0.24
    arg.lb_end_ratio = 1
    arg.lb_inter_ratio = 1
    arg.node_lb_ratio = 1
    arg.unfulfill_sku_unit_cost = 5000
    arg.backorder = 0
    arg.transportation_sku_unit_cost = 1
    arg.new_data = 1
    arg.num_periods = 40

    # Data Scale
    arg.conf_label = 1
    arg.T = 1
    # 3: customer 100； 7: customer 200
    arg.pick_instance = 1

    # Termination Condition
    arg.terminate_condition = 1e-5

    # # Rounding Heuristic
    # arg.rounding_heuristic = False
    # arg.rounding_heuristic_1 = False
    # arg.rounding_heuristic_2 = False
    arg.rounding_heuristic_4= True

    # RMP Algorithm
    # arg.if_del_col = True
    # arg.del_col_alg = 4
    # arg.column_pool_len = 2

    arg.if_del_col = 1
    arg.del_col_alg = 1
    arg.del_col_freq = 3

    # MIP Algorithm
    # arg.cg_mip_recover = True
    # arg.cg_rmp_mip_iter = 10
    # arg.cg_method_mip_heuristic = 0

    utils.configuration(arg.conf_label, arg)
    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
    )
    # arg.fpath = "data/data_0inv/"
    # arg.fpath = "data/data_1219/"

    arg.fpath = "data/us_generate_202403122342/"  # easy
    # arg.fpath = "data/us_generate_202403151725/"  # hard
    # arg.fpath = "data/small_instance_generate_single_period/"
    # arg.fpath = 'data/sechina_202403130301/'
    # 测试rounding
    # arg.fpath = 'data/sechina_202403130110/'
    # 测试RMP
    # arg.fpath = 'data/sechina_202403152133/'
    # arg.fpath = 'data/sechina_202403152155/'
    # arg.fpath = 'data/us_generate_202403152049/'
    (
        sku_list,
        plant_list,
        warehouse_list,
        customer_list,
        edge_list,
        network,
        node_list,
        *_,
    ) = utils.scale(arg.pick_instance, arg.fpath, arg)
    utils.add_attr(edge_list, node_list, arg, const)
    network = construct_network(node_list, edge_list, sku_list)
    solver = arg.backend.upper()

    # print("----------NCS------------")
    # init_ray = True
    # # init_ray = False
    # num_workers = min(os.cpu_count(), 24)
    # num_cpus = min(os.cpu_count(), 24)
    # utils.logger.info(f"detecting up to {os.cpu_count()} cores")
    # utils.logger.info(f"using up to {num_cpus} cores")
    # arg.DNP = 0
    # np_cg = NCS(
    #     arg,
    #     network,
    #     customer_list,
    #     sku_list,
    #     # max_iter=arg.cg_itermax,
    #     max_iter=10,
    #     init_ray=init_ray,
    #     num_workers=num_workers,
    #     num_cpus=num_cpus,
    #     solver=solver,
    # )
    # with utils.TimerContext(0, "column generation main routine"):
    #     np_cg.run()
        # np_cg.get_solution(data_dir="facility-loc-inventory/out")
        # np_cg.watch_col_weight() 
    # print("----------DNP Model------------")
    # arg.DNP = 1
    # arg.sku_list = sku_list
    # model = DNP(arg, network)
    # model.modeling()
    # model.model.setParam("Logging", 1)
    # model.model.setParam("Threads", 8)
    # model.model.setParam("TimeLimit", 7200)
    # model.model.setParam("LpMethod", 2)
    # model.model.setParam("Crossover", 0)
    # # print(f"save mps name {dnp_mps_name}")
    # # model.model.write(dnp_mps_name)
    # model.solve()
    # # print("----------DNP Result------------")
    # # model.print_result()

    print("----------NCG------------")
    max_iter = 20
    init_primal = None
    init_dual = None  # 'dual'
    init_sweeping = True
    np_cg = NetworkColumnGeneration(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        init_primal=init_primal,
        init_dual=init_dual,
        init_sweeping=init_sweeping,
        init_ray=True,
    )

    np_cg.run()
    # np_cg.get_solution("new_sol_1/")