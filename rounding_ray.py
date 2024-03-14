import json
from ncg.np_cg import *
import gurobipy as gp
import numpy as np
import pandas as pd
from coptpy import COPT
from gurobipy import GRB
from config.instance_generator import *
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

    # arg.backorder_sku_unit_cost=20000000
    # arg.capacity_node_ratio=100000000000
    # arg.capacity_ratio= 100000000000
    # arg.cardinality_limit= 3000000
    # arg.distance_limit=5000000000
    # arg.holding_sku_unit_cost=1
    # arg.in_upper_ratio= 0.54
    # arg.lb_end_ratio=1e-09
    # arg.lb_inter_ratio=1e-09
    # arg.node_lb_ratio= 0.1
    # arg.unfulfill_sku_unit_cost= 50000000
    # arg.conf_label = 7
    # arg.pick_instance = 8
    # arg.backorder = 0

    arg.backorder_sku_unit_cost = 5000
    arg.capacity_node_ratio = 100
    arg.capacity_ratio = 100
    arg.cardinality_limit = 30
    arg.distance_limit = 5000
    arg.holding_sku_unit_cost = 1
    arg.in_upper_ratio = 0.24
    arg.lb_end_ratio = 0.1
    arg.lb_inter_ratio = 0.1
    arg.node_lb_ratio = 0.1
    arg.unfulfill_sku_unit_cost = 5000
    arg.conf_label = 7
    arg.pick_instance = 8
    arg.backorder = 0
    arg.transportation_sku_unit_cost = 1

    arg.template_choose = 'us'
    arg.demand_type = 1
    utils.configuration(arg.conf_label, arg)
    # arg.fpath = "data/data_random/"
    # arg.cardinality_limit = 1000
    # arg.distance_limit = 10000
    # arg.in_upper_ratio = 0.0004
    # arg.capacity_ratio = 100000000000
    # arg.capacity_node_ratio = 100000000000
    # arg.lb_end_ratio = 0.000000001
    # arg.lb_inter_ratio = 0.000000001
    # arg.cardinality_limit = 3000000
    # arg.distance_limit = 5000000000
    # arg.node_lb_ratio = 100

    arg.fpath = "data/data_generate/"
    # arg.fpath = "data/data_1219/"
    # arg.fpath = "data/data_0inv/"
    # arg.fpath = 'data/_history_/'
    # arg.fpath = 'data/_history_/data_0401_0inv.xlsx'
    datapath = arg.fpath
    arg.pricing_relaxation = 0
    arg.T = 7
    
    arg.cg_mip_recover = 1
    # arg.pick_instance = 15
    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
    )
    # dnp_mps_name = f"allinone_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    if "history" in datapath:
        arg.new_data = 0
        dnp_mps_name = f"history_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
    else:
        arg.new_data = 1
        dnp_mps_name = f"new_guro_V2_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
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
    pickle.dump(network, open(f"data_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.pickle", 'wb'))
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
    model.get_solution("sol/")
    print("write")
    # variables = model.model.getVars()
    # print('-----------MIP------------')
    # # model.model.solve()
    
    # # for v in variables:
    # #     if v.getType() == COPT.BINARY:
    # #         v.setType(COPT.CONTINUOUS)
    # # print("-----------LP------------")
    # model.model.solve()
    # print('------Cost Information--------')
    # print('holding_cost',model.obj['holding_cost'][0].getExpr().getValue())
    # print('transportation_cost',model.obj['transportation_cost'][0].getExpr().getValue())
    # print('unfulfilled_demand_cost',model.obj['unfulfilled_demand_cost'][0].getExpr().getValue())

    # dual_value = model.model.getDuals()
    # dual_index = model.dual_index_for_RMP
    # for t, edge in tuple(dual_index["transportation_capacity"].keys()):
    #     print('capacity',t,edge,dual_value[dual_index["transportation_capacity"][(t, edge)]])

    # for t, node in tuple(dual_index["node_capacity"].keys()):
    #     if node.type == const.PLANT:
    #         print('production',t,node,dual_value[dual_index["node_capacity"][(t, node)]])
    #     elif node.type == const.WAREHOUSE:
    #         print('warehouse',t,node,dual_value[dual_index["node_capacity"][(t, node)]])

    # for t, edge in tuple(dual_index["transportation_variable_lb"].keys()):
    #     print('transportation_lb',t,edge,dual_value[dual_index["transportation_capacity"][(t, edge)]])

    # for t, node in tuple(dual_index["production_variable_lb"].keys()):
    #     print('production_variable_lb',t,edge,dual_value[dual_index["production_variable_lb"][(t, node)]])

    # for t, node in tuple(dual_index["holding_variable_lb"].keys()):
    #     print('holding_variable_lb',t,edge,dual_value[dual_index["holding_variable_lb"][(t, node)]])

    # for t, node in tuple(dual_index["in_upper"].keys()):
    #     print('in_upper',t,node,dual_value[dual_index["in_upper"][(t, node)]])

    # for t, node in tuple(dual_index["cardinality"].keys()):
    #     print("cardinality",t,node,dual_value[dual_index["cardinality"][(t, node)]])

    # for t, node in tuple(dual_index["distance"].keys()):
    #     print("distance",t,node,dual_value[dual_index["distance"][(t, node)]])

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
    # print("----------NCS------------")
    # # init_ray = arg.use_ray
    # arg.DNP = 0
    # init_ray = True
    # num_workers = 22
    # num_cpus = 22
    # np_cg = NCS(
    #     arg,
    #     network,
    #     customer_list,
    #     sku_list,
    #     max_iter=arg.cg_itermax,
    #     init_ray=init_ray,
    #     num_workers=num_workers,
    #     num_cpus=num_cpus,
    #     solver=solver,
    # )
    # with utils.TimerContext(0, "column generation main routine"):
    #     np_cg.run()
