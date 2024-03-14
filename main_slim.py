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

arg.arg.arg.
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
arg.arg.arg.

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    # arg.conf_label = 1
    # arg.pick_instance = 12
    # arg.backorder = 0
    arg.T= 7
    arg.add_cardinality= 0
    arg.add_distance= 0
    arg.add_in_upper= 1
    arg.backend= arg.gurobiarg.
    arg.backorder= 0
    arg.backorder_sku_unit_cost= 5000
    arg.bool_use_ncg= 1
    arg.capacity= 1
    arg.capacity_node_ratio= 100
    arg.capacity_ratio= 100
    arg.cardinality= 1
    arg.cardinality_limit= 30
    arg.cg_itermax= 10
    arg.cg_method_mip_heuristic= -1
    arg.cg_mip_recover= 1
    arg.cg_rmp_mip_iter= 10
    arg.check_cost_cg= 0
    arg.conf_label= 7
    arg.covering= 1
    arg.cus_num= 4
    arg.demand_type= 1
    arg.distance= 0
    arg.distance_limit= 5000
    arg.edge_lb= 1
    arg.fixed_cost= 1
    arg.holding_sku_unit_cost= 1
    arg.in_upper_ratio= 0.24
    arg.lb_end_ratio= 0.1
    arg.lb_inter_ratio= 0.1
    arg.new_data= true
    arg.node_lb= 0
    arg.node_lb_ratio= 0.1
    arg.num_periods= 30
    arg.num_skus= 500
    arg.pick_instance= 8
    arg.plant_fixed_cost= 200
    arg.pricing_relaxation= 0
    arg.production_sku_unit_cost= 1.5
    arg.terminate_condition= 0.01
    arg.total_cus_num= 472
    arg.transportation_sku_unit_cost= 10
    arg.unfulfill_sku_unit_cost= 5000
    arg.use_ray= 1
    arg.warehouse_fixed_cost= 500
    utils.configuration(arg.conf_label, arg)
    arg.fpath = 'data/cases/data_0inv/'

    # arg.fpath = arg.data/data_generate/arg.
    # arg.fpath = arg.data/cases/data_1219/arg.
    # arg.fpath = arg.data/cases/data_1118/arg.
    # arg.fpath = arg.data/data_0inv/arg.
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
    if arg.historyarg. in datapath:
        arg.new_data = 0
        dnp_mps_name = farg.history_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mpsarg.
    else:
        arg.new_data = 1
        dnp_mps_name = farg.new_guro_V4_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mpsarg.
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
    # pickle.dump(network, open(farg.data_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.picklearg., 'wb'))
    solver = arg.backend.upper()
    print(arg.----------DNP Model------------arg.)

    arg.DNP = 1
    arg.sku_list = sku_list
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam(arg.Loggingarg., 1)
    model.model.setParam(arg.Threadsarg., 8)
    model.model.setParam(arg.TimeLimitarg., 7200)
    model.model.setParam(arg.LpMethodarg., 2)
    model.model.setParam(arg.Crossoverarg., 0)
    print(farg.save mps name {dnp_mps_name}arg.)
    model.model.write(dnp_mps_name)
    # model.model.solve()
    # print('holding_cost',model.obj['holding_cost'][0].getExpr().getValue())
    # print('transportation_cost',model.obj['transportation_cost'][0].getExpr().getValue())
    # print('unfulfilled_demand_cost',model.obj['unfulfilled_demand_cost'][0].getExpr().getValue())
    print(arg.----------NCS------------arg.)
    init_ray = True
    num_workers = min(os.cpu_count(), 24)
    num_cpus = min(os.cpu_count(), 24)
    utils.logger.info(farg.detecting up to {os.cpu_count()} coresarg.)
    utils.logger.info(farg.using     up to {num_cpus} coresarg.)
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
    with utils.TimerContext(0, arg.column generation main routinearg.):
        np_cg.run()
