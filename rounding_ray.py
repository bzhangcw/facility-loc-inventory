import json

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

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    # 1-8
    # arg.conf_label = 1
    # arg.conf_label = 2
    utils.configuration(arg.conf_label, arg)
    # datapath = "data/data_0401_V4_1219_0inv.xlsx"
    # datapath = "data/data_0401_0inv.xlsx"
    # datapath = "data/data_0401_V4.xlsx"
    datapath = arg.fpath
    arg.pricing_relaxation = 0
    arg.T = 7
    arg.rmp_mip_iter = 5
    arg.check_rmp_mip = 1

    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
    )
    dnp_mps_name = f"allinone_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
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
    # model = DNP(arg, network)
    # model.modeling()
    # model.model.setParam("Logging", 1)
    # model.model.setParam("Threads", 8)
    # model.model.setParam("TimeLimit", 3600)
    # # model.model.setParam("LpMethod", 2)
    # model.model.setParam("Crossover", 0)
    # model.model.write(dnp_mps_name)

    # model.model.solve()

    # variables = model.model.getVars()
    # for v in variables:
    #     if v.getName().startswith("w"):
    #         if v.getName().split('_')[1].split(',')[2].startswith(' C'):
    #             print(v.getName(), v.x)
    # #
    # for k in model.obj.keys():
    #     for t in model.obj[k].keys():
    #         if type(model.obj[k][t]) is not float:
    #             print(k, t, model.obj[k][t].getExpr().getValue())
    # for t in range(arg.T):
    #     demand_customer = 0
    #     demand_slack = 0
    #     for customer in customer_list:
    #         demand_slack_sku = 0.0
    #         demand_sku = 0.0
    #         if arg.backorder:
    #             type = "sku_backorder"
    #         else:
    #             type = "sku_slack"
    #         for k in sku_list:
    #             # if type(model.variables[type].get((t,customer,k),0.0)) is not float:
    #                 # print(t,customer,k,model.variables["sku_demand_slack"].get((t,customer,k),0).x)
    #             demand_slack_sku += model.variables[type].get((t,customer,k),0.0).x
    #             demand_sku += customer.demand.get((t, k), 0)
    #             # print("CUSTOMER",customer,"TIME",t,"SKU",k,"DEMAND SLACK",  model.variables["sku_backorder"][t, customer, k].x, "DEMAND",customer.demand.get((t, k), 0))
    #         demand_customer += demand_sku
    #         demand_slack += demand_slack_sku
    #     print("TIME",t,"DEMAND SLACK",demand_slack,"DEMAND",demand_customer)
    #
    #
    # # for t in range(arg.T):
    # #     print(t, model.obj["backlogged_demand_cost"][t].getExpr().getValue())
    # # dnp_mps_lp_name = f"allinone_lp_{datapath.split('/')[-1].split('.')[0]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}.mps"
    # # model.model.write(dnp_mps_lp_name)
    # m = gp.read(dnp_mps_name)
    # print("----------DNP Model(MIP)------------")
    # m.optimize()
    # r = m.relax()
    # print("----------DNP Model(LP)------------")
    # r.optimize()
    # # #
    # # # # ###############################################################
    # print("----------DCS Model------------")
    max_iter = 5
    init_primal = None
    init_dual = None
    init_ray = True
    num_workers = 22
    num_cpus = 22
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
        num_workers=num_workers,
        num_cpus=num_cpus,
        solver=solver,
    )
    np_cg.run()
