import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import construct_network
from np_cg import *
from param import Param

if __name__ == "__main__":
    param = Param()
    arg = param.arg

    datapath = "data/data_0401_0inv.xlsx"
    pick_instance = 2
    if pick_instance == 1:
        cfg = dict(
            data_dir=datapath,
            sku_num=2,
            plant_num=2,
            warehouse_num=13,
            customer_num=5,
        )
    elif pick_instance == 2:
        # smallest instance causing bug
        cfg = dict(
            data_dir=datapath,
            sku_num=1,
            plant_num=1,
            warehouse_num=25,
            customer_num=15,
        )
    elif pick_instance == 3:
        cfg = dict(
            data_dir=datapath,
            sku_num=140,
            plant_num=23,
            warehouse_num=28,
            customer_num=50,
            one_period=True,
        )
    else:
        cfg = dict(data_dir=datapath, one_period=True)
    (
        sku_list,
        plant_list,
        warehouse_list,
        customer_list,
        edge_list,
        network,
        node_list,
        *_,
    ) = utils.get_data_from_cfg(cfg)
    # arg.node_cost = True
    # arg.partial_fixed = False
    if arg.capacity == 1:
        cap = pd.read_csv("data/random_capacity_updated.csv").set_index("id")
        for e in edge_list:
            # e.capacity = cap["qty"].get(e.idx, np.inf)
            # 修改点6 因为论文中uhat是inf
            e.capacity = cap["qty"].get(e.idx, 0.4e5)
    # if arg.lowerbound == 1:
    #     lb_end = pd.read_csv("data/lb_end.csv").set_index("id")
    #     for e in edge_list:
    #         if e.idx in lb_end["lb"]:
    #             e.variable_lb = lb_end["lb"].get(e.idx, 0)
    # if arg.cp_lowerbound == 1:
    #     lb_inter = pd.read_csv("data/lb_inter.csv").set_index("id")
    #     for e in edge_list:
    #         if e.idx in lb_inter["lb"]:
    #             e.variable_lb = lb_inter["lb"].get(e.idx, 0) / 10
    #             print(f"setting {e.idx} to {e.variable_lb}")
    #
    # if arg.nodelb == 1:
    #     lb_df = pd.read_csv("./data/node_lb_V3.csv").set_index("id")
    #     for n in node_list:
    #         if n.type == const.WAREHOUSE:
    #             n.inventory_lb = lb_df["lb"].get(n.idx, np.inf)
    #         if n.type == const.PLANT:
    #             n.production_lb = lb_df["lb"].get(n.idx, np.inf)

    network = construct_network(node_list, edge_list, sku_list)
    ###############################################################

    max_iter = 10
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
        bool_covering=True,
        bool_edge_lb=True,
        init_ray=False
    )

    np_cg.run()
    np_cg.get_solution("new_sol_1/")

    ###############################################################
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method

    ###############################################################
    # get the LP relaxation
    variables = model.model.getVars()
    binary_vars_index = []
    for v in variables:
        if v.getType() == COPT.BINARY:
            binary_vars_index.append(v.getIdx())
            v.setType(COPT.CONTINUOUS)
    model.solve()

    lpval = model.get_model_objval()
    #############################################################
    # starting from the LP relaxation to find a feasible MIP solution
    for idx in binary_vars_index:
        variables[idx].setType(COPT.BINARY)
    model.model.solve()
    mipval = model.get_model_objval()
    print(
        f"""
    --- summary ---------------------------
    lp relaxation: {lpval},
    mip          : {mipval},
    cg           : {np_cg.RMP_model.objval}
    """
    )