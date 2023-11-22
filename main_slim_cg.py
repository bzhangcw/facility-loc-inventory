import numpy as np
import pandas as pd
from coptpy import COPT

import const
import utils
from slim.slim_rmp_model import DNPSlim
from slim.slim_cg import NetworkColumnGenerationSlim as NCS
from network import construct_network
from param import Param

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    # # 工厂是100 仓库是500
    # arg.node_cost = 0
    # # 边是10
    # arg.edge_cost = 0
    arg.capacity = 1
    # arg.lowerbound = 1
    datapath = "data/data_0401_0inv.xlsx"
    pick_instance = 3
    if pick_instance == 1:
        cfg = dict(
            data_dir=datapath,
            sku_num=2,
            plant_num=2,
            warehouse_num=5,
            customer_num=5,
            one_period=arg.T == 1,
        )
    elif pick_instance == 2:
        # smallest instance causing bug
        cfg = dict(
            data_dir=datapath,
            sku_num=100,
            plant_num=20,
            warehouse_num=20,
            customer_num=100,
            one_period=True,
        )
    elif pick_instance == 3:
        cfg = dict(
            data_dir=datapath,
            sku_num=140,
            plant_num=28,
            warehouse_num=23,
            customer_num=arg.cus_num,
            one_period=False,
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

    for e in edge_list:
        e.variable_lb = 0

    # arg.node_cost = True
    # arg.partial_fixed = False
    if arg.capacity == 1:
        cap = pd.read_csv("data/random_capacity_updated.csv").set_index("id")
        for e in edge_list:
            # e.capacity = cap["qty"].get(e.idx, np.inf)
            # 修改点6 因为论文中uhat是inf
            e.capacity = cap["qty"].get(e.idx, 0.4e5)
    if arg.lowerbound == 1:
        lb_end = pd.read_csv("data/lb_end.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_end["lb"]:
                e.variable_lb = lb_end["lb"].get(e.idx, 0)
    if arg.cp_lowerbound == 1:
        lb_inter = pd.read_csv("data/lb_inter.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_inter["lb"]:
                e.variable_lb = lb_inter["lb"].get(e.idx, 0) / 10
                print(f"setting {e.idx} to {e.variable_lb}")

    if arg.nodelb == 1:
        lb_df = pd.read_csv("./data/node_lb_V3.csv").set_index("id")
        for n in node_list:
            if n.type == const.WAREHOUSE:
                n.inventory_lb = lb_df["lb"].get(n.idx, np.inf)
            if n.type == const.PLANT:
                n.production_lb = lb_df["lb"].get(n.idx, np.inf)
    network = construct_network(node_list, edge_list, sku_list)

    ###############################################################
    # get the LP relaxation
    print("----------DCG Model------------")
    # max_iter = 1000
    max_iter = 50

    init_primal = None
    init_dual = None
    init_ray = False
    num_workers = 10
    num_cpus = 36

    np_cg = NCS(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        bool_covering=True,
        init_primal=init_primal,
        init_dual=init_dual,
        bool_edge_lb=True,
        init_ray=init_ray,
        num_workers=num_workers,
        num_cpus=num_cpus,
    )
    np_cg.run()
    np_cg.get_solution("new_sol_1/")
