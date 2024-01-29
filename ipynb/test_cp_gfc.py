import numpy as np
import pandas as pd

import utils as utils
from config.network import construct_network
from config.param import Param
from dnp_model import DNP
from ncg.np_cg import *

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    arg.T = 7
    arg.partial_fixed = True
    datapath = "/Users/xue/github/facility-loc-inventory/data/data_0401_0inv.xlsx"
    pick_instance = 1
    if pick_instance == 1:
        cfg = dict(
            data_dir=datapath,
            sku_num=2,
            plant_num=2,
            warehouse_num=13,
            customer_num=5,
            one_period=True if arg.T == 1 else False,
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
            plant_num=23,
            warehouse_num=28,
            customer_num=100,
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
    ) = config.utils.get_data_from_cfg(cfg)

    for e in edge_list:
        e.variable_lb = 0

    if arg.capacity == 1:
        cap = pd.read_csv(
            "/Users/xue/github/facility-loc-inventory/data/random_capacity_updated.csv"
        ).set_index("id")
        for e in edge_list:
            # e.capacity = cap["qty"].get(e.idx, np.inf)
            # 修改点6 因为论文中uhat是inf
            e.capacity = cap["qty"].get(e.idx, 0.4e5)
    if arg.lowerbound == 1:
        lb_end = pd.read_csv(
            "/Users/xue/github/facility-loc-inventory/data/lb_end.csv"
        ).set_index("id")
        for e in edge_list:
            if e.idx in lb_end["lb"]:
                e.variable_lb = lb_end["lb"].get(e.idx, 0)
    if arg.lowerbound == 1:
        lb_inter = pd.read_csv(
            "/Users/xue/github/facility-loc-inventory/data/lb_inter.csv"
        ).set_index("id")
        for e in edge_list:
            if e.idx in lb_inter["lb"]:
                e.variable_lb = lb_inter["lb"].get(e.idx, 0)
                print(f"setting {e.idx} to {e.variable_lb}")

    network = construct_network(node_list, edge_list, sku_list)
    ###############################################################

    max_iter = 10
    init_primal = None
    init_dual = None  # 'dual'
    init_sweeping = True

    # np_cg = NetworkColumnGeneration(
    #     arg,
    #     network,
    #     customer_list,
    #     sku_list,
    #     max_iter=max_iter,
    #     init_primal=init_primal,
    #     init_dual=init_dual,
    #     init_sweeping=init_sweeping,
    #     bool_covering=True,
    #     bool_edge_lb=True,
    # )
    # # CG跑的结果是 7.372590e+03
    # np_cg.run()
    # os.makedirs(f"sol_mip_{pick_instance}_DCG_NL/", exist_ok=True)
    # np_cg.get_solution(f"sol_mip_{pick_instance}_DCG_NL/")

    ###############################################################

    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method
    # 把DNP的model写出来
    model.model.write("mm.lp")
    # DNP的最优解是 7373.645201525
    model.model.solve()
    mipval = model.get_model_objval()
    # 在这ipynb下面新建了一个文件夹来存储输出数据
    os.makedirs(f"sol_mip_{pick_instance}_DNP/", exist_ok=True)
    model.get_solution(f"sol_mip_{pick_instance}_DNP/")
