from np_cg import *
import pandas as pd

import utils
from dnp_model import DNP
from network import construct_network
from param import Param
from entity import Warehouse, Edge
if __name__ == "__main__":
    param = Param()
    arg = param.arg
    arg.T = 7
    arg.capacity = 1
    arg.bool_distance = 1
    arg.node_cost, arg.edge_cost = 1,1
    arg.nodelb = 0
    # arg.cus_num = 100
    # arg.lowerbound, arg.cp_lowerbound, arg.add_in_upper, arg.node_cost, arg.edge_cost, arg.nodelb, arg.add_cardinality = 1, 1, 1, 1, 1, 0, 1
    datapath = "data/data_0401_0inv_V2.xlsx"
    pick_instance = 2
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
            e.capacity = cap["qty"].get(e.idx, 0.4e5)
    if arg.lowerbound == 1:
        lb_end = pd.read_csv("data/lb_end.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_end["lb"]:
                e.variable_lb = lb_end["lb"].get(e.idx, 0)
    # if arg.cp_lowerbound == 1:
    if arg.lowerbound == 1:
        lb_inter = pd.read_csv("data/lb_inter.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_inter["lb"]:
                e.variable_lb = lb_inter["lb"].get(e.idx, 0)/10
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

    max_iter = 10
    init_primal = None
    init_dual = None  # 'dual'
    init_sweeping = True
    # arg.add_in_upper = 1
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method
    # model.model.write("mm.lp")
    # C: Capacity L:Lowerbound CL:coupling lowerbound
    # NU: 每日仓库入库上限
    # Fn: node cost Fe: edge cost NL: node lowerbound
    # Ca: 每期履约消费者的仓库个数上限 FR: fulfill rate
    if arg.lowerbound ==0 and arg.add_in_upper == 0:
        type1 = 0
    elif arg.lowerbound ==1 and arg.add_in_upper == 1:
        type1 = 3
    elif arg.lowerbound ==1:
        type1 = 1
    elif arg.add_in_upper == 1:
        type1 = 2

    if arg.add_cardinality == 0 and arg.bool_distance == 0:
        type2 = 0
    elif arg.lowerbound==1 and arg.bool_distance == 1:
        type2 = 3
    elif arg.lowerbound==1:
        type1 = 1
    elif arg.bool_distance == 1:
        type1 = 2
    model.model.write("1123/C{}_O{}_E{}.mps".format(arg.cus_num, type1, type2))
    # model.model.solve()
    # model.get_solution(f"sol_mip_{pick_instance}_DNP/")