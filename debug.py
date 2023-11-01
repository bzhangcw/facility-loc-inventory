from np_cg import *
import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import construct_network
from param import Param

if __name__ == "__main__":
    param = Param()
    arg = param.arg

    datapath = "/Users/xue/github/facility-loc-inventory/data/data_0401_0inv.xlsx"
    pick_instance = 2
    if pick_instance == 1:
        cfg = dict(
            data_dir=datapath,
            sku_num=2,
            plant_num=2,
            warehouse_num=13,
            customer_num=5,
            one_period=True,
        )
    elif pick_instance == 2:
        # smallest instance causing bug
        cfg = dict(
            data_dir=datapath,
            sku_num=1,
            plant_num=1,
            warehouse_num=25,
            customer_num=3,
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
    ) = utils.get_data_from_cfg(cfg)

    for e in edge_list:
        e.variable_lb = 0

    if arg.capacity == 1:
        cap = pd.read_csv("/Users/xue/github/facility-loc-inventory/data/random_capacity_updated.csv").set_index("id")
        for e in edge_list:
            # e.capacity = cap["qty"].get(e.idx, np.inf)
            # 修改点6 因为论文中uhat是inf
            e.capacity = cap["qty"].get(e.idx, 0.4e5)
    if arg.lowerbound == 1:
        lb_end = pd.read_csv("/Users/xue/github/facility-loc-inventory/data/lb_end.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_end["lb"]:
                e.variable_lb = lb_end["lb"].get(e.idx, 0)
    if arg.lowerbound == 1:
        lb_inter = pd.read_csv("/Users/xue/github/facility-loc-inventory/data/lb_inter.csv").set_index("id")
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
    )
    # CG跑的结果是 7.372590e+03
    np_cg.run()
    os.makedirs(f"sol_mip_{pick_instance}_DCG_NL/", exist_ok=True)
    np_cg.get_solution(f"sol_mip_{pick_instance}_DCG_NL/")

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

for k, v in model.variables['select_edge'].items():
    #k(0,edge)
    if k[1].end.type != "C":
        v.setType(COPT.CONTINUOUS)
        print(f"relaxing {k}")

model.solve()

lpval = model.get_model_objval()
os.makedirs(f"sol_lp_{pick_instance}/", exist_ok=True)
lpval = model.model.objval
# 注意这里面的cg是没考虑edge的lowerbound的（无论是在pricing里面还是在RMP里面）
# 解更大可能是因为松弛后有的是0.3 正常来讲就是0了 但是现在额外加了新的约束 导致搜索空间变小 所以目标函数变大
print(
    f"""
--- summary ---------------------------
lp relaxation: {lpval},
mip          : {mipval},
cg           : {np_cg.RMP_model.objval}
"""
)
from tqdm.notebook import tqdm as tqdm_notebook

import utils
from cp_gfc import *
model.model.setParam("Logging", 0)
# get the first solutions
# 重写了下get_solution及存储的表结构
df_edges = model.get_solution(f"sol_lp_{pick_instance}/")
lpval = model.model.objval
k = 0
while k < 20:
    for _, row in query_fractional_inter_edges(df_edges).iterrows():
        start, end, t = row["obj_start"], row["obj_end"], row["t"]
        break
    # # 修改点1
    for e in utils.get_out_edges(network, end):
        if e.capacity > 1e6:
            print("无上界约束的边",e)
            e.capacity = 0.4e5

    N1 = [e for e in utils.get_out_edges(network, end) if e.capacity < 1e6]
    N2 = [e for e in utils.get_in_edges(network, end) if e.capacity < 1e6]
    # 这样修改的话会导致seperation_gcf那里需要用capacity乘的地方有问题
    # N1 = [e for e in utils.get_out_edges(network, end)]
    # N2 = [e for e in utils.get_in_edges(network, end)]
    x = model.variables["sku_flow"]
    y = model.variables["select_edge"]
    d = 0
    t = 0
    cutv, lbdv, subsets = seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False)
    print("lbd",lbdv)
    if cutv > 1e-2:
        # 修改点2
        expr, cut_value, lbd, bool_voilate = eval_cut_c3(x, y, t, d, *subsets, lbdv=lbdv)
        model.model.addConstr(expr <= 0)
        model.model.solve()
        df_edges = model.get_solution(f"sol_lp_{pick_instance}/")
    else:
        print("cannot find")
        print(df_edges['y'])
        break
    # 看这一步和上一步相比的提升效果 新增了看这一步和mip之间的效果
    print(f"""
--- {k} ---
{start, end, t}
{model.model.objval - lpval}
{model.model.objval - mipval}
    """)
    lpval = model.model.objval
    # print(df_edges)
    k += 1