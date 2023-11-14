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
    # # 工厂是100 仓库是500
    # arg.node_cost = 0
    # # 边是10
    # arg.edge_cost = 0

    arg.T = 7
    arg.capacity = 1
    # arg.lowerbound = 1
    datapath = "data/data_0401_0inv.xlsx"
    pick_instance = 3
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

    # arg.add_in_upper = 1
    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.model.setParam("Threads", 8)
    model.model.setParam("TimeLimit", 3600)
    # model.model.setParam("RelGap", 1.3)
    # model.model.setParam("LpMethod", 2)  # interior point method
    # 把DNP的model写出来
    # model.model.write("mm.lp")

    model.model.write("finaltest/C{}_L{}_CL{}_NU{}_Fn{}_Fe{}_NL{}.mps".format(arg.cus_num, arg.lowerbound, arg.cp_lowerbound, arg.add_in_upper, arg.node_cost, arg.edge_cost, arg.nodelb))

    # DNP的最优解是 7373.645201525
    # model.model.solve()
    # mipval = model.get_model_objval()
    # # 在这ipynb下面新建了一个文件夹来存储输出数据
    # os.makedirs(f"sol_mip_{pick_instance}_DNP/", exist_ok=True)
    # model.get_solution(f"sol_mip_{pick_instance}_DNP/")
#
# for k, v in model.variables['select_edge'].items():
#     #k(0,edge)
#     if k[1].end.type != "C":
#         # if v != 1:
#         v.setType(COPT.CONTINUOUS)
#         print(f"relaxing {k}")
#
# model.solve()
#
# lpval = model.get_model_objval()
# os.makedirs(f"sol_lp_{pick_instance}/", exist_ok=True)
# lpval = model.model.objval
# print(
#     f"""
# --- summary ---------------------------
# lp relaxation: {lpval},
# mip          : {mipval},
#
# """
# )
# # from tqdm.notebook import tqdm as tqdm_notebook
# #
# # import utils
# from cp_gfc import *
# # model.model.setParam("Logging", 0)
# df_edges = model.get_solution(f"sol_lp_{pick_instance}/")
# if df_edges.empty:
#     print("no solution")
# else:
#     print('----------Original Solution----------')
#     for _,row in df_edges.iterrows():
#         if 0 < row['y'] <1:
#             print(row["obj_start"], row["obj_end"], row["t"],row['y'])
#     # print(df_edges['y'])
# lpval = model.model.objval
#
# # for t in range(arg.T):
# #
# if not query_fractional_inter_edges(df_edges).empty:
#     n = 1
#     for _, row in query_fractional_inter_edges(df_edges).iterrows():
#         start, end, t = row["obj_start"], row["obj_end"], row["t"]
#         print("------------------Fractional Inter Edge------------------")
#         print(k, start, end, t)
#         k = 0
#         while k < 20:
#             # break
#             for e in utils.get_out_edges(network, end):
#                 if e.capacity > 1e6:
#                     print("无上界约束的边", e)
#                     e.capacity = 0.4e5
#             n1_temp = Warehouse(idx="Feak{}T{}".format(end, t), location=[], inventory_capacity=end.inventory_capacity,
#                                 inventory_lb=0)
#             n2_temp = Warehouse(idx="Feak{}T{}".format(end, t - 1), location=[],
#                                 inventory_capacity=end.inventory_capacity, inventory_lb=0)
#             edge_id_1 = end.idx + "_" + n1_temp.idx
#             edge_id_2 = n2_temp.idx + "_" + end.idx
#             n1_edge = Edge(
#                 idx=edge_id_1,
#                 start=end,
#                 end=n1_temp,
#                 capacity=end.inventory_capacity,
#                 variable_lb=0,
#                 distance=0,
#                 transportation_fixed_cost=0,
#             )
#             n2_edge = Edge(
#                 idx=edge_id_2,
#                 start=n2_temp,
#                 end=end,
#                 capacity=end.inventory_capacity,
#                 variable_lb=0,
#                 distance=0,
#                 transportation_fixed_cost=0,
#             )
#             N1 = [e for e in utils.get_out_edges(network, end) if e.capacity < 1e6]
#             N2 = [e for e in utils.get_in_edges(network, end) if e.capacity < 1e6]
#             N1.append(n1_edge)
#             N2.append(n2_edge)
#             x = model.variables["sku_flow"]
#             y = model.variables["select_edge"]
#             d = 0
#             print('---------Seperation Start----------')
#             cutv, lbdv, subsets = seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False)
#             print("lbd", lbdv)
#             if cutv > 1e-2:
#                 # 修改点2
#                 print('---------Add Cut----------')
#                 expr, cut_value, lbd, bool_voilate = eval_cut_c3(model, x, y, t, d, *subsets, lbdv=lbdv)
#                 model.model.addConstr(expr <= 0)
#                 model.model.solve()
#                 df_edges = model.get_solution(f"sol_lp_{pick_instance}/")
#             else:
#                 print("cannot find")
#                 print(model.model.objval - mipval,mipval,model.model.objval - mipval/mipval)
#                 # print(df_edges['y'])
#                 for _, row in df_edges.iterrows():
#                     if 0 < row['y'] < 1:
#                         print(row["obj_start"], row["obj_end"], row["t"], row['y'])
#                 break
#             # 看这一步和上一步相比的提升效果 新增了看这一步和mip之间的效果
#             print(f"""
#             --- {k} ---
#             {start, end, t}
#             {model.model.objval - lpval}
#             {model.model.objval - mipval}
#                 """)
#             lpval = model.model.objval
#             k += 1
#         print('After adding cuts for {}th fractional inter edge, then we have:'.format(n))
#         for _, row in df_edges.iterrows():
#             if 0 < row['y'] < 1:
#                 print(row["obj_start"], row["obj_end"], row["t"], row['y'])
#         n += 1
# else:
#     print("no fractional inter edges")
#     print(df_edges['y'])
#
# # while k < 20:
# #     if not query_fractional_inter_edges(df_edges).empty:
# #         for _, row in query_fractional_inter_edges(df_edges).iterrows():
# #             start, end, t = row["obj_start"], row["obj_end"], row["t"]
# #             print(k,start,end,t)
# #             break
# #         # print(start,end)
# #         # # 修改点1
# #         for e in utils.get_out_edges(network, end):
# #             if e.capacity > 1e6:
# #                 print("无上界约束的边",e)
# #                 e.capacity = 0.4e5
# #
# #         N1 = [e for e in utils.get_out_edges(network, end) if e.capacity < 1e6]
# #         N2 = [e for e in utils.get_in_edges(network, end) if e.capacity < 1e6]
# #         # 这样修改的话会导致seperation_gcf那里需要用capacity乘的地方有问题
# #         # N1 = [e for e in utils.get_out_edges(network, end)]
# #         # N2 = [e for e in utils.get_in_edges(network, end)]
# #         x = model.variables["sku_flow"]
# #         y = model.variables["select_edge"]
# #         d = 0
# #         print('---------Seperation Start----------')
# #         cutv, lbdv, subsets = seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False)
# #         print("lbd", lbdv)
# #         if cutv > 1e-2:
# #             # 修改点2
# #             print('---------Add Cut----------')
# #             expr, cut_value, lbd, bool_voilate = eval_cut_c3(x, y, t, d, *subsets, lbdv=lbdv)
# #             model.model.addConstr(expr <= 0)
# #             model.model.solve()
# #             df_edges = model.get_solution(f"sol_lp_{pick_instance}/")
# #         else:
# #             print("cannot find")
# #             # print(df_edges['y'])
# #             for _, row in df_edges.iterrows():
# #                 if 0 < row['y'] < 1:
# #                     print(row["obj_start"], row["obj_end"], row["t"], row['y'])
# #             break
# #         # 看这一步和上一步相比的提升效果 新增了看这一步和mip之间的效果
# #         print(f"""
# # --- {k} ---
# # {start, end, t}
# # {model.model.objval - lpval}
# # {model.model.objval - mipval}
# #     """)
# #         lpval = model.model.objval
# #         for _, row in df_edges.iterrows():
# #             if 0 < row['y'] < 1:
# #                 print(row["obj_start"], row["obj_end"], row["t"], row['y'])
# #         # print(df_edges)
# #         k += 1
# #     else:
# #         print("no fractional inter edges")
# #         print(df_edges['y'])
# #         break