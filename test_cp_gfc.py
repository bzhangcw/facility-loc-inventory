import pandas as pd
from coptpy import COPT
from gurobipy import GRB
import numpy as np
from template_generate import *
import utils as utils
from config.network import construct_network
from config.param import Param
from dnp_model import DNP
from ncg.np_cg import *
from csp.cp_gfc import *
import pickle
import folium

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    arg.backorder_sku_unit_cost = 5000
    arg.capacity_node_ratio = 1
    arg.capacity_ratio = 1
    arg.cardinality_limit = 30
    arg.distance_limit = 5000
    arg.holding_sku_unit_cost = 1
    arg.in_upper_ratio = 0.24    # 2/2.8的时候LP=MIP
    # arg.lb_end_ratio = 2.8
    # arg.lb_inter_ratio = 2.8
    arg.lb_end_ratio = 1
    arg.lb_inter_ratio = 1
    arg.node_lb_ratio = 1
    arg.d = 150
    arg.unfulfill_sku_unit_cost = 5000
    arg.conf_label = 3
    arg.pick_instance = 2
    arg.backorder = 0
    arg.transportation_sku_unit_cost = 1
    arg.T = 1
    arg.terminate_condition = 1e-5
    arg.new_data = 1
    arg.num_periods = 20
    arg.add_in_upper = False

    utils.configuration(arg.conf_label, arg)
    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
    )
    arg.new_data = 1
    arg.num_periods = 20
    arg.terminate_condition = 1e-4
    datapath = 'facility-loc-inventory/data/small_instance_generate_2/'
    arg.fpath = datapath
    # print(datapath)
    dnp_mps_name = f"new_guro_{datapath.split('/')[1]}_{arg.T}_{arg.conf_label}@{arg.pick_instance}@{arg.backorder}.mps"
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

    # save graph object to file
    pickle.dump(network, open('data/mygraph.pickle', 'wb'))
    # load graph object from file
    G = pickle.load(open('data/mygraph.pickle', 'rb'))
    m = folium.Map(location=[40, -100], zoom_start=3)
    # 标记不同的颜色
    for node in G.nodes:
        if str(node).startswith('C'):
            color = 'red'
        elif str(node).startswith('T'):
            color = 'blue'
        else:
            color = 'green'
        folium.Marker(location=node.location, popup=f"Name: {node.idx}", icon=folium.Icon(color=color)).add_to(m)

    for edge in G.edges:
        folium.PolyLine(locations=[edge[0].location, edge[1].location], color='blue').add_to(m)

    # 保存地图为 HTML 文件
    m.save('data/mygraph.pickle.html')
    print('save over')
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
    model.solve()
    mipval = model.get_model_objval()
    os.makedirs(f"sol_lp_{arg.pick_instance}_MIP_DNP/", exist_ok=True)
    model.get_solution(f"sol_lp_{arg.pick_instance}_MIP_DNP/")
    model.write('data/model_mip.lp')
    df_edges = pd.read_csv(f"sol_lp_{arg.pick_instance}_MIP_DNP/edge_sku_t_flow.csv")
    print('MIP SOLUTION',mipval)
    print('holding_cost',model.obj['holding_cost'][0].getExpr().getValue())
    print('transportation_cost',model.obj['transportation_cost'].getExpr().getValue())
    print('unfulfilled_demand_cost',model.obj['unfulfilled_demand_cost'][0].getExpr().getValue())
    for _,row in df_edges.iterrows():
        print(row["obj_start"], row["obj_end"], row["t"],row['y'])
    print("-----------LP------------")
    # model.model.solve()
    for k, v in model.variables['select_edge'].items():
        # if k[1].end.type != "C":
        v.setType(COPT.CONTINUOUS)
        print(f"relaxing {k}")
    
    for k, v in model.variables['sku_select_edge'].items():
        # if k[1].end.type != "C":
        v.setType(COPT.CONTINUOUS)
        print(f"relaxing {k}")
    model.solve()
    lpval = model.get_model_objval()
    model.write('data/model_lp.lp')
    print('LP SOLUTION',lpval)
    os.makedirs(f"sol_lp_{arg.pick_instance}_LP_DNP/", exist_ok=True)
    model.get_solution(f"sol_lp_{arg.pick_instance}_LP_DNP/")
    df_edges = pd.read_csv(f"sol_lp_{arg.pick_instance}_LP_DNP/edge_sku_t_flow.csv")
    if df_edges.empty:
        print("No solution")
    else:
        print('----------Originial Solution----------')
        for _,row in df_edges.iterrows():
            if 0 < row['y'] <1:
                print(row["obj_start"], row["obj_end"], row["t"],row['y'])
    print('------Cost Information--------')
    print('holding_cost',model.obj['holding_cost'][0].getExpr().getValue())
    print('transportation_cost',model.obj['transportation_cost'].getExpr().getValue())
    print('unfulfilled_demand_cost',model.obj['unfulfilled_demand_cost'][0].getExpr().getValue())
    t = 0
    for e in network.edges:
        edge = network.edges[e]["object"]
        variable_y = model.variables["select_edge"][t, edge].x
        if 0 < variable_y < 1:
            print("fractional edge", edge)
            start, end = edge.start, edge.end
            k = 0
            node = edge.end
            # if edge.end.type == "C":
            #     node = edge.start
            # else:
            #     node = edge.end
            while k < 100:
                for e in utils.get_out_edges(network, node):
                    if e.capacity > 1e6:
                        print("无上界约束的边", e)
                        e.capacity = 0.4e5
                N1 = [e for e in utils.get_out_edges(network, node)]
                N2 = []
                # N2 = [e for e in utils.get_in_edges(network, node)]
                print(N1,N2)
                x = model.variables["sku_flow"]
                y = model.variables["select_edge"]
                if node.type == "C":
                    d = 100
                else:
                    d = arg.d
                print('---------Seperation Start----------')
                n = 0
                if len(N1)>0 or len(N2)>0:
                    print(node)
                    cutv, lbdv, subsets = seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False)
                    print("lbd", lbdv)
                    print("cutv", cutv)
                    if cutv > 1e-2:
                        n = n+1
                        print('---------Add Cut----------')
                        expr, cut_value, lbd, bool_voilate = eval_cut_c3(model,x, y, t, d, *subsets, lbdv=lbdv)
                        model.model.addConstr(expr <= 0)
                        model.model.solve()
                        os.makedirs(f"sol_mip_{arg.pick_instance}_DNP_{n}/", exist_ok=True)
                        model.get_solution(f"sol_mip_{arg.pick_instance}_DNP_{n}/")
                        df_edges = pd.read_csv(f"sol_mip_{arg.pick_instance}_DNP_{n}/edge_sku_t_flow.csv")
                        for _, row in df_edges.iterrows():
                            print('------------------')
                            print(row["obj_start"], row["obj_end"], row["t"], row['y'])
                else:
                    print("cannot find")
                    for _, row in df_edges.iterrows():
                        print(row["obj_start"], row["obj_end"], row["t"], row['y'])
                    break
                print(f"""
                --- {k} ---
                {start, end, t}
                {model.model.objval}
                {model.model.objval}
                    """)
                lpval = model.model.objval
                k += 1

            for _, row in df_edges.iterrows():
                if 0 < row['y'] < 1:
                    print(row["obj_start"], row["obj_end"], row["t"], row['y'])

    # # 筛选qty的数量小于lb的边
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

    #             N1 = [e for e in utils.get_out_edges(network, end) if e.capacity < 1e6]
    #             N2 = [e for e in utils.get_in_edges(network, end) if e.capacity < 1e6]
    #             x = model.variables["sku_flow"]
    #             y = model.variables["select_edge"]
    #             d = 0
    #             print('---------Seperation Start----------')
    #             if len(N1)>0 or len(N2)>0:
    #                 print(end)
    #                 cutv, lbdv, subsets = seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False)
    #                 print("lbd", lbdv)
    #                 if cutv > 1e-2:
    #                     print('---------Add Cut----------')
    #                     expr, cut_value, lbd, bool_voilate = eval_cut_c3(x, y, t, d, *subsets, lbdv=lbdv)
    #                     model.model.addConstr(expr <= 0)
    #                     model.model.solve()
    #                     os.makedirs(f"sol_mip_{arg.pick_instance}_DNP_{n}/", exist_ok=True)
    #                     model.get_solution(f"sol_mip_{arg.pick_instance}_DNP_{n}/")
    #                     df_edges = pd.read_csv(f"sol_mip_{arg.pick_instance}_DNP_{n}/edge_sku_t_flow.csv")
    #             else:
    #                 print("cannot find")
    #                 for _, row in df_edges.iterrows():
    #                     if 0 < row['y'] < 1:
    #                         print(row["obj_start"], row["obj_end"], row["t"], row['y'])
    #                 break
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