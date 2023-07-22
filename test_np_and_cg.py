from np_cg import *
import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import construct_network
from param import Param
from coptpy import COPT

import os
import time


def write1dict2csv(
    path, result: dict, float_format: str = None, mode="a", encoding="utf-8"
):
    """
    write/append one dict to csv
    """

    assert path[-4:] == ".csv"

    if os.path.isfile(path):
        header_flag = False
    else:
        header_flag = True

    df = pd.DataFrame(result, index=[0])
    df.to_csv(
        path,
        header=header_flag,
        float_format=float_format,
        mode=mode,
        encoding=encoding,
        index=False,
    )

    return


def now():
    t = time.localtime()
    now = "{}-{}-{}".format(t.tm_year, t.tm_mon, t.tm_mday)
    return now


if __name__ == "__main__":
    # output_flag = True
    output_flag = False
    outputfile = "./out_debug/large_np_cg_{}.csv".format(now())

    datapath = "data/data_0401_V3.xlsx"
    # sku_num_list = list(range(10, 60, 20))
    # plant_num_list = list(range(20, 60, 20))
    # warehouse_num_list = list(range(60, 120, 20))
    # customer_num_list = list(range(20, 120, 20))

    sku_num_list = [140]  # max 140
    plant_num_list = [23]  # max 23
    warehouse_num_list = [28]  # max 28
    customer_num_list = [120]  # max 472

    num_prob = (
        len(sku_num_list)
        * len(plant_num_list)
        * len(warehouse_num_list)
        * len(customer_num_list)
    )

    k_prob = 0

    for sku in sku_num_list:
        for plant in plant_num_list:
            for wareh in warehouse_num_list:
                for cus in customer_num_list:
                    k_prob += 1
                    print(
                        f"Problem {k_prob}/{num_prob}: sku_num={sku}, plant_num={plant}, warehouse_num={wareh}, customer_num={cus}"
                    )

                    result = dict()
                    result["param-sku-num"] = sku
                    result["param-plant-num"] = plant
                    result["param-warehouse-num"] = wareh
                    result["param-customer-num"] = cus

                    cfg = dict(
                        data_dir=datapath,
                        sku_num=sku,
                        plant_num=plant,
                        warehouse_num=wareh,
                        customer_num=cus,
                        one_period=True,
                    )
                    # cfg = dict(data_dir=datapath, one_period=True)
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

                    result["real-sku-num"] = len(sku_list)
                    result["real-plant-num"] = len(plant_list)
                    result["real-warehouse-num"] = len(warehouse_list)
                    result["real-customer-num"] = len(customer_list)
                    result["real-edge-num"] = len(edge_list)

                    # # use external capacity, todo, move internals
                    cap = pd.read_csv("./data/random_capacity.csv").set_index("id")
                    for e in edge_list:
                        e.bool_capacity = cap["qty"].get(e.idx, np.inf)
                        # e.variable_lb = cap["lb"].get(e.idx, np.inf)
                        pass
                    network = construct_network(node_list, edge_list, sku_list)
                    ###############################################################

                    ############################   args   #########################
                    param = Param()
                    arg = param.arg
                    arg.T = 1
                    arg.backorder = False

                    ############################  Solver  #########################
                    try:
                        model = DNP(arg, network, cus_num=1)
                        model.modeling()
                        model.model.setParam("Logging", 1)
                        # model.model.setParam("RelGap", 1.3)
                        # model.model.setParam("LpMethod", 2)  # interior point method

                        model.solve()
                        model.get_solution(data_dir=utils.CONF.DEFAULT_SOL_PATH)
                        result["DNP-time"] = model.model.solvingtime
                        result["DNP-obj"] = model.model.objVal
                    except:
                        print("DNP failed")
                        result["DNP-time"] = np.inf
                        result["DNP-obj"] = np.inf
                    ###############################################################

                    ############################    CG    #########################
                    max_iter = 100
                    init_primal = None
                    init_dual = None  # 'dual'

                    result["CG-max-iter"] = max_iter

                    try:
                        t_begin = time.time()
                        np_cg = NetworkColumnGeneration(
                            arg,
                            network,
                            customer_list,
                            sku_list,
                            max_iter=max_iter,
                            bool_covering=True,
                            init_primal=init_primal,
                            init_dual=init_dual,
                        )

                        np_cg.run()
                        t_end = time.time()
                        result["CG-iter"] = np_cg.iter
                        result["CG-time"] = t_end - t_begin
                        result["CG-obj"] = np_cg.RMP_model.objval
                    except:
                        print("CG failed")
                        result["CG-iter"] = np.inf
                        result["CG-time"] = np.inf
                        result["CG-obj"] = np.inf

                    ############################  output  #########################

                    print(result)

                    if output_flag:
                        write1dict2csv(path=outputfile, result=result)
