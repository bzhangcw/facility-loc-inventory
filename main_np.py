import numpy as np
import pandas as pd

import utils
from dnp_model import DNP
from network import constuct_network
from param import Param

if __name__ == "__main__":
    param = Param()
    arg = param.arg

    datapath = "data/data_0401_V3.xlsx"
    # sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
    #     data_dir=f"{utils.CONF.DEFAULT_DATA_PATH}/{fpath}",
    #     sku_num=5,
    #     plant_num=5,
    #     warehouse_num=5,
    #     customer_num=5,
    # )
    cfg = dict(data_dir=datapath, one_period=True)
    # cfg = dict(
    #     data_dir=datapath,
    #     sku_num=2,
    #     plant_num=2,
    #     warehouse_num=13,
    #     customer_num=2,
    #     one_period=True
    # )

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
    # use external capacity, todo, move internals
    cap = pd.read_csv("./data/random_capacity.csv").set_index("id")
    for e in edge_list:
        e.capacity = cap["qty"].get(e.idx, np.inf)
        e.variable_lb = cap["lb"].get(e.idx, np.inf)
    network = constuct_network(node_list, edge_list, sku_list)

    model = DNP(arg, network)
    model.modeling()
    model.model.setParam("Logging", 1)
    model.solve()

    model.get_solution(data_dir=utils.CONF.DEFAULT_SOL_PATH)
