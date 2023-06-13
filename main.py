from read_data import read_data
from network import constuct_network
from dnp_model import DNP
from param import Param
import os
import utils

if __name__ == "__main__":
    param = Param()
    arg = param.arg

    arg.T = 27
    fpath = "data_0401_V3.xlsx"
    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=f"{utils.CONF.DEFAULT_DATA_PATH}/{fpath}",
        sku_num=5,
        plant_num=5,
        warehouse_num=5,
        customer_num=5,
    )

    node_list = plant_list + warehouse_list + customer_list

    network = constuct_network(node_list, edge_list, sku_list)
    model = DNP(arg, network)
    model.modeling()
    model.solve()

    # solpath = '/Users/sky/Desktop/computer/sky/projects/NetworkFlow/output/output'
    model.get_solution(data_dir=utils.CONF.DEFAULT_SOL_PATH)
