from read_data import read_data
from network import constuct_network
from network import prune
from DNP_model import DNP
from Param import Param

if __name__ == '__main__':
    param = Param()
    arg = param.arg

    arg.T = 27

    datapath = './data_0401_V3.xlsx'

    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=datapath, sku_num=5, plant_num=5, warehouse_num=5, customer_num=5)

    node_list = plant_list + warehouse_list + customer_list

    network = constuct_network(node_list, edge_list, sku_list)

    simplifed_network = prune(network,0.2)
    model = DNP(arg, simplifed_network)
    model.modeling()
    model.solve()

    solpath = './simplifed_output'
    model.get_solution(solpath)
