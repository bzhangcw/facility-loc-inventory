from read_data import read_data
from network import constuct_network
from DNP_model import DNP
from Param import Param

if __name__ == '__main__':
    param = Param()
    arg = param.arg

    arg.T = 27

    # datapath = '/Users/sky/Desktop/computer/sky/projects/NetworkFlow/data/Data_0401/data_0401_V3.xlsx'
    datapath = '/Users/liu/Desktop/仓网/data/waiyun/data_0401_V3.xlsx'

    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=datapath)

    node_list = plant_list + warehouse_list + customer_list

    network = constuct_network(node_list, edge_list, sku_list)
    model = DNP(arg, network)
    model.modeling()
    model.solve()
