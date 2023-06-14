from np_cg import *

if __name__ == '__main__':
    datapath = "data/data_0401_V3.xlsx"
    cfg = dict(
        data_dir=datapath,
        sku_num=2,
        plant_num=2,
        warehouse_num=13,
        customer_num=2,
        one_period=True
    )

    (sku_list, plant_list, warehouse_list, customer_list, edge_list,
     network, node_list, *_) = utils.get_data_from_cfg(cfg)

    param = Param()
    arg = param.arg
    arg.T = 1
    arg.backorder = False
    np_cg = NP_CG(arg, network, customer_list, sku_list, max_iter=15)

    np_cg.CG()
