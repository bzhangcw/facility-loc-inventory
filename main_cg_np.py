from np_cg import *

if __name__ == "__main__":
    datapath = "data/data_0401_V3.xlsx"
    cfg = dict(
        data_dir=datapath,
        sku_num=2,
        plant_num=2,
        warehouse_num=13,
        customer_num=5,
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

    # use external capacity, todo, move internals
    # cap = pd.read_csv("./data/random_capacity.csv").set_index("id")
    # for e in edge_list:
    #     e.capacity = cap["qty"].get(e.idx, np.inf)
    #     e.variable_lb = cap["lb"].get(e.idx, np.inf)
    network = constuct_network(node_list, edge_list, sku_list)
    ###############################################################

    param = Param()
    arg = param.arg
    arg.T = 1
    # arg.backorder = False
    # max_iter = 15
    max_iter = 15

    # pd = "primal"  # the way to initialize columns
    pd = None  # the way to initialize columns

    np_cg = NP_CG(
        arg,
        network,
        customer_list,
        sku_list,
        max_iter=max_iter,
        open_relationship=True,
        pd=pd,
    )

    np_cg.CG()
