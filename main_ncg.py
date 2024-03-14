from ncg.np_cg import *

"""
Run following command in the command line of Turing when using Ray:
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
"""

if __name__ == "__main__":
    param = Param()
    arg = param.arg
    utils.configuration(arg.conf_label, arg)
    datapath = arg.fpath
    arg.bool_use_ncg = 1
    print(
        json.dumps(
            arg.__dict__,
            indent=2,
            sort_keys=True,
        )
    )
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

    solver = arg.backend.upper()

    print("----------NCG------------")
    max_iter = 100
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
        init_ray=False,
    )

    np_cg.run()
    np_cg.get_solution("new_sol_1/")
