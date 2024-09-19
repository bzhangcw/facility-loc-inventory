import argparse


class Param:
    """
    this is a class for parameters
    """

    def __init__(self) -> None:
        parser = self.init_parser()
        # self.arg = parser.parse_args()
        self.arg, unknown = parser.parse_known_args()

    def init_parser(self):
        """
        add or change default args here
        """
        parser = argparse.ArgumentParser(
            "Dynamic Network Problem",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--template_choose",
            type=str,
        )

        parser.add_argument(
            "--path",
            type=str,
            help="data path",
            default="data/data_0401_0inv.xlsx",
            # required=True,
        )

        parser.add_argument(
            "--fpath",
            type=str,
            help="data path",
            default="data/data_0401_0inv.xlsx",
            # required=True,
        )
        parser.add_argument(
            "--bool_use_ncg",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--new_data",
            type=bool,
            default=True,
        )

        parser.add_argument(
            "--sku_list",
            type=list,
            default=[],
            # required=True,
        )

        parser.add_argument(
            "--terminate_condition",
            type=float,
            default=1e-2,
            # required=True,
        )
        parser.add_argument("--T", type=int, default=7, help="time horizon")

        parser.add_argument(
            "--backend",
            type=str,
            help="solver backend",
            default="gurobi",
            choices=["gurobi", "copt"],
        )
        parser.add_argument(
            "--use_ray",
            type=int,
            help="whether to use distributed mode powered by ray",
            default=1,
        )
        ##### 0. CG parameters #####
        parser.add_argument(
            "--cg_method_mip_heuristic",
            "-p",
            type=int,
            help="""
            the choice of primal method after collecting a 
                set of columns (finished convexification)
            """,
            default=-1,
        )

        parser.add_argument(
            "--cg_itermax",
            "-i",
            type=int,
            help="""
            the maximum number of iterations
            """,
            default=10,
        )

        parser.add_argument(
            "--cg_mip_recover",
            type=int,
            default=1,
            help="""
            whether to recover an integral feasible solution
                in the RMP
            """,
        )
        parser.add_argument(
            "--cg_rmp_mip_iter",
            type=int,
            default=10,
            help="""
            the interval to invoke an integral heuristic
            """,
        )
        parser.add_argument(
            "--check_cost_cg",
            type=int,
            default=0,
            help="""
            for debugging only, 
                check the cost functions of CG
            """,
        )

        ##############################
        # 1. problem size & configuration
        ##############################
        parser.add_argument(
            "--conf_label",
            type=int,
            default=0,
            help="""
            a problem is defined by
             - a config that specify
             - using the same config you can choose different size by
                `pick_instance` 
            """,
        )

        parser.add_argument(
            "--pick_instance",
            type=int,
            default=0,
        )

        parser.add_argument(
            "--pricing_relaxation",
            type=int,
            default=0,
            help="""
            0: no; 1: yes;
            whether solving pricing problem as LP relaxation; 
                in principle, pricing should be solved as MIP only.
            """,
        )

        ##############################
        # 2. fixed cost
        ##############################
        parser.add_argument(
            "--production_sku_unit_cost",
            type=float,
            # default=1e10,
            default=1.5,
        )

        parser.add_argument(
            "--holding_sku_unit_cost",
            type=float,
            default=1,
        )

        parser.add_argument(
            "--transportation_sku_unit_cost",
            type=float,
            default=0.01,
        )

        parser.add_argument(
            "--plant_fixed_cost",
            type=float,
            default=200,
        )

        parser.add_argument(
            "--warehouse_fixed_cost",
            type=float,
            default=500,
        )

        parser.add_argument(
            "--unfulfill_sku_unit_cost",
            type=float,
            default=500,
        )

        # todo
        parser.add_argument(
            "--backorder_sku_unit_cost",
            type=float,
            default=200,
        )
        ##############################
        # 3. geometric restrictions
        ##############################

        parser.add_argument(
            "--add_cardinality",
            type=int,
            default=0,
        )

        parser.add_argument(
            "--cardinality_limit",
            type=int,
            default=2,
        )

        parser.add_argument(
            "--add_distance",
            type=int,
            default=0,
        )

        parser.add_argument(
            "--distance_limit",
            type=int,
            default=1000,
        )
        parser.add_argument(
            "--in_upper_ratio",
            type=int,
            default=0.54,
        )

        parser.add_argument(
            "--backorder",
            type=int,
            default=0,
            help="""
            whether to allow backorder, inventory can be true if backorder allowed
            """,
        )
        ##############################
        # 4. basic configuration
        ##############################
        parser.add_argument(
            "--covering",
            type=int,
            default=1,
            help="""
            covering
            """,
        )

        parser.add_argument(
            "--capacity",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--add_in_upper",
            type=int,
            default=0,
        )

        parser.add_argument(
            "--in_upper_qty",
            type=int,
            default=0,
        )
    

        parser.add_argument(
            "--edge_lb",
            type=int,
            default=0,
        )

        parser.add_argument(
            "--node_lb",
            type=int,
            default=0,
        )

        parser.add_argument(
            "--if_fixed_cost",
            type=int,
            default=0,
        )

        ##############################
        # 5. miscellanea
        ##############################
        parser.add_argument(
            "--total_cus_num",
            type=int,
            default=472,
        )

        parser.add_argument(
            "--cus_num",
            type=int,
            default=4,
        )

        parser.add_argument(
            "--num_skus",
            type=int,
            default=500,
        )

        parser.add_argument(
            "--num_periods",
            type=int,
            default=30,
        )

        parser.add_argument(
            "--demand_type",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--capacity_ratio",
            type=int,
            default=0.1,
        )

        parser.add_argument(
            "--capacity_node_ratio",
            type=int,
            default=0.1,
        )

        parser.add_argument(
            "--node_lb_ratio",
            type=int,
            default=0.1,
        )

        parser.add_argument(
            "--d",
            type=int,
            default=150,
        )

        parser.add_argument(
            "--lb_end_ratio",
            type=int,
            default=10,
        )

        parser.add_argument(
            "--lb_inter_ratio",
            type=int,
            default=100,
        )

        # control deleteing columns
        parser.add_argument(
            "--if_del_col",
            type=int,

        )

        parser.add_argument(
            "--del_col_freq",
            type=int,
            default=3,
        )

        parser.add_argument(
            "--del_col_stra",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--check_number",
            type=int,
            default=3,
        )

        parser.add_argument(
            "--del_col_alg",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--column_pool_len",
            type=int,
            default=3,
        )
        
        # parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

        parser.add_argument(
            "--rounding_heuristic",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--rounding_heuristic_1",
            type=bool,
            default=False,
        )


        parser.add_argument(
            "--rounding_heuristic_2",
            type=bool,
            default=False,
        )

        parser.add_argument(
            "--rounding_heuristic_3",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--rounding_heuristic_4",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--print_solution",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--NCS",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--DNP",
            type=int,
            default=1,
        )

        parser.add_argument(
            "--pricing_network",
            type=bool,
            default=False,
        )
        return parser


if __name__ == "__main__":
    param = Param()
    arg = param.arg
    print(arg.T)
