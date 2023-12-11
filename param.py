import argparse


class Param:
    """
    this is a class for parameters
    """

    def __init__(self) -> None:
        parser = self.init_parser()
        self.arg = parser.parse_args()

    def init_parser(self):
        """
        add or change default args here
        """
        parser = argparse.ArgumentParser(
            "Dynamic Network Problem",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--T",
            type=int,
            default=1,
            help="""
            number of periods for the problem
            """,
        )

        parser.add_argument(
            "--backorder",
            type=bool,
            default=True,
            help="""
            whether to allow backorder, inventory can be true if backorder allowed
            """,
        )

        parser.add_argument(
            "--M",
            type=float,
            # default=1e10,
            default=1e6,
            help="""
            big M for modeling
            """,
        )
        # lk更改点2: unfulfill 取的小点
        parser.add_argument(
            "--unfulfill_sku_unit_cost",
            type=float,
            default=0.5,
            help="""
            default unfulfill_sku_unit_cost if not given
            """,
        )

        parser.add_argument(
            "--holding_sku_unit_cost",
            type=float,
            default=5,
            help="""
            default holding_sku_unit_cost if not given
            """,
        )

        parser.add_argument(
            "--backorder_sku_unit_cost",
            type=float,
            default=20,
            help="""
            default backorder_sku_unit_cost if not given
            """,
        )

        parser.add_argument(
            "--end_inventory_bias_cost",
            type=float,
            default=200,
            help="""
            default end_inventory_bias_cost if not given
            """,
        )

        parser.add_argument(
            "--transportation_sku_unit_cost",
            type=float,
            default=0.01,
            help="""
            default transportation_sku_unit_cost if not given
            """,
        )

        parser.add_argument(
            "--end_inventory",
            type=bool,
            # default=True,
            default=False,
            help="""
            whether to consider end inventory level
            """,
        )

        parser.add_argument(
            "--total_cus_num",
            type=int,
            default=472,
            help="""
            total customer number
            """,
        )

        parser.add_argument(
            "--cus_num",
            type=int,
            default=4,
            help="""
            tested customer number
            """,
        )

        parser.add_argument(
            "--capacity",
            type=int,
            default=1,
            help="""
            whether add capacity constraint or not
            """,
        )
        parser.add_argument(
            "--lowerbound",
            type=int,
            default=1,
            help="""
            whether add lower bound constraint or not
            """,
        )
        parser.add_argument(
            "--cp_lowerbound",
            type=int,
            default=0,
            help="""
            whether add cp lower bound constraint or not
            """,
        )
        parser.add_argument(
            "--partial_fixed",
            type=int,
            default=0,
            help="""
                   whether fixed some edges or not
                   """,
        )
        parser.add_argument(
            "--node_cost",
            type=int,
            default=0,
            help="""
                   whether add fixed cost for nodes or not
                   """,
        )
        parser.add_argument(
            "--nodelb",
            type=int,
            default=0,
            help="""
                   whether add lower bound for nodes or not
                   """,
        )
        parser.add_argument(
            "--edge_cost",
            type=int,
            default=0,
            help="""
                   whether add fixed cost for edges or not
                   """,
        )
        parser.add_argument(
            "--add_inv",
            type=int,
            default=0,
            help="""
                   whether add initial inventory for T0014 and T0015
                   """,
        )
        parser.add_argument(
            "--add_in_upper",
            type=int,
            default=0,
            help="""
                   whether add upper bound for daily inflow of inventories
                   """,
        )

        parser.add_argument(
            "--customer_backorder",
            type=float,
            default=0,
            help="""
                   whether add customer backorder cost or not
                    """,
        )

        parser.add_argument(
            "--add_distance",
            type=float,
            default=0,
            help="""
                whether add customer distance constraint or not
                """,
        )


        parser.add_argument(
            "--add_cardinality",
            type=int,
            default=0,
            help="""
               whether add cardinality constraints for customers or not
               """,
        )

        return parser


if __name__ == "__main__":
    param = Param()
    arg = param.arg
    print(arg.T)
