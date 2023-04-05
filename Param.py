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
            "Dynamic Network Problem"
        )

        parser.add_argument(
            "--T",
            type=int,
            default=1,
            help="""
            number of periods for the problem
            """
        )

        parser.add_argument(
            "--transportation_cost",
            type=bool,
            default=True,
            help="""
            whether to consider end inventory level
            """
        )

        parser.add_argument(
            "--end_inventory",
            type=bool,
            default=True,
            help="""
            whether to consider end inventory level
            """
        )

        return parser


if __name__ == "__main__":
    param = Param()
    arg = param.arg
    print(arg.T)
