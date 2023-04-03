import CONST
import numpy as np
import pandas as pd
from typing import List


class SKU:
    """
    this is a class for SKU
    """

    def __init__(self, idx: str, weight: float = 1.0) -> None:
        self.idx = idx
        self.weight = weight

    def __str__(self) -> str:
        return f"SKU_{self.idx}"

    def __repr__(self) -> str:
        return f"SKU_{self.idx}"

    # def __hash__(self) -> int:
        # return hash(self.__repr__())


class Node:
    """
    this is a class for nodes
    """

    def __init__(self, idx: str, location: np.ndarray) -> None:
        self.idx = idx
        self.location = location

        self.type = None

    def __str__(self) -> str:
        return f"Node_{self.idx}"


class Plant(Node):
    """
    this is a class for plant

    extend from Node
    """

    def __init__(
        self, idx: str, location: np.ndarray,
        production_capacity: float,
        producible_sku: List[SKU],
        production_sku_rate: "pd.Series[SKU, float]" = None,
        production_sku_capacity: "pd.Series[SKU, float]" = None,
        production_fixed_cost: float = 0.0,
        production_sku_fixed_cost: "pd.Series[SKU, float]" = None,
        production_sku_unit_cost: "pd.Series[SKU, float]" = None
    ) -> None:

        super().__init__(idx, location)
        self.production_capacity = production_capacity
        self.producible_sku = producible_sku
        self.production_sku_rate = production_sku_rate
        self.production_sku_capacity = production_sku_capacity
        self.production_fixed_cost = production_fixed_cost
        self.production_sku_fixed_cost = production_sku_fixed_cost
        self.production_sku_unit_cost = production_sku_unit_cost

        self.type = CONST.PLANT

    def __str__(self) -> str:
        return f"Plant_{self.idx}"

    def __repr__(self) -> str:
        return f"Plant_{self.idx}"


class Warehouse(Node):
    """
    this is a class for warehouse

    extend from node
    """

    def __init__(
        self, idx: str, location: 'np.ndarray[float, float]',
        inventory_capacity: float,
        if_current: bool = False,
        inventory_sku_capacity: "pd.Series[SKU, float]" = None,
        holding_fixed_cost: float = 0.0,
        holding_sku_unit_cost: "pd.Series[SKU, float]" = None,
        initial_inventory: "pd.Series[SKU, float]" = None,
        end_inventory: "pd.Series[SKU, float]" = None,
        end_inventory_bias_cost: float = 0.0,
        demand: "pd.Series[(int, SKU), float]" = None,
        demand_sku: "pd.Series[int, List[SKU]]" = None,
        unfulfill_sku_unit_cost: "pd.Series[(int, SKU), float]" = None
    ) -> None:

        super().__init__(idx, location)
        self.inventory_capacity = inventory_capacity
        self.inventory_sku_capacity = inventory_sku_capacity
        self.holding_fixed_cost = holding_fixed_cost
        # self.holding_sku_fixed_cost = holding_sku_fixed_cost
        self.holding_sku_unit_cost = holding_sku_unit_cost
        self.initial_inventory = initial_inventory
        self.end_inventory = end_inventory
        self.end_inventory_bias_cost = end_inventory_bias_cost
        self.demand = demand
        self.demand_sku = demand_sku
        self.unfulfill_sku_unit_cost = unfulfill_sku_unit_cost
        self.if_current = if_current
        self.type = CONST.WAREHOUSE

    def __str__(self) -> str:
        return f"Warehouse_{self.idx}"

    def __repr__(self) -> str:
        return f"Warehouse_{self.idx}"


class Customer(Node):
    """
    this is a class for customer

    extend from node
    """

    def __init__(
        self, idx: str, location: np.ndarray,
        demand: "pd.Series[(int, SKU), float]",
        demand_sku: "pd.Series[int, List[SKU]]",
        unfulfill_sku_unit_cost: "pd.Series[(int, SKU), float]" = None
    ) -> None:

        super().__init__(idx, location)
        self.demand = demand
        self.demand_sku = demand_sku
        self.unfulfill_sku_unit_cost = unfulfill_sku_unit_cost

        self.type = CONST.CUSTOMER

    def __str__(self) -> str:
        return f"Customer_{self.idx}"

    def __repr__(self) -> str:
        return f"Customer_{self.idx}"


class Edge:
    """
    this is a class for edge
    """

    def __init__(
        self, idx: str, start: Node, end: Node,
        capacity: float, distance: float = None,
        transportation_fixed_cost: float = 0.0,
        transportation_sku_fixed_cost: "pd.Series[SKU, float]" = None,
        transportation_sku_unit_cost: "pd.Series[SKU, float]" = None
    ) -> None:

        self.idx = idx
        self.start = start
        self.end = end
        self.capacity = capacity
        self.distance = distance if distance is not None else self.cal_distance()
        self.transportation_fixed_cost = transportation_fixed_cost
        self.transportation_sku_fixed_cost = transportation_sku_fixed_cost
        self.transportation_sku_unit_cost = transportation_sku_unit_cost

    def cal_distance(self):
        return np.linalg.norm(self.start.location - self.end.location)

    def __str__(self) -> str:
        return f"Edge_{self.idx}_({self.start}, {self.end})"

    def __repr__(self) -> str:
        return f"Edge_{self.idx}_({self.start}, {self.end})"


if __name__ == "__main__":
    sku = SKU('1')
    print(sku)

    sku_list = [sku]

    plant = Plant('1', np.array([1, 1]), 1, sku_list)
    print(plant)

    demand = pd.Series({(0, sku): 1})
    demand_sku = pd.Series({0: [sku]})
    customer = Customer('1', np.array([2, 3]), demand, demand_sku)
    print(customer)

    edge = Edge('e', plant, customer, 10)
    print(edge)
