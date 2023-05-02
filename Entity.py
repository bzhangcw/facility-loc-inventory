import CONST
import numpy as np
import pandas as pd
from typing import List
from abc import abstractmethod


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
        self.visited = None

    @abstractmethod
    def get_node_sku_list(self, t: int, full_sku_list: List[SKU]):
        """
        The function gets all possible SKUs on node at period t:
        - for a plant: producible SKUs
        - for a warehouse: full_sku_list
        - for a customer: demand SKUs at period t

        :param node: node
        :param t: period t
        :param full_sku_list: full possible SKU list
        """

        return

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
        self.visited = None
        self.type = CONST.PLANT

    def get_node_sku_list(self, t: int, full_sku_list: List[SKU]):
        """
        The function gets all possible SKUs on node at period t:
        - for a plant: producible SKUs
        - for a warehouse: full_sku_list
        - for a customer: demand SKUs at period t

        :param node: node
        :param t: period t
        :param full_sku_list: full possible SKU list
        """

        sku_list = self.producible_sku

        return sku_list

    def construct_output(self, output):
        '''
        quantity of each SKU produced at each period
        '''
        pass

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
        self, idx: str, location: np.ndarray,
        inventory_capacity: float,
        if_current: bool = False,
        inventory_sku_capacity: "pd.Series[SKU, float]" = None,
        holding_fixed_cost: float = 0.0,
        holding_sku_unit_cost: "pd.Series[SKU, float]" = None,
        backorder_sku_unit_cost: "pd.Series[SKU, float]" = None,
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
        self.backorder_sku_unit_cost = backorder_sku_unit_cost
        self.initial_inventory = initial_inventory
        self.end_inventory = end_inventory
        self.end_inventory_bias_cost = end_inventory_bias_cost
        self.demand = demand
        self.demand_sku = demand_sku
        self.unfulfill_sku_unit_cost = unfulfill_sku_unit_cost
        self.if_current = if_current
        self.type = CONST.WAREHOUSE
        self.visited = None

    def get_node_sku_list(self, t: int, full_sku_list: List[SKU]):
        """
        The function gets all possible SKUs on node at period t:
        - for a plant: producible SKUs
        - for a warehouse: full_sku_list
        - for a customer: demand SKUs at period t

        :param node: node
        :param t: period t
        :param full_sku_list: full possible SKU list
        """

        sku_list = full_sku_list

        return sku_list

    def has_demand(self, t: int, sku: SKU = None) -> bool:
        """
        > This function check whether node has demand (or has certain demand SKU if sku is given) at period t
        """

        demand_flag = False

        # if self.demand_sku is not None and t in self.demand_sku:
        #     if sku is None:
        #         demand_flag = len(self.demand_sku[t]) > 0
        #     else:
        #         demand_flag = sku in self.demand_sku[t]

        return demand_flag

    def construct_output(self, output):
        '''
        quantity of each SKU stored at each period
        quantity of all SKUs stored at each period
        '''
        pass

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
        self.visited = None

    def get_node_sku_list(self, t: int, full_sku_list: List[SKU]):
        """
        The function gets all possible SKUs on node at period t:
        - for a plant: producible SKUs
        - for a warehouse: full_sku_list
        - for a customer: demand SKUs at period t

        :param node: node
        :param t: period t
        :param full_sku_list: full possible SKU list
        """

        if self.has_demand(t):
            sku_list = self.demand_sku[t]
        else:
            sku_list = list()

        return sku_list

    def has_demand(self, t: int, sku: SKU = None) -> bool:
        """
        > This function check whether node has demand (or has certain demand SKU if sku is given) at period t
        """

        demand_flag = False

        if self.demand_sku is not None and t in self.demand_sku:
            if sku is None:
                demand_flag = len(self.demand_sku[t]) > 0
            else:
                demand_flag = sku in self.demand_sku[t]

        return demand_flag

    def construct_output(self, output):
        '''
        quantity of each SKU got at each period
        '''
        pass

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

    def cal_performance(self):
        return self.transportation_sku_unit_cost.sum()
    def get_edge_sku_list(self, t: int, full_sku_list: List[SKU]) -> List[SKU]:
        """
        > The function gets all possible SKUs flow on edge, i.e. intersection of possible SKUs on start node and end node, at period t

        :param edge: edge
        :param t: period t
        :param full_sku_list: full possible SKU list
        """

        if self.start.type == CONST.PLANT and self.end.type == CONST.CUSTOMER:
            sku_list_start = self.start.producible_sku
            if self.end.has_demand(t):
                sku_list_end = self.end.demand_sku[t]
            else:
                sku_list_end = list()
        elif self.start.type == CONST.PLANT and self.end.type == CONST.WAREHOUSE:
            sku_list_start = self.start.producible_sku
            sku_list_end = None
        elif self.start.type == CONST.WAREHOUSE and self.end.type == CONST.CUSTOMER:
            sku_list_start = None
            if self.end.has_demand(t):
                sku_list_end = self.end.demand_sku[t]
            else:
                sku_list_end = list()
        elif self.start.type == CONST.WAREHOUSE and self.end.type == CONST.WAREHOUSE:
            sku_list_start = None
            sku_list_end = None

        if sku_list_start is None and sku_list_end is None:
            sku_list = full_sku_list
        elif sku_list_start is not None and sku_list_end is None:
            sku_list = sku_list_start
        elif sku_list_start is None and sku_list_end is not None:
            sku_list = sku_list_end
        else:
            sku_list = self.intersect_list(
                sku_list_start, sku_list_end)

        return sku_list

    def get_edge_sku_list_with_transportation_cost(self, t: int, full_sku_list: List[SKU]):

        sku_list = self.get_edge_sku_list(
            t, full_sku_list)

        sku_list_with_fixed_transportation_cost = self.intersect_list(
            sku_list, self.transportation_sku_fixed_cost.index.tolist()) if self.transportation_sku_fixed_cost is not None else list()

        sku_list_with_unit_transportation_cost = self.intersect_list(
            sku_list, self.transportation_sku_unit_cost.index.tolist()) if self.transportation_sku_unit_cost is not None else list()

        return sku_list_with_fixed_transportation_cost, sku_list_with_unit_transportation_cost

    def intersect_list(self, l1: List, l2: List) -> List:
        """
        > The function calculates intersection of two lists

        :param l1: list one
        :param l2: list two
        """
        l = []
        for e in l1:
            if e in l2:
                l.append(e)
        return l

    def construct_output(self, output):
        '''
        quantity of each SKU transit at each period
        '''
        pass

    def __str__(self) -> str:
        return f"Edge_{self.idx}_({self.start}, {self.end})"

    def __repr__(self) -> str:
        return f"Edge_{self.idx}_({self.start}, {self.end})"


if __name__ == "__main__":
    sku = SKU('1')
    print(sku)

    sku_list = [sku]

    plant = Plant('1', np.array([1, 1]), 1, sku_list)
    print(plant.get_node_sku_list(0,sku_list))
    print(plant)

    demand = pd.Series({(0, sku): 1})
    demand_sku = pd.Series({0: [sku]})
    customer = Customer('1', np.array([2, 3]), demand, demand_sku)
    print(customer)

    edge = Edge('e', plant, customer, 10)
    print(edge)
    plant.visited = True
    print(plant.visited)
    print(customer.visited)
