from typing import List
from Entity import Node, SKU, Edge
import CONST
import networkx as nx


def intersect_list(l1: List, l2: List) -> List:
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


def get_edge_sku_list(edge: Edge, t: int, full_sku_list: List[SKU]) -> List[SKU]:
    """
    > The function gets all possible SKUs flow on edge, i.e. intersection of possible SKUs on start node and end node, at period t

    :param edge: edge
    :param t: period t
    :param full_sku_list: full possible SKU list
    """

    if edge.start.type == CONST.PLANT and edge.end.type == CONST.CUSTOMER:
        sku_list_start = edge.start.producible_sku
        if edge.end.has_demand(t):
            sku_list_end = edge.end.demand_sku[t]
        else:
            sku_list_end = list()
    elif edge.start.type == CONST.PLANT and edge.end.type == CONST.WAREHOUSE:
        sku_list_start = edge.start.producible_sku
        sku_list_end = None
    elif edge.start.type == CONST.WAREHOUSE and edge.end.type == CONST.CUSTOMER:
        sku_list_start = None
        if edge.end.has_demand(t):
            sku_list_end = edge.end.demand_sku[t]
        else:
            sku_list_end = list()
    elif edge.start.type == CONST.WAREHOUSE and edge.end.type == CONST.WAREHOUSE:
        sku_list_start = None
        sku_list_end = None

    if sku_list_start is None and sku_list_end is None:
        sku_list = full_sku_list
    elif sku_list_start is not None and sku_list_end is None:
        sku_list = sku_list_start
    elif sku_list_start is None and sku_list_end is not None:
        sku_list = sku_list_end
    else:
        sku_list = intersect_list(
            sku_list_start, sku_list_end)

    return sku_list


def get_node_sku_list(node: "Node", t: int, full_sku_list: "List[SKU]"):
    """
    The function gets all possible SKUs on node at period t:
     - for a plant: producible SKUs
     - for a warehouse: full_sku_list
     - for a customer: demand SKUs at period t

    :param node: node
    :param t: period t
    :param full_sku_list: full possible SKU list
    """

    if node.type == CONST.PLANT:
        sku_list = node.producible_sku
    elif node.type == CONST.WAREHOUSE:
        sku_list = full_sku_list
    elif node.type == CONST.CUSTOMER:
        if node.has_demand(t):
            sku_list = node.demand_sku[t]
        else:
            sku_list = list()

    return sku_list


def get_in_edges(network: nx.DiGraph, node: Node) -> List[Edge]:
    """
    The function returns list of edges into node

    :param network: network
    :param node: node
    """

    return [e[2]['object']
            for e in list(network.in_edges(node, data=True))]


def get_out_edges(network: nx.DiGraph, node: Node) -> List[Edge]:
    """
    The function returns list of edges out of node

    :param network: network
    :param node: node
    """

    return [e[2]['object']
            for e in list(network.out_edges(node, data=True))]
