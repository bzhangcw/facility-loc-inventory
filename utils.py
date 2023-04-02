from typing import List
from Entity import Node, SKU, Edge
import CONST


def intersect_list(l1: List, l2: List) -> List:
    l = []
    for e in l1:
        if e in l2:
            l.append(e)
    return l


def get_edge_sku_list(edge: Edge, t: int, full_sku_list: List[SKU]) -> List[SKU]:
    if edge.start.type == CONST.PLANT and edge.end.type == CONST.CUSTOMER:
        sku_list_start = edge.start.producible_sku
        sku_list_end = edge.end.demand_sku[t]
    elif edge.start.type == CONST.PLANT and edge.end.type == CONST.WAREHOUSE:
        sku_list_start = edge.start.producible_sku
        sku_list_end = None
    elif edge.start.type == CONST.WAREHOUSE and edge.end.type == CONST.CUSTOMER:
        sku_list_start = None
        sku_list_end = edge.end.demand_sku[t]
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
    if node.type == CONST.PLANT:
        sku_list = node.producible_sku
    elif node.type == CONST.WAREHOUSE:
        sku_list = full_sku_list
    elif node.type == CONST.CUSTOMER:
        sku_list = node.demand_sku[t]

    return sku_list
