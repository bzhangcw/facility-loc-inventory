from typing import List
from Entity import Node, Edge
import networkx as nx


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
