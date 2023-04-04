import networkx as nx
from Entity import SKU, Node, Plant, Warehouse, Customer, Edge
import numpy as np
import pandas as pd
from typing import List


def constuct_network(nodes: List[Node], edges: List[Edge], SKUs: List[SKU]) -> nx.DiGraph:
    """
    construct a graph from given nodes and edges
    """

    graph = nx.DiGraph(sku_list=SKUs)  # directed graph
    graph.add_nodes_from(nodes)

    for e in edges:
        graph.add_edge(e.start, e.end, object=e)

    return graph


if __name__ == "__main__":
    sku = SKU('1')
    print(sku)

    sku_list = [sku]

    plant = Plant('1', np.array([1, 1]), 1, sku_list)
    print(plant)

    warehouse = Warehouse('1', np.array([1, 2]), 1)
    print(warehouse)

    demand = pd.Series({(0, sku): 1})
    demand_sku = pd.Series({0: [sku]})
    customer = Customer('1', np.array([2, 3]), demand, demand_sku)
    print(customer)

    nodes = [plant, warehouse, customer]

    edges = [
        Edge('e1', plant, warehouse, 10),
        Edge('e2', warehouse, customer, 10)
    ]
    for e in edges:
        print(e)

    graph = constuct_network(nodes, edges, sku_list)
    print(graph.graph['sku_list'])
    print(graph.nodes, graph.edges)

    print(graph.edges[plant, warehouse])

    print(list(graph.successors(plant)))
    print(list(graph.predecessors(customer)))
    print(list(graph.neighbors(warehouse)))

    print(list(graph.in_edges(data=True)), list(graph.in_edges(customer)))
    lst = list(graph.in_edges(customer))

    for node in graph.nodes:
        print(node.location)

    for e in graph.edges:
        edge = graph.edges[e]['object']
        print(edge.cal_distance())
