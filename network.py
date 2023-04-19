import networkx as nx
from Entity import SKU, Node, Plant, Warehouse, Customer, Edge
import numpy as np
import pandas as pd
from typing import List
import math
def constuct_network(nodes: List[Node], edges: List[Edge], SKUs: List[SKU]) -> nx.DiGraph:
    """
    construct a graph from given nodes and edges
    """

    graph = nx.DiGraph(sku_list=SKUs)  # directed graph
    graph.add_nodes_from(nodes)

    for e in edges:
        graph.add_edge(e.start, e.end, object=e)

    return graph

def prune(graph,ratio):
    """
    Simplify the topology based on performances
    """

    nodes = graph.nodes()
    # print("Before:",len(graph.edges))

    performance_map = {}
    for start in nodes:
        performances = []
        for e in graph.out_edges(start):
            performance = graph.edges[e]['object'].cal_performance()
            # print(performance)
            performances.append((e, performance))
        n = math.ceil(ratio * len(graph.out_edges(start)))
        performances.sort(key=lambda x:x[1])
        performance_map[start] = performances[:n]

    edges_to_remove = []
    for start, performances in performance_map.items():
        node_to_remove = set(graph.out_edges(start)) - set([dist[0] for dist in performances])
        edges_to_remove.extend([end for end in node_to_remove])
    graph.remove_edges_from(edges_to_remove)

    # print("After:",len(graph.edges))
    return graph
