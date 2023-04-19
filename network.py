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
    Simplify the topology based on distances
    """

    nodes = graph.nodes()
    # print("Before:",len(graph.edges))

    distance_map = {}
    for start in nodes:
        distances = []
        for e in graph.out_edges(start):
            distance = graph.edges[e]['object'].cal_distance()
            print(distance)
            distances.append((e, distance))
        n = math.ceil(ratio * len(graph.out_edges(start)))
        distances.sort(key=lambda x:x[1])
        distance_map[start] = distances[:n]

    edges_to_remove = []
    for start, distances in distance_map.items():
        node_to_remove = set(graph.out_edges(start)) - set([dist[0] for dist in distances])
        edges_to_remove.extend([end for end in node_to_remove])
    graph.remove_edges_from(edges_to_remove)

    # print("After:",len(graph.edges))
    return graph
