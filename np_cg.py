import coptpy as cp
from coptpy import COPT
import CONST
from utils import get_in_edges, get_out_edges
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from Entity import SKU, Customer, Warehouse,Plant,Edge
from tqdm import tqdm
import os
from network import constuct_network,get_pred_reachable_nodes
from read_data import read_data


class NP_CG:

    def __init__(self, arg: argparse.Namespace, network: nx.DiGraph, full_sku_list: List[SKU] = None) -> None:
        self.arg = arg
        self.network = network
        self.full_sku_list = full_sku_list if full_sku_list is not None else self.network.graph['sku_list']

        self.RMP_env = cp.Envr('RMP_env')
        self.RMP_model = self.RMP_env.createModel('RMP')

        self.subgraph = None # Dict[customer, nx.DiGraph]
        self.customer_list = None # List[Customer]
        self.columns = None # Dict[customer, List[tuple(x, y, p)]]
        self.oracles = None # Dict[customer, copt.model]

    def get_subgraph(self, customer: Customer):
        """
        Get a subgraph for each customer from the original graph
        """
        pred_reachable_nodes = set()
        get_pred_reachable_nodes(self.network, customer, pred_reachable_nodes)
        related_nodes = pred_reachable_nodes.copy()
        for node in pred_reachable_nodes:
            if bool(node.get_node_sku_list(0, sku_list)):
                if not set(node.get_node_sku_list(0, self.full_sku_list)) & set(customer.get_node_sku_list(0, self.full_sku_list)):
                    related_nodes.remove(node)
            else:
                related_nodes.remove(node)
        related_nodes.add(customer)
        self.subgraph = self.network.subgraph(related_nodes)
        return self.subgraph

if __name__ == "__main__":
    datapath = './data_0401_V3.xlsx'
    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=datapath, sku_num=10, plant_num=20, warehouse_num=10, customer_num=3)
    node_list = plant_list + warehouse_list + customer_list
    network = constuct_network(node_list, edge_list, sku_list)
    print(network)
    print(network.edges)
    customer1 = customer_list[1]
    np_cg = NP_CG(argparse.Namespace,network,sku_list)
    subgraph = np_cg.get_subgraph(customer1)
    print(subgraph.edges())
    for node in subgraph.nodes():
        print(node,node.get_node_sku_list(0,sku_list))
    print(subgraph)