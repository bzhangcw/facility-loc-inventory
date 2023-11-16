import argparse
import json
import os
import pickle
from typing import List

import coptpy
import coptpy as cp
import networkx as nx
import numpy as np
import pandas as pd
from coptpy import COPT
from tqdm import tqdm

import cg_init
import const
import utils
from entity import SKU, Customer
from network import construct_network, get_pred_reachable_nodes
from np_cg import *
from param import Param
from read_data import read_data

# def update_edge_capacity(cg_object,cus_idx):
#     updated_capacity = {}
#     for e in cg_object.network.edges:
#         edge = cg_object.network.edges[e]["object"]
#         updated_capacity[edge] = 0
#     for i in range(cus_idx):
#         customer_before = cg_object.customer_list[i]
#         for e in cg_object.subgraph[customer_before].edges:
#             edge = cg_object.network.edges[e]["object"]
#             updated_capacity[edge] = updated_capacity[edge] + cg_object.columns_helpers[customer_before]["sku_flow_sum"][edge]
#     return  updated_capacity
