import json
import pickle

import coptpy
import coptpy as cp
from coptpy import COPT
import const
import utils
import cg_init
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from entity import SKU, Customer
from tqdm import tqdm
from np_cg import *
import os
from network import constuct_network, get_pred_reachable_nodes
from read_data import read_data

from param import Param

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