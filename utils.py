import logging.handlers
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import List

import networkx as nx
import pandas as pd

from entity import Edge, Node
from config.network import construct_network
from config.read_data import read_data

import numpy as np


class CONF:
    DEFAULT_DATA_PATH = "./data"
    DEFAULT_TMP_PATH = "./tmp"
    DEFAULT_SOL_PATH = "./out"


if not os.path.exists(CONF.DEFAULT_TMP_PATH):
    os.mkdir(CONF.DEFAULT_TMP_PATH)

if not os.path.exists(CONF.DEFAULT_SOL_PATH):
    os.mkdir(CONF.DEFAULT_SOL_PATH)

lft = logging.Formatter("[%(name)s] %(asctime)-8s: %(message)s")
logger = logging.getLogger("facinv")
logger.setLevel(logging.INFO)

chd = logging.StreamHandler(sys.stdout)
chd.setFormatter(lft)
fhd = logging.handlers.RotatingFileHandler(
    filename=f"{CONF.DEFAULT_TMP_PATH}/events.log", backupCount=5
)
logger.addHandler(chd)
logger.addHandler(fhd)

logger.info("{:^20s}".format("-- The FACINV python package --"))
logger.info("{:^20s}".format("LLLGZ, 2023 (c)"))
logger.info(f":solution      to {CONF.DEFAULT_SOL_PATH}")
logger.info(f":data          to {CONF.DEFAULT_DATA_PATH}")
logger.info(f":logs and tmps to {CONF.DEFAULT_TMP_PATH}")


def configuration(conf_label, arg):
    if conf_label == 1:
        # Basic version: only consider the capacity constraint
        arg.backorder = 0
        arg.customer_backorder = 0
        arg.node_cost = 0
        arg.edge_cost = 0
        arg.capacity = 1
        arg.lowerbound = 0
        arg.cp_lowerbound = 0
        arg.nodelb = 0
    elif conf_label == 2:
        # Consider the capacity constraint and the edge lower bound constraint
        arg.backorder = 0
        arg.customer_backorder = 0
        arg.node_cost = 0
        arg.edge_cost = 0
        arg.capacity = 1
        arg.lowerbound = 1
        arg.cp_lowerbound = 1
        arg.nodelb = 0
    elif conf_label == 3:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint
        arg.backorder = 1
        arg.customer_backorder = 1
        arg.node_cost = 0
        arg.edge_cost = 0
        arg.capacity = 1
        arg.lowerbound = 1
        arg.cp_lowerbound = 1
        arg.distance = 1
        arg.cardinality = 1
        arg.nodelb = 0
    elif conf_label == 4:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges
        arg.backorder = 1
        arg.customer_backorder = 1
        arg.node_cost = 1
        arg.edge_cost = 1
        arg.capacity = 1
        arg.lowerbound = 1
        arg.cp_lowerbound = 1
        arg.nodelb = 0
    elif conf_label == 5:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges. Consider the customization constraints such as distance and cardinality constraints.
        arg.backorder = 1
        arg.customer_backorder = 1
        arg.node_cost = 1
        arg.edge_cost = 1
        arg.capacity = 1
        arg.lowerbound = 1
        arg.cp_lowerbound = 1
        arg.nodelb = 0
        arg.distance = 1
        arg.cardinality = 1
    elif conf_label == 6:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges. Consider the customization constraints such as distance and cardinality constraints.
        arg.backorder = 1
        arg.customer_backorder = 1
        arg.node_cost = 1
        arg.edge_cost = 1
        arg.capacity = 1
        arg.lowerbound = 1
        arg.cp_lowerbound = 1
        arg.nodelb = 0
        arg.distance = 1
        arg.cardinality = 1
        arg.T = 7
    elif conf_label == 7:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges. Consider the customization constraints such as distance and cardinality constraints.
        arg.backorder = 0
        arg.customer_backorder = 0
        arg.node_cost = 1
        arg.edge_cost = 1
        arg.capacity = 1
        arg.lowerbound = 1
        arg.cp_lowerbound = 1
        arg.nodelb = 0
        arg.distance = 1
        arg.cardinality = 1
        arg.T = 7
    elif conf_label == 8:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges. Consider the customization constraints such as distance and cardinality constraints.
        arg.backorder = 0
        arg.customer_backorder = 1
        arg.fixed_cost = 1
        arg.capacity = 1
        arg.edgelb = 1
        arg.nodelb = 0
        arg.distance = 0
        arg.cardinality = 1
        arg.add_in_upper = 1
        arg.T = 7
    elif conf_label == 9:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges. Consider the customization constraints such as distance and cardinality constraints.
        arg.backorder = 0
        arg.customer_backorder = 1
        arg.fixed_cost = 1
        arg.capacity = 1
        arg.edgelb = 1
        arg.nodelb = 1
        arg.distance = 1
        arg.cardinality = 1
        arg.add_in_upper = 1
        arg.T = 432
    elif conf_label == 10:
        # Consider the capacity constraint, the edge lower bound constraint and backorder constraint. Consider the fixed cost of nodes and edges. Consider the customization constraints such as distance and cardinality constraints.
        arg.backorder = 0
        arg.customer_backorder = 1
        arg.fixed_cost = 1
        arg.capacity = 1
        arg.edgelb = 1
        arg.nodelb = 0
        arg.distance = 0
        arg.cardinality = 1
        arg.add_in_upper = 1
        arg.T = 14
    


def scale(pick_instance, datapath, arg):
    logger.info(f"time scale {arg.T}")
    if pick_instance == 1:
        # 只有一个customer的成功的案例
        cfg = dict(
            data_dir=datapath,
            sku_num=1,
            plant_num=20,
            warehouse_num=2,
            customer_num=2,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 2:
        cfg = dict(
            data_dir=datapath,
            sku_num=10,
            plant_num=20,
            warehouse_num=20,
            customer_num=4,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 3:
        cfg = dict(
            data_dir=datapath,
            sku_num=20,
            plant_num=20,
            warehouse_num=20,
            customer_num=100,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 4:
        cfg = dict(
            data_dir=datapath,
            sku_num=30,
            plant_num=20,
            warehouse_num=20,
            customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 5:
        cfg = dict(
            data_dir=datapath,
            sku_num=100,
            plant_num=20,
            warehouse_num=20,
            customer_num=100,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 6:
        cfg = dict(
            data_dir=datapath,
            sku_num=140,
            plant_num=20,
            warehouse_num=20,
            customer_num=519,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 7:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=519,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 8:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=200,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 9:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=1000,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 10:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=10000,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 11:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=10,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    else:
        cfg = dict(data_dir=datapath, one_period=True)
    package = get_data_from_cfg(cfg)

    return package


def add_attr(edge_list, node_list, arg, const):
    for e in edge_list:
        e.variable_lb = 0
    if arg.capacity == 1:
        cap = pd.read_csv("data/random_capacity_updated.csv").set_index("id")
        for e in edge_list:
            # e.capacity = cap["qty"].get(e.idx, np.inf)
            e.capacity = cap["qty"].get(e.idx, 1e4)
    # if arg.lowerbound == 1:
    #     lb_end = pd.read_csv("data/lb_end.csv").set_index("id")
    #     for e in edge_list:
    #         if e.idx in lb_end["lb"]:
    #             e.variable_lb = lb_end["lb"].get(e.idx, 0)
    # if arg.cp_lowerbound == 1:
    #     lb_inter = pd.read_csv("data/lb_inter.csv").set_index("id")
    #     for e in edge_list:
    #         if e.idx in lb_inter["lb"]:
    #             e.variable_lb = lb_inter["lb"].get(e.idx, 0) / 10
    #             print(f"setting {e.idx} to {e.variable_lb}")
    if arg.edgelb == 1:
        lb_end = pd.read_csv("data/lb_end.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_end["lb"]:
                e.variable_lb = lb_end["lb"].get(e.idx, 0)
        lb_inter = pd.read_csv("data/lb_inter.csv").set_index("id")
        for e in edge_list:
            if e.idx in lb_inter["lb"]:
                e.variable_lb = lb_inter["lb"].get(e.idx, 0) / 10
                # print(f"setting {e.idx} to {e.variable_lb}")
    if arg.nodelb == 1:
        lb_df = pd.read_csv("./data/node_lb_V3.csv").set_index("id")
        for n in node_list:
            if n.type == const.WAREHOUSE:
                n.inventory_lb = lb_df["lb"].get(n.idx, np.inf)
            if n.type == const.PLANT:
                n.production_lb = lb_df["lb"].get(n.idx, np.inf)


def dump_cfg_tofname(cfg):
    """
    create a signiture to dump data
    :param cfg:
    """
    import json

    infostr = json.dumps(cfg, indent=2, sort_keys=True)
    logger.info("generating the signature of this problem")
    logger.info(infostr)
    keys = sorted(cfg.keys())

    return (
        cfg["data_dir"].split("/")[-1].split(".")[0]
        + "-"
        + "-".join([str(cfg[k]) for k in keys if k != "data_dir"])
    )


def get_data_from_cfg(cfg):
    sig = dump_cfg_tofname(cfg)
    fp = f"{CONF.DEFAULT_SOL_PATH}/{sig}.pk"
    if os.path.exists(fp):
        logger.info("current data has been generated before")
        logger.info(f"reading from cache: {fp}")
        package = pickle.load(open(fp, "rb"))
    else:
        logger.info("current data has not been generated before")
        logger.info(f"creating a temporary cache @{fp}")
        sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
            **cfg
        )

        node_list = plant_list + warehouse_list + customer_list
        network = construct_network(node_list, edge_list, sku_list)
        package = (
            sku_list,
            plant_list,
            warehouse_list,
            customer_list,
            edge_list,
            network,
            node_list,
        )
        logger.info(f"dumping a temporary cache @{fp}")
        with open(fp, "wb") as _fo:
            pickle.dump(package, _fo)
    return package


def get_in_edges(network: nx.DiGraph, node: Node) -> List[Edge]:
    """
    The function returns list of edges into node

    :param network: network
    :param node: node
    """

    return [e[2]["object"] for e in list(network.in_edges(node, data=True))]


def get_out_edges(network: nx.DiGraph, node: Node) -> List[Edge]:
    """
    The function returns list of edges out of node

    :param network: network
    :param node: node
    """

    return [e[2]["object"] for e in list(network.out_edges(node, data=True))]


global_timers = []


class TimerContext:
    def __init__(self, k, name):
        self.k = k
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        global_timers.append([self.k, self.name, self.interval])


def visualize_timers():
    df = pd.DataFrame(data=global_timers, columns=["k", "name", "time"])
    df.set_index(["name", "k"]).to_excel(f"{CONF.DEFAULT_SOL_PATH}/timing.xlsx")
    logger.info(
        f"""
=== describing time statistics ===
{df.groupby("name")['time'].describe().reset_index()}
    """
    )
