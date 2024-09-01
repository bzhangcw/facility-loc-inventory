import logging.handlers
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import List
from instance_generator import *
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from config.network import construct_network
from config.read_data import read_data
from entity import Edge, Node


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


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""

    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


fhd = logging.handlers.RotatingFileHandler(
    filename=f"{CONF.DEFAULT_TMP_PATH}/events.log", backupCount=5
)
tqd = TqdmLoggingHandler()
tqd.setFormatter(lft)
logger.addHandler(fhd)
logger.addHandler(tqd)

logger.info("{:^20s}".format("-- The FACINV python package --"))
logger.info("{:^20s}".format("LLLGZ, 2023 (c)"))
logger.info(f":solution      to {CONF.DEFAULT_SOL_PATH}")
logger.info(f":data          to {CONF.DEFAULT_DATA_PATH}")
logger.info(f":logs and tmps to {CONF.DEFAULT_TMP_PATH}")


def configuration(conf_label, arg):
    if conf_label == 1:
        # Basic version: only consider the capacity constraint
        # 1. Basic constraints
        arg.covering = 0
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 0
        arg.add_in_upper = 0
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 0
        arg.cardinality = 0
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 0
    elif conf_label == 2:
        # Consider the capacity constraint and the edge lower bound constraint
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 0
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 0
        arg.cardinality = 0
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 0
    elif conf_label == 3:
        # Consider the capacity constraint, the edge lower bound constraint, add_in_upper constraint
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 0
        arg.cardinality = 0
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 0
    elif conf_label == 4:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 1
        arg.cardinality = 0
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 0
    elif conf_label == 5:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 1
        # arg.backorder = 1
        arg.cardinality = 1
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 0
    elif conf_label == 6:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 1
        # arg.backorder = 1
        arg.cardinality = 1
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 1
    elif conf_label == 7:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 1
        arg.cardinality = 1
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 1
    elif conf_label == 8:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 1
        # arg.backorder = 1
        arg.cardinality = 1
        # 4. Node and cost
        arg.node_lb = 1
        arg.if_fixed_cost = 1
    elif conf_label == 9:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 0
        # 2. Operational constraints
        arg.edge_lb = 0
        arg.add_in_upper = 0
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 1
        arg.cardinality = 0
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 1
    elif conf_label == 10:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 0
        # arg.backorder = 1
        arg.cardinality = 0
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 1
    elif conf_label == 11:
        # 1. Basic constraints
        arg.covering = 1
        arg.capacity = 1
        # 2. Operational constraints
        arg.edge_lb = 1
        arg.add_in_upper = 1
        # 3. Customization constraints
        arg.distance = 1
        # arg.backorder = 1
        arg.cardinality = 1
        # 4. Node and cost
        arg.node_lb = 0
        arg.if_fixed_cost = 1


def scale(pick_instance, datapath, arg):
    logger.info(f"time scale {arg.T}")
    if pick_instance == 0:
        cfg = dict(
            data_dir=datapath,
            sku_num=1,
            plant_num=1,
            warehouse_num=2,
            customer_num=1,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 1:
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
    elif pick_instance == 8:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=519,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 7:
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
    elif pick_instance == 12:
        cfg = dict(
            data_dir=datapath,
            sku_num=500,
            plant_num=2,
            warehouse_num=5,
            customer_num=4,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 13:
        cfg = dict(
            data_dir=datapath,
            sku_num=500,
            plant_num=20,
            warehouse_num=50,
            customer_num=200,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 14:
        cfg = dict(
            data_dir=datapath,
            sku_num=500,
            plant_num=50,
            warehouse_num=100,
            customer_num=300,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 15:
        cfg = dict(
            data_dir=datapath,
            sku_num=500,
            plant_num=50,
            warehouse_num=200,
            customer_num=500,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 16:
        cfg = dict(
            data_dir=datapath,
            sku_num=500,
            plant_num=200,
            warehouse_num=1841,
            customer_num=1456,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )

    elif pick_instance == 17:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=50,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 18:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=100,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 19:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=150,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 20:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=200,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 21:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=250,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 22:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=300,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 23:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=350,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 24:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=400,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 25:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=450,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    elif pick_instance == 26:
        cfg = dict(
            data_dir=datapath,
            sku_num=141,
            plant_num=23,
            warehouse_num=28,
            customer_num=500,
            # customer_num=10,
            one_period=(True if arg.T == 1 else False),
        )
    else:
        cfg = dict(data_dir=datapath, one_period=True)
    package = get_data_from_cfg(cfg, arg)

    return package


def add_attr(edge_list, node_list, arg, const):
    if arg.new_data:
        data_dir = arg.fpath
    else:
        data_dir = "data/_history_/"
    for e in edge_list:
        e.variable_lb = 0
    if arg.capacity == 1:
        capacity_path = data_dir + "edge_capacity.csv"
        cap = pd.read_csv(capacity_path).set_index("id")
        for e in edge_list:
            # e.capacity = cap["qty"].get(e.idx, np.inf)
            e.capacity = cap["qty"].get(e.idx, 1e4) * arg.capacity_ratio
    if arg.edge_lb == 1:
        lb_end_path = data_dir + "lb_end.csv"
        lb_end = pd.read_csv(lb_end_path).set_index("id")
        for e in edge_list:
            if e.idx in lb_end["lb"]:
                e.variable_lb = lb_end["lb"].get(e.idx, 0) * arg.lb_end_ratio
        lb_end_path = data_dir + "lb_inter.csv"
        lb_inter = pd.read_csv(lb_end_path).set_index("id")
        for e in edge_list:
            if e.idx in lb_inter["lb"]:
                e.variable_lb = lb_inter["lb"].get(e.idx, 0) * arg.lb_inter_ratio
    if arg.node_lb == 1:
        lb_node_path = data_dir + "lb_node.csv"
        lb_df = pd.read_csv(lb_node_path).set_index("id")
        for n in node_list:
            if n.type == const.WAREHOUSE:
                n.inventory_lb = lb_df["lb"].get(n.idx, np.inf) * arg.node_lb_ratio
            if n.type == const.PLANT:
                n.production_lb = lb_df["lb"].get(n.idx, np.inf) * arg.node_lb_ratio


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

    # return (
    #         cfg["data_dir"].split("/")[-1].split(".")[0]
    #         + "-"
    #         + "-".join([str(cfg[k]) for k in keys if k != "data_dir"])
    # )
    return (
            cfg["data_dir"].split("/")[1]
            + "-"
            + "-".join([str(cfg[k]) for k in keys if k != "data_dir"])
    )


def get_data_from_cfg(cfg, arg):
    sig = dump_cfg_tofname(cfg)
    fp = f"{CONF.DEFAULT_SOL_PATH}/{sig}.pk"
    if os.path.exists(fp):
        logger.info("current data has been generated before")
        logger.info(f"reading from cache: {fp}")
        package = pickle.load(open(fp, "rb"))
    else:
        logger.info("current data has not been generated before")
        logger.info(f"creating a temporary cache @{fp}")
    if arg.new_data:
        (
            sku_list,
            plant_list,
            warehouse_list,
            customer_list,
            edge_list,
        ) = generate_instance(**cfg)
    else:
        (
            sku_list,
            plant_list,
            warehouse_list,
            customer_list,
            edge_list,
        ) = read_data(**cfg)

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
    logger.info(f"setting arg.sku_list...")
    arg.sku_list = package[0]
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
    def __init__(self, k, name, logging=True):
        self.k = k
        self.name = name
        self.logging = logging

    def __enter__(self):
        if self.logging:
            logger.info(f"task {self.name} started")
        self.start = time.time()
        return self

    def __exit__(
            self,
            *arg,
    ):
        self.end = time.time()
        self.interval = self.end - self.start
        global_timers.append([self.k, self.name, self.interval])
        if self.logging:
            logger.info(f"task {self.name} takes {self.interval:.2f} seconds")


def visualize_timers():
    pd.set_option("display.max_columns", None)
    df = pd.DataFrame(data=global_timers, columns=["k", "name", "time"])
    df.set_index(["name", "k"]).to_excel(f"{CONF.DEFAULT_SOL_PATH}/timing.xlsx")
    logger.info(
        f"""
=== describing time statistics ===
{df.groupby("name")['time'].describe().fillna("-").reset_index()}
    """
    )
