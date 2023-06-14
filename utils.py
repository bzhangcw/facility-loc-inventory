import logging
import logging.handlers
import os
import pickle
import sys
from typing import List

import networkx as nx

from entity import Node, Edge
from network import constuct_network
from param import Param
from read_data import read_data


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
fhd = logging.handlers.RotatingFileHandler(filename=f"{CONF.DEFAULT_TMP_PATH}/events.log", backupCount=5)
logger.addHandler(chd)
logger.addHandler(fhd)

logger.info("{:^20s}".format("-- The FACINV python package --"))
logger.info("{:^20s}".format("LLLGZ, 2023 (c)"))
logger.info(f":solution      to {CONF.DEFAULT_SOL_PATH}")
logger.info(f":data          to {CONF.DEFAULT_DATA_PATH}")
logger.info(f":logs and tmps to {CONF.DEFAULT_TMP_PATH}")

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
    return "-".join([str(cfg[k]) for k in keys if k != "data_dir"])

def get_data_from_cfg(cfg):
    sig = dump_cfg_tofname(cfg)
    fp = f"{CONF.DEFAULT_SOL_PATH}/{sig}.pk"
    if os.path.exists(fp):
        logger.info("current data has been generated before")
        logger.info(f"reading from cache: {fp}")
        package = pickle.load(open(fp, 'rb'))
    else:
        logger.info("current data has not been generated before")
        logger.info(f"creating a temporary cache @{fp}")
        sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
            **cfg
        )

        node_list = plant_list + warehouse_list + customer_list
        network = constuct_network(node_list, edge_list, sku_list)
        package = (
            sku_list, plant_list, warehouse_list, customer_list, edge_list,
            network, node_list
        )
        logger.info(f"dumping a temporary cache @{fp}")
        with open(fp, 'wb') as _fo:
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
