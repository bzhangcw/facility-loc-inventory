import coptpy as cp
from coptpy import COPT
import CONST
from utils import get_in_edges, get_out_edges
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from Entity import SKU, Customer
from tqdm import tqdm
import os



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

    def get_subgraph(self):
        """
        Get a subgraph for each customer from the original graph
        """

        self.subgraph = {}
        pass

    def construct_oracle(self, customer: Customer):
        """
        Construct oracles  for each customer
        """

        pass

    def init_cols(self, customer: Customer):
        """
        Initialize the columns for the subproblem according to the oracles
        """

        self.columns = {}
        pass

    def init_RMP(self):
        """
        Initialize the RMP with initial columns
        """

        pass

    def solve_RMP(self):
        """
        Solve the RMP and get the dual variables to construct the subproblem
        """

        pass

    def update_RMP(self):
        """
        Update the RMP with new columns
        """

        pass

    def subproblem(self, customer: Customer, dual_vars):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

        v = 0 # the objective value of the subproblem
        added = False # whether a new column is added

        x = None # the x variable of the subproblem
        y = None # the y variable of the subproblem
        p = None # the p variable of the subproblem

        if v < 0:
            added = True
            self.columns[customer].append((x, y, p))

        return added
    
    def CG(self):
        """
        The main loop of column generation algorithm
        """

        self.get_subgraph()
        for customer in self.customer_list:
            self.construct_oracle(customer)
            self.init_cols(customer)
        self.init_RMP()

        while True: # may need to add a termination condition
            self.solve_RMP()

            added = False
            for customer in self.customer_list:
                added = self.subproblem(customer, self.RMP_model.getDuals()) or added

            if not added:
                break

            self.update_RMP()