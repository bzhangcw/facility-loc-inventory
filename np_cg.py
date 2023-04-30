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
from DNP_model import DNP
from Param import Param


class NP_CG:

    def __init__(self, arg: argparse.Namespace, network: nx.DiGraph, customer_list: List[Customer], full_sku_list: List[SKU] = None, open_relationship = False) -> None:
        self.arg = arg
        self.network = network
        self.full_sku_list = full_sku_list if full_sku_list is not None else self.network.graph['sku_list']

        self.RMP_env = cp.Envr('RMP_env')
        self.RMP_model = self.RMP_env.createModel('RMP')
        self.customer_list = customer_list # List[Customer]
        self.subgraph = {}# Dict[customer, nx.DiGraph]
        self.columns = {} # Dict[customer, List[tuple(x, y, p)]]
        self.oracles = {} # Dict[customer, copt.model]
        self.open_relationship = open_relationship

    def get_subgraph(self):
        """
        Get a subgraph for each customer from the original graph
        """
        for customer in self.customer_list:
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
            self.subgraph[customer] = self.network.subgraph(related_nodes)


    def construct_oracle(self, customer: Customer):
        """
        Construct oracles for each customer
        """
        subgraph = self.subgraph[customer]
        full_sku_list = subgraph.graph['sku_list']
        param = Param()
        arg = param.arg
        arg.T = 1
        env_name = customer.idx + '_oracle_env'
        model_name = customer.idx + '_oracle'

        return DNP(arg, subgraph, full_sku_list, env_name, model_name, self.open_relationship, False, True)


    def init_cols(self, customer: Customer):
        """
        Initialize the columns for the subproblem according to the oracles
        """

        oracle = self.oracles[customer]
        oracle.modeling()
        oracle.solve()
        beta = 0 # the objective value of the oracle
        self.columns[customer] = [(oracle.vars, beta)]

    def init_RMP(self):
        """
        Initialize the RMP with initial columns
        """

        ################# add variables #################
        var_types = {
            'column_weights': {
                'lb': 0,
                'ub': 1,
                'vtype': COPT.CONTINUOUS,
                'nameprefix': 'lambda',
                'index': '(customer, number)'
            },
        }

        # generate index tuple
        idx = dict()
        for vt in var_types.keys():
            idx[vt] = list()

        for customer in self.customers:
            for number in range(len(self.columns[customer])):
                idx['column_weights'].append((customer, number))
        
        # add variables
        for vt, param in var_types.items():
            print(f"  - {vt}")
            vars[vt] = self.RMP_model.addVars(
                idx[vt],
                lb=param['lb'],
                ub=param['ub'],
                vtype=param['vtype'],
                nameprefix=f"{param['nameprefix']}_"
            )

        ################# add constraints #################
        self.constr_types = {
            'transportation_capacity': {
                'index': '(edge)'
            },
            'production_capacity': {
                'index': '(node)'
            },
            'holding_capacity': {
                'index': '(node)'
            }
        }

        # edge transportation capacity
        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            transportation = 0
            for customer in self.customer_list:
                if e in self.subgraph[customer].edges: #can we do better?
                    for number in range(len(self.columns[customer])):
                        transportation += vars[customer, number] * self.columns[customer][number][0]['sku_flow'].sum(0, edge, '*')
                    
            constr = self.RMP_model.addConstr(transportation <= edge.capacity)
            self.constrs['transportation_capacity'][edge] = constr

        for node in self.network.nodes:
            # node production capacity
            if node.type == CONST.PLANT:

                production = 0
                for customer in self.customer_list:
                    if node in self.subgraph[customer].nodes:
                        for number in range(len(self.columns[customer])):
                            production += vars[customer, number] * self.columns[customer][number][0]['sku_production'].sum(0, node, '*')
                
                constr = self.RMP_model.addConstr(production <= node.production_capacity)
                self.constrs['production_capacity'][node] = constr

            if node.type == CONST.WAREHOUSE:
                # node holding capacity
                holding = 0
                for customer in self.customer_list:
                    if node in self.subgraph[customer].nodes:
                        for number in range(len(self.columns[customer])):
                            holding += vars[customer, number] * self.columns[customer][number][0]['sku_inventory'].sum(0, node, '*')
                
                constr = self.RMP_model.addConstr(holding <= node.inventory_capacity)
                self.constrs['holding_capacity'][node] = constr

        # weights sum to 1
        constr = self.RMP_model.addConstr(vars['column_weights'].sum() == 1)
        self.constrs['weights_sum'] = constr

        ################# set objective #################
        obj = 0
        for customer in self.customer_list:
            for number in range(len(self.columns[customer])):
                obj += vars[customer, number] * self.columns[customer][number][1]
        
        self.RMP_model.setObjective(obj, COPT.MINIMIZE)

    def solve_RMP(self):
        """
        Solve the RMP and get the dual variables to construct the subproblem
        """
        self.RMP_model.solve()

    def update_RMP(self):
        """
        Update the RMP with new columns
        """

        # can incrementally update the RMP?
        self.init_RMP()

    def subproblem(self, customer: Customer, dual_vars):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

        added = False # whether a new column is added
        self.oracles[customer].update_objective(dual_vars)
        self.oracles[customer].solve()
        new_column = self.oracles[customer].vars
        v = self.oracles[customer].objVal

        if v < 0:
            added = True
            self.columns[customer].append(new_column)

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


if __name__ == "__main__":
    datapath = './data_0401_V3.xlsx'
    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
        data_dir=datapath, sku_num=10, plant_num=20, warehouse_num=10, customer_num=3)
    node_list = plant_list + warehouse_list + customer_list
    network = constuct_network(node_list, edge_list, sku_list)
    print(network)
    print(network.edges)
    customer1 = customer_list[1]
    np_cg = NP_CG(argparse.Namespace,network,customer_list, sku_list)
    subgraph = np_cg.get_subgraph()
    print(subgraph.edges())
    for node in subgraph.nodes():
        print(node,node.get_node_sku_list(0,sku_list))
    print(subgraph)