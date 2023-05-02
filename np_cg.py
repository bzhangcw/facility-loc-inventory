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
        self.dual_index = dict()
        self.vars = dict()  # variables
        self.num_cols = 0

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
        arg = self.arg
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
        # beta = oracle.model.getObjective().getValue()
        beta = oracle.get_original_objective().getExpr().getValue()

        self.columns[customer] = [(oracle.vars, beta)]

    def init_RMP(self):
        """
        Initialize the RMP with initial columns
        """

        ################# add variables #################
        self.var_types = {
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
        for vt in self.var_types.keys():
            idx[vt] = list()

        for customer in self.customer_list:
            for number in range(len(self.columns[customer])):
                idx['column_weights'].append((customer, number))
        
        # add variables
        self.vars = dict()
        for vt, param in self.var_types.items():
            print(f"  - {vt}")
            self.vars[vt] = self.RMP_model.addVars(
                idx[vt],
                lb=param['lb'],
                ub=param['ub'],
                vtype=param['vtype'],
                nameprefix=f"{param['nameprefix']}_"
            )

        ################# add constraints #################
        constr_types = {
            'transportation_capacity': {
                'index': '(edge)'
            },
            'production_capacity': {
                'index': '(node)'
            },
            'holding_capacity': {
                'index': '(node)'
            },
            'weights_sum': {
                'index': '(customer)'
            },
        }
        constrs = dict()
        for constr in constr_types.keys():
            constrs[constr] = dict()

        self.dual_index = {
            'transportation_capacity': dict(),
            'node_capacity': dict(),
            'weights_sum': dict(),
        }
        index = 0

        # edge transportation capacity
        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            transportation = 0.0
            for customer in self.customer_list:
                if e in self.subgraph[customer].edges: #can we do better?
                    for number in range(len(self.columns[customer])):
                        transportation += self.vars['column_weights'][customer, number] * self.columns[customer][number][0]['sku_flow'].sum(0, edge, '*').getValue()

            if transportation == 0.0:
                    continue

            constr = self.RMP_model.addConstr(transportation <= edge.capacity)
            constrs['transportation_capacity'][edge] = constr
            self.dual_index['transportation_capacity'][edge] = index
            index += 1

        for node in self.network.nodes:
            # node production capacity
            if node.type == CONST.PLANT:

                production = 0.0
                for customer in self.customer_list:
                    if node in self.subgraph[customer].nodes:
                        for number in range(len(self.columns[customer])):
                            production += self.vars['column_weights'][customer, number] * self.columns[customer][number][0]['sku_production'].sum(0, node, '*').getValue()
                if production == 0.0:
                    continue
                constr = self.RMP_model.addConstr(production <= node.production_capacity)
                constrs['production_capacity'][node] = constr

            elif node.type == CONST.WAREHOUSE:
                # node holding capacity
                holding = 0.0
                for customer in self.customer_list:
                    if node in self.subgraph[customer].nodes:
                        for number in range(len(self.columns[customer])):
                            holding += self.vars['column_weights'][customer, number] * self.columns[customer][number][0]['sku_inventory'].sum(0, node, '*').getValue()
                
                if holding == 0.0:
                    continue

                constr = self.RMP_model.addConstr(holding <= node.inventory_capacity)
                constrs['holding_capacity'][node] = constr
            else:
                continue

            self.dual_index['node_capacity'][node] = index
            index += 1

        for customer in self.customer_list:
            # weights sum to 1
            constr = self.RMP_model.addConstr(self.vars['column_weights'].sum(customer, '*') == 1)
            constrs['weights_sum'][customer] = constr

            self.dual_index['weights_sum'][customer] = index
            index += 1

        ################# set objective #################
        obj = 0
        for customer in self.customer_list:
            for number in range(len(self.columns[customer])):
                obj += self.vars['column_weights'][customer, number] * self.columns[customer][number][1]
        
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
        self.RMP_model.clear()
        self.init_RMP()

    def subproblem(self, customer: Customer, dual_vars):
        """
        Construct and solve the subproblem
        Only need to change the objective function, subject to the same oracle constraints
        """

        added = False # whether a new column is added
        self.oracles[customer].update_objective(dual_vars, self.dual_index)
        self.oracles[customer].solve()
        new_column = self.oracles[customer].vars
        v = self.oracles[customer].model.getObjective().getValue()

        if v < 0:
            added = True
            beta = self.oracles[customer].original_obj.getExpr().getValue()
            self.columns[customer].append((new_column, beta))

        return added
    
    def CG(self):
        """
        The main loop of column generation algorithm
        """

        self.get_subgraph()
        for customer in self.customer_list:
            self.oracles[customer] = self.construct_oracle(customer)
            self.init_cols(customer)
        self.init_RMP()

        while True: # may need to add a termination condition
            self.num_cols += 1

            self.solve_RMP()

            added = False
            for customer in self.customer_list:
                added = self.subproblem(customer, self.RMP_model.getDuals()) or added

            if not added:
                break

            self.update_RMP()

    def get_solution(self, data_dir: str = './', preserve_zeros: bool = False):
        
        cus_col_value = pd.DataFrame(index=range(self.num_cols), columns=[c.idx for c in self.customer_list])
        cus_col_weights = pd.DataFrame(index=range(self.num_cols), columns=[c.idx for c in self.customer_list])

        for customer in self.customer_list:
            for number in range(len(self.columns[customer])):
                cus_col_value.loc[number, customer.idx] = self.columns[customer][number][1]
                cus_col_weights.loc[number, customer.idx] = self.vars['column_weights'][customer, number].value

        cus_col_value.to_csv(os.path.join(data_dir, 'cus_col_cost.csv'), index=False)
        cus_col_weights.to_csv(os.path.join(data_dir, 'cus_col_weight.csv'), index=False)

if __name__ == "__main__":
    datapath = 'data/data_0401_V3.xlsx'
    # sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(
    #     data_dir=datapath, sku_num=10, plant_num=20, warehouse_num=10, customer_num=3, one_period=True)
    sku_list, plant_list, warehouse_list, customer_list, edge_list = read_data(data_dir=datapath, one_period=True)
    
    node_list = plant_list + warehouse_list + customer_list
    network = constuct_network(node_list, edge_list, sku_list)
    param = Param()
    arg = param.arg
    arg.T = 1
    np_cg = NP_CG(arg,network, customer_list, sku_list)

    np_cg.CG()
    solpath = '/Users/liu/Desktop/MyRepositories/facility-loc-inventory/output'
    np_cg.get_solution(solpath)