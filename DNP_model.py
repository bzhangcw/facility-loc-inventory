import coptpy as cp
from coptpy import COPT
import CONST
from utils import get_in_edges, get_out_edges
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from Entity import SKU
from tqdm import tqdm
import os

class DNP:
    """
    this is a class for dynamic network flow (DNP)
    """

    def __init__(self, arg: argparse.Namespace, network: nx.DiGraph, full_sku_list: List[SKU] = None, env_name: str = "DNP_env", model_name: str = "DNP") -> None:
        self.arg = arg
        self.T = arg.T
        self.network = network
        self.full_sku_list = full_sku_list if full_sku_list is not None else self.network.graph[
            'sku_list']

        self.env = cp.Envr(env_name)
        self.model = self.env.createModel(model_name)
        self.vars = dict()  # variables
        self.constrs = dict()  # constraints
        self.obj = dict()  # objective

    def modeling(self):
        """
        build DNP model
        """

        print("add variables ...")
        self.add_vars()

        print("add constraints ...")
        self.add_constraints()

        print("set objective ...")
        self.set_objective()

    def add_vars(self):
        """
        add variables
        """

        self.var_types = {
            'select_edge': {
                'lb': 0,
                'ub': 1,
                'vtype': COPT.BINARY,
                'nameprefix': 'p',
                'index': '(t, edge)'
            },
            'sku_select_edge': {
                'lb': 0,
                'ub': 1,
                'vtype': COPT.BINARY,
                'nameprefix': 'pk',
                'index': '(t, edge, k)'
            },
            'sku_flow': {
                'lb': 0,
                'ub': COPT.INFINITY,
                'vtype': COPT.CONTINUOUS,
                'nameprefix': 'w',
                'index': '(t, edge, k)'
            },
            'sku_production': {
                'lb': 0,
                'ub': COPT.INFINITY,
                'vtype': COPT.CONTINUOUS,
                'nameprefix': 'x',
                'index': '(t, plant, k)'
            },
            'open': {
                'lb': 0,
                'ub': 1,
                'vtype': COPT.BINARY,
                'nameprefix': 'y',
                'index': '(t, node)'
            },
            'sku_open': {
                'lb': 0,
                'ub': 1,
                'vtype': COPT.BINARY,
                'nameprefix': 'yk',
                'index': '(t, plant, k)'
            },
            'sku_inventory': {
                'lb': -COPT.INFINITY,
                'ub': COPT.INFINITY,
                'vtype': COPT.CONTINUOUS,
                'nameprefix': 'I',
                'index': '(t, warehouse, k)'
            },
            'sku_demand_slack': {
                'lb': 0,
                'ub': [],  # TBD
                'vtype': COPT.CONTINUOUS,
                'nameprefix': 's',
                'index': '(t, warehouse with demand / customer, k)'
            }
        }

        # generate index tuple
        idx = dict()
        for vt in self.var_types.keys():
            idx[vt] = list()

        # periods
        for t in range(self.T):
            # edges
            for e in self.network.edges:
                edge = self.network.edges[e]['object']

                # select edge (i,j) at t
                idx['select_edge'].append((t, edge))

                sku_list = edge.get_edge_sku_list(
                    t, self.full_sku_list)

                for k in sku_list:
                    # sku k select edge (i,j) at t
                    idx['sku_select_edge'].append((t, edge, k))
                    # flow of sku k on edge (i,j) at t
                    idx['sku_flow'].append((t, edge, k))

            # nodes
            for node in self.network.nodes:
                # open node i at t
                idx['open'].append((t, node))

                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:
                    if node.type == CONST.PLANT:
                        # sku k produced on node i at t
                        idx['sku_open'].append((t, node, k))
                        # amount of sku k produced on node i at t
                        idx['sku_production'].append((t, node, k))
                    elif node.type == CONST.WAREHOUSE:
                        # amount of sku k stored on node i at t
                        idx['sku_inventory'].append((t, node, k))
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            # print(node.demand)
                            idx['sku_demand_slack'].append((t, node, k))
                            self.var_types['sku_demand_slack']['ub'].append(
                                node.demand[(t, k)])
                    elif node.type == CONST.CUSTOMER:
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            idx['sku_demand_slack'].append((t, node, k))
                            self.var_types['sku_demand_slack']['ub'].append(
                                node.demand[t, k])

        # add variables
        for vt, param in self.var_types.items():
            print(f"  - {vt}")
            self.vars[vt] = self.model.addVars(
                idx[vt],
                lb=param['lb'],
                ub=param['ub'],
                vtype=param['vtype'],
                nameprefix=f"{param['nameprefix']}_"
            )

    def add_constraints(self):

        self.constr_types = {
            'flow_conservation': {
                'index': '(t, node, k)'
            },
            'open_relationship': {
                'select_edge': {
                    'index': '(t, edge, node)'
                },
                'sku_select_edge': {
                    'index': '(t, edge, k)'
                },
                'open': {
                    'index': '(t, warehouse with demand / customer)'
                },
                'sku_open': {
                    'index': '(t, node, k)'
                }
            },
            'transportation_capacity': {
                'index': '(t, edge)'
            },
            'production_capacity': {
                'index': '(t, node)'
            },
            'holding_capacity': {
                'index': '(t, node)'
            }
        }

        for constr in self.constr_types.keys():
            self.constrs[constr] = dict()

        for constr in self.constr_types['open_relationship'].keys():
            self.constrs['open_relationship'][constr] = dict()

        for t in tqdm(range(self.T)):

            # initial status and flow conservation
            self.add_constr_flow_conservation(t)

            # node status and open relationship
            self.add_constr_open_relationship(t)

            # transportation/production/holding capacity
            self.add_constr_transportation_capacity(t)
            self.add_constr_production_capacity(t)
            self.add_constr_holding_capacity(t)

    def add_constr_flow_conservation(self, t: int):

        for node in self.network.nodes:
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            for k in sku_list:

                in_edges = get_in_edges(self.network, node)
                out_edges = get_out_edges(self.network, node)

                if node.type == CONST.PLANT:
                    constr = self.model.addConstr(self.vars['sku_production'][t, node, k] - self.vars['sku_flow'].sum(
                        t, out_edges, k) == 0)
                elif node.type == CONST.WAREHOUSE:
                    fulfilled_demand = 0
                    if node.has_demand(t, k):
                        fulfilled_demand = node.demand[t,
                                                       k] - self.vars['sku_demand_slack'][t, node, k]

                    if t == 0:
                        last_period_inventory = node.initial_inventory[
                            k] if node.initial_inventory is not None and k in node.initial_inventory else 0.0
                    else:
                        last_period_inventory = self.vars['sku_inventory'][t-1, node, k]

                    constr = self.model.addConstr(self.vars['sku_flow'].sum(
                        t, in_edges, k) + last_period_inventory - self.vars['sku_flow'].sum(
                        t, out_edges, k) - fulfilled_demand == self.vars['sku_inventory'][t, node, k])

                elif node.type == CONST.CUSTOMER:
                    fulfilled_demand = node.demand[t,
                                                   k] - self.vars['sku_demand_slack'][t, node, k]
                    constr = self.model.addConstr(self.vars['sku_flow'].sum(
                        t, in_edges, k) - fulfilled_demand == 0)

                self.constrs['flow_conservation'][(t, node, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):

        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            sku_list = edge.get_edge_sku_list(
                t, self.full_sku_list)

            constr = self.model.addConstr(
                self.vars['select_edge'][t, edge] <= self.vars['open'][t, edge.start])

            self.constrs['open_relationship']['select_edge'][(
                t, edge, edge.start)] = constr

            constr = self.model.addConstr(
                self.vars['select_edge'][t, edge] <= self.vars['open'][t, edge.end])
            self.constrs['open_relationship']['select_edge'][(
                t, edge, edge.end)] = constr

            for k in sku_list:
                constr = self.model.addConstr(
                    self.vars['sku_select_edge'][t, edge, k] <= self.vars['select_edge'][t, edge])
                self.constrs['open_relationship']['sku_select_edge'][(
                    t, edge, k)] = constr

        for node in self.network.nodes:
            sku_list = node.get_node_sku_list(t, self.full_sku_list)

            if node.type == CONST.WAREHOUSE and node.has_demand(t) and len(node.demand_sku[t]) > 0:
                constr = self.model.addConstr(
                    self.vars['open'][t, node] == 1
                )
                self.constrs['open_relationship']['open'][(t, node)] = constr
            elif node.type == CONST.CUSTOMER:
                constr = self.model.addConstr(
                    self.vars['open'][t, node] == 1
                )
                self.constrs['open_relationship']['open'][(t, node)] = constr

            for k in sku_list:
                if node.type == CONST.PLANT:
                    constr = self.model.addConstr(
                        self.vars['sku_open'][t, node, k] <= self.vars['open'][t, node])

                self.constrs['open_relationship']['sku_open'][(
                    t, node, k)] = constr

        return

    def add_constr_transportation_capacity(self, t: int):

        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            # sku_list = get_edge_sku_list(edge,
            #                              t, self.full_sku_list)

            constr = self.model.addConstr(
                self.vars['sku_flow'].sum(t, edge, '*') <= edge.capacity * self.vars['select_edge'][t, edge])
            self.constrs['transportation_capacity'][(t, edge)] = constr

        return

    def add_constr_production_capacity(self, t: int):

        for node in self.network.nodes:
            if node.type == CONST.PLANT:
                constr = self.model.addConstr(
                    self.vars['sku_production'].sum(t, node, '*') <= node.production_capacity * self.vars['open'][t, node])
                self.constrs['production_capacity'][(t, node)] = constr

        return

    def add_constr_holding_capacity(self, t: int):

        for node in self.network.nodes:
            if node.type == CONST.WAREHOUSE:
                constr = self.model.addConstr(
                    self.vars['sku_inventory'].sum(t, node, '*') <= node.inventory_capacity * self.vars['open'][t, node])
                self.constrs['holding_capacity'][(t, node)] = constr

        return

    def set_objective(self):

        self.obj_types = {
            'sku_producing_cost': {
                'index': '(t, plant, k)'
            },
            'sku_holding_cost': {
                'index': '(t, warehouse, k)'
            },
            'sku_transportation_cost': {
                'index': '(t, edge)'
            },
            'unfulfill_demand_cost': {
                'index': '(t, warehouse with demand / customer, k)'
            },
            'fixed_node_cost': {
                'index': '(t, plant / warehouse, k)'
            },
            'fixed_edge_cost': {
                'index': '(t, edge, k)'
            },
            'end_inventory_cost': {
                'index': '(node, k)'
            }
        }

        for obj in self.obj_types.keys():
            self.obj[obj] = dict()

        obj = 0.0

        for t in tqdm(range(self.T)):

            obj = obj + self.cal_sku_producing_cost(t)
            obj = obj + self.cal_sku_holding_cost(t)

            if self.arg.transportation_cost:
                obj = obj + self.cal_sku_transportation_cost(t)

            obj = obj + self.cal_sku_unfulfill_demand_cost(t)

        obj = obj + self.cal_fixed_node_cost()
        obj = obj + self.cal_fixed_edge_cost()

        if self.arg.end_inventory:
            obj = obj + self.cal_end_inventory_bias_cost()

        self.model.setObjective(obj, sense=COPT.MINIMIZE)

        return

    def cal_sku_producing_cost(self, t: int):

        producing_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.PLANT:

                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:

                    node_sku_producing_cost = 0.0

                    if node.production_sku_fixed_cost is not None:
                        node_sku_producing_cost = node_sku_producing_cost + node.production_sku_fixed_cost[k] * self.vars['sku_open'][t,
                                                                                                                                      node, k]
                    if node.production_sku_unit_cost is not None:
                        node_sku_producing_cost = node_sku_producing_cost + \
                            node.production_sku_unit_cost[k] * \
                            self.vars['sku_production'][t, node, k]

                    producing_cost = producing_cost + node_sku_producing_cost

                    self.obj['sku_producing_cost'][(
                        t, node, k)] = node_sku_producing_cost

        return producing_cost

    def cal_sku_holding_cost(self, t: int):

        holding_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.WAREHOUSE:

                sku_list = node.get_node_sku_list(t, self.full_sku_list)

                for k in sku_list:

                    node_sku_holding_cost = 0.0

                    if node.holding_sku_unit_cost is not None:
                        # I_hat = max(I, 0)
                        I_hat = self.model.addVar(
                            name=f"I_hat_({t},_{node},_{k})")
                        self.model.addConstr(
                            I_hat >= self.vars['sku_inventory'][t, node, k])
                        node_sku_holding_cost = node_sku_holding_cost + \
                            node.holding_sku_unit_cost[k] * I_hat

                    holding_cost = holding_cost + node_sku_holding_cost

                    self.obj['sku_holding_cost'][(
                        t, node, k)] = node_sku_holding_cost

        return holding_cost

    def cal_sku_transportation_cost(self, t: int):

        transportation_cost = 0.0

        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            edge_transportation_cost = 0.0

            sku_list_with_fixed_transportation_cost, sku_list_with_unit_transportation_cost = edge.get_edge_sku_list_with_transportation_cost(
                t, self.full_sku_list)

            for k in sku_list_with_fixed_transportation_cost:
                if edge.transportation_sku_fixed_cost is not None and k in edge.transportation_sku_fixed_cost:
                    edge_transportation_cost = edge_transportation_cost + \
                        edge.transportation_sku_fixed_cost[k] * \
                        self.vars['sku_select_edge'][t, edge, k]

            for k in sku_list_with_unit_transportation_cost:
                if edge.transportation_sku_unit_cost is not None and k in edge.transportation_sku_unit_cost:
                    edge_transportation_cost = edge_transportation_cost + \
                        edge.transportation_sku_unit_cost[k] * \
                        self.vars['sku_flow'][t, edge, k]

            transportation_cost = transportation_cost + edge_transportation_cost

            self.obj['sku_transportation_cost'][(
                t, edge)] = edge_transportation_cost

        return transportation_cost

    def cal_sku_unfulfill_demand_cost(self, t: int):

        unfulfill_demand_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.WAREHOUSE or node.type == CONST.CUSTOMER:
                if node.has_demand(t):

                    for k in node.demand_sku[t]:
                        unfulfill_node_sku_cost = 0.0

                        if node.unfulfill_sku_unit_cost is not None:
                            unfulfill_sku_unit_cost = node.unfulfill_sku_unit_cost[(
                                t, k)]
                        else:
                            unfulfill_sku_unit_cost = 1.0

                        unfulfill_node_sku_cost = unfulfill_sku_unit_cost * \
                            self.vars['sku_demand_slack'][(t, node, k)]

                        unfulfill_demand_cost = unfulfill_demand_cost + unfulfill_node_sku_cost

                        self.obj['unfulfill_demand_cost'][(
                            t, node, k)] = unfulfill_node_sku_cost

        return unfulfill_demand_cost

    def cal_fixed_node_cost(self):

        fixed_node_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.PLANT:
                this_node_fixed_cost = node.production_fixed_cost
            elif node.type == CONST.WAREHOUSE:
                this_node_fixed_cost = node.holding_fixed_cost
            elif node.type == CONST.CUSTOMER:
                break

            y = self.model.addVar(vtype=COPT.BINARY, name=f"y_{node}")
            for t in range(self.T):
                self.model.addConstr(self.vars['open'][(t, node)] <= y)

            node_fixed_node_cost = this_node_fixed_cost * y

            fixed_node_cost = fixed_node_cost + node_fixed_node_cost

            self.obj['fixed_node_cost'][node] = node_fixed_node_cost

        return fixed_node_cost

    def cal_fixed_edge_cost(self):

        fixed_edge_cost = 0.0

        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            p = self.model.addVar(vtype=COPT.BINARY, name=f"p_{edge}")

            for t in range(self.T):
                self.model.addConstr(
                    self.vars['select_edge'][(t, edge)] <= p)

            edge_fixed_edge_cost = edge.transportation_fixed_cost * p

            fixed_edge_cost = fixed_edge_cost + edge_fixed_edge_cost

            self.obj['fixed_edge_cost'][edge] = edge_fixed_edge_cost

        return fixed_edge_cost

    def cal_end_inventory_bias_cost(self):

        end_inventory_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.WAREHOUSE and node.end_inventory is not None:

                for k in node.end_inventory.index:

                    node_end_inventory_cost = 0.0

                    end_inventory_bias = self.model.addVar(
                        name=f"end_I_bias_{node}")

                    self.model.addConstr(
                        end_inventory_bias >= node.end_inventory[k] - self.vars['sku_inventory'].sum(self.T-1, node, '*'))
                    self.model.addConstr(end_inventory_bias >= self.vars['sku_inventory'].sum(
                        self.T-1, node, '*') - node.end_inventory[k])

                    node_end_inventory_cost = node_end_inventory_cost + \
                        node.end_inventory_bias_cost * end_inventory_bias

                    end_inventory_cost = end_inventory_cost + node_end_inventory_cost

                    self.obj['end_inventory_cost'][(
                        node, k)] = node_end_inventory_cost

        return end_inventory_cost

    def solve(self):
        self.model.solve()


    def get_solution(self, data_dir: str = './'):

        # node output
        plant_sku_t_production = pd.DataFrame(index=range(len(self.vars['sku_production'])), columns=['node', 'type', 'sku', 't', 'qty'])
        warehouse_sku_t_storage = pd.DataFrame(index=range(len(self.vars['sku_inventory'])), columns=['node', 'type', 'sku', 't', 'qty'])
        node_sku_t_demand_slack = pd.DataFrame(index=range(len(self.vars['sku_demand_slack'])), columns=['node', 'type', 'sku', 't', 'qty'])
        node_t_open = pd.DataFrame(index=range(len(self.vars['open'])), columns=['node', 'type', 't', 'open'])

        # edge output
        edge_sku_t_flow = pd.DataFrame(index=range(len(self.vars['sku_flow'])), columns=['id', 'start', 'end', 'sku', 't', 'qty'])

        node_open_index = 0
        plant_index = 0
        warehouse_index = 0
        demand_slack_index = 0
        edge_index = 0

        for node in self.network.nodes:
            for t in range(self.T):
                node_t_open.iloc[node_open_index] = {'node': node.idx, 'type': node.type, 't': t, 'open': self.vars['open'][(t, node)].x}
                node_open_index += 1

                if node.type == CONST.PLANT:
                    if node.producible_sku is not None:
                        for k in node.producible_sku:
                            plant_sku_t_production.iloc[plant_index] = {'node': node.idx, 'type': node.type, 'sku': k.idx, 't': t, 'qty': self.vars['sku_production'][(t, node, k)].x}
                            plant_index += 1
            
                if node.type == CONST.WAREHOUSE:
                    sku_list = node.get_node_sku_list(t, self.full_sku_list)
                    for k in sku_list:
                            warehouse_sku_t_storage.iloc[warehouse_index] = {'node': node.idx, 'type': node.type, 'sku': k.idx, 't': t, 'qty': self.vars['sku_inventory'][(t, node, k)].x}
                            warehouse_index += 1

            if node.type != CONST.PLANT and node.demand_sku is not None:
                for t in node.demand_sku.index:
                    for k in node.demand_sku[t]:
                        node_sku_t_demand_slack.iloc[demand_slack_index] = {'node': node.idx, 'type': node.type, 'sku': k.idx, 't': t, 'qty': self.vars['sku_demand_slack'][(t, node, k)].x}
                        demand_slack_index += 1

        for e in self.network.edges:
            edge = self.network.edges[e]['object']            
            for t in range(self.T):
                edge_sku_list = edge.get_edge_sku_list(t, self.full_sku_list)
                for k in edge_sku_list:
                    edge_sku_t_flow.iloc[edge_index] = {'id': edge.idx, 'start': edge.start.idx, 'end': edge.end.idx, 'sku': k.idx, 't': t, 'qty': self.vars['sku_flow'][(t, edge, k)].x}
                    edge_index += 1
       

        plant_sku_t_production.to_csv(os.path.join(data_dir, 'plant_sku_t_production.csv'), index=False)
        warehouse_sku_t_storage.to_csv(os.path.join(data_dir, 'warehouse_sku_t_storage.csv'), index=False)
        node_sku_t_demand_slack.to_csv(os.path.join(data_dir, 'node_sku_t_demand_slack.csv'), index=False)
        node_t_open.to_csv(os.path.join(data_dir, 'node_t_open.csv'), index=False)
        edge_sku_t_flow.to_csv(os.path.join(data_dir, 'edge_sku_t_flow.csv'), index=False)


if __name__ == "__main__":
    from Param import Param
    from Entity import SKU, Node, Plant, Warehouse, Customer, Edge
    from network import constuct_network
    import pandas as pd

    param = Param()
    arg = param.arg

    sku = SKU('1')
    print(sku)

    sku_list = [sku]

    production_sku_unit_cost = pd.Series({sku: 10.0})
    plant = Plant('1', np.array([1, 1]), 10,
                  sku_list, production_sku_unit_cost=production_sku_unit_cost)
    print(plant)

    holding_sku_unit_cost = pd.Series({sku: 1.0})
    end_inventory_bias_cost = 100
    initial_inventory = pd.Series({sku: 0.0})
    end_inventory = pd.Series({sku: 0.0})

    warehouse = Warehouse('1', np.array(
        [1, 2]), 1, initial_inventory=initial_inventory, end_inventory=end_inventory, holding_sku_unit_cost=holding_sku_unit_cost, end_inventory_bias_cost=end_inventory_bias_cost)
    print(warehouse)

    demand = pd.Series({(0, sku): 5})
    demand_sku = pd.Series({0: [sku]})
    unfulfill_sku_unit_cost = pd.Series({(0, sku): 1000})
    customer = Customer('1', np.array(
        [2, 3]), demand, demand_sku, unfulfill_sku_unit_cost=unfulfill_sku_unit_cost)
    print(customer)

    nodes = [plant, warehouse, customer]

    transportation_sku_unit_cost = pd.Series({sku: 1})

    edges = [
        Edge('e1', plant, warehouse, 10,
             transportation_sku_unit_cost=transportation_sku_unit_cost),
        Edge('e2', warehouse, customer, 10,
             transportation_sku_unit_cost=transportation_sku_unit_cost)
    ]
    for e in edges:
        print(e)

    network = constuct_network(nodes, edges, sku_list)
    model = DNP(arg, network)
    model.modeling()
    model.solve()
    # model.model.write("./output/solution.sol")
    # model.model.write("./output/toy.lp")

    for t in range(arg.T):
        for edge in edges:
            temp_sku_list = edge.get_edge_sku_list(
                t, model.full_sku_list)
            for k in temp_sku_list:
                print(
                    f"Flow on {edge}: {model.vars['sku_flow'][(t, edge, k)].x}")

        print(
            f"Unfulfill {customer} demand is {model.vars['sku_demand_slack'][(t, customer, sku)].x}")
        print(
            f"Inventory {warehouse} is {model.vars['sku_inventory'][(t, warehouse, sku)].x}")
