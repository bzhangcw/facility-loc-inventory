import coptpy as cp
from coptpy import COPT
import CONST
from utils import get_edge_sku_list, get_node_sku_list
import networkx as nx
import numpy as np
import pandas as pd
from typing import List
import argparse
from Entity import SKU
from tqdm import tqdm


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

        var_types = {
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
                'vtype': COPT.BINARY,
                'nameprefix': 'I',
                'index': '(t, warehouse, k)'
            },
            'sku_demand_slack': {
                'lb': 0,
                'ub': [],  # TBD
                'vtype': COPT.BINARY,
                'nameprefix': 's',
                'index': '(t, warehouse with demand / customer, k)'
            }
        }

        # generate index tuple
        idx = dict()
        for vt in var_types.keys():
            idx[vt] = list()

        # periods
        for t in range(self.T):
            # edges
            for e in self.network.edges:
                edge = self.network.edges[e]['object']

                # select edge (i,j) at t
                idx['select_edge'].append((t, edge))

                sku_list = get_edge_sku_list(edge,
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

                sku_list = get_node_sku_list(
                    node, t, self.full_sku_list)

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
                            print(node.demand)
                            idx['sku_demand_slack'].append((t, node, k))
                            var_types['sku_demand_slack']['ub'].append(
                                node.demand[t, k])
                    elif node.type == CONST.CUSTOMER:
                        # demand of sku k not fulfilled on node i at t
                        if node.has_demand(t, k):
                            idx['sku_demand_slack'].append((t, node, k))
                            var_types['sku_demand_slack']['ub'].append(
                                node.demand[t, k])

        # add variables
        for vt, param in var_types.items():
            print(f"  - {vt}")
            self.vars[vt] = self.model.addVars(
                idx[vt],
                lb=param['lb'],
                ub=param['ub'],
                vtype=param['vtype'],
                nameprefix=f"{param['nameprefix']}_"
            )

    def add_constraints(self):

        constr_types = {
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

        self.constrs['flow_conservation'] = dict()

        for node in self.network.nodes:
            sku_list = get_node_sku_list(node, t, self.full_sku_list)

            for k in sku_list:
                constr_name = f"flow_con_{t}_{node}_{k}"

                in_edges = [e[2]['object']
                            for e in list(self.network.in_edges(node, data=True))]
                out_edges = [e[2]['object']
                             for e in list(self.network.out_edges(node, data=True))]

                if node.type == CONST.PLANT:
                    constr = self.model.addConstr(self.vars['sku_production'][t, node, k] - self.vars['sku_flow'].sum(
                        t, out_edges, k) == 0, name=constr_name)
                elif node.type == CONST.WAREHOUSE:
                    fulfilled_demand = 0
                    if node.has_demand(t, k):
                        fulfilled_demand = node.demand[t,
                                                       k] - self.vars['sku_demand_slack'][t, node, k]

                    if t == 0:
                        initial_inventory = node.initial_inventory[
                            k] if node.initial_inventory is not None else 0.0
                        constr = self.model.addConstr(self.vars['sku_flow'].sum(
                            t, in_edges, k) + initial_inventory - self.vars['sku_flow'].sum(
                            t, out_edges, k) - fulfilled_demand == self.vars['sku_inventory'][t, node, k], name=constr_name)
                    else:
                        constr = self.model.addConstr(self.vars['sku_flow'].sum(
                            t, in_edges, k) + self.vars['sku_inventory'][t-1, node, k] - self.vars['sku_flow'].sum(
                            t, out_edges, k) - fulfilled_demand == self.vars['sku_inventory'][t, node, k], name=constr_name)
                elif node.type == CONST.CUSTOMER:
                    fulfilled_demand = node.demand[t,
                                                   k] - self.vars['sku_demand_slack'][t, node, k]
                    constr = self.model.addConstr(self.vars['sku_flow'].sum(
                        t, in_edges, k) - fulfilled_demand == 0, name=constr_name)

                self.constrs['flow_conservation'][(t, node, k)] = constr

        return

    def add_constr_open_relationship(self, t: int):
        self.constrs['open_relationship'] = dict()
        self.constrs['open_relationship']['select_edge'] = dict()
        self.constrs['open_relationship']['sku_select_edge'] = dict()
        self.constrs['open_relationship']['open'] = dict()
        self.constrs['open_relationship']['sku_open'] = dict()

        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            sku_list = get_edge_sku_list(edge,
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
            sku_list = get_node_sku_list(
                node, t, self.full_sku_list)

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

        self.constrs['transportation_capacity'] = dict()

        for e in self.network.edges:
            edge = self.network.edges[e]['object']

            # sku_list = get_edge_sku_list(edge,
            #                              t, self.full_sku_list)

            constr = self.model.addConstr(
                self.vars['sku_flow'].sum(t, edge, '*') <= edge.capacity * self.vars['select_edge'][t, edge])
            self.constrs['transportation_capacity'][(t, edge)] = constr

        return

    def add_constr_production_capacity(self, t: int):

        self.constrs['production_capacity'] = dict()

        for node in self.network.nodes:
            if node.type == CONST.PLANT:
                constr = self.model.addConstr(
                    self.vars['sku_production'].sum(t, node, '*') <= node.production_capacity * self.vars['open'][t, node])
                self.constrs['production_capacity'][(t, node)] = constr

        return

    def add_constr_holding_capacity(self, t: int):

        self.constrs['holding_capacity'] = dict()

        for node in self.network.nodes:
            if node.type == CONST.WAREHOUSE:
                constr = self.model.addConstr(
                    self.vars['sku_inventory'].sum(t, node, '*') <= node.inventory_capacity * self.vars['open'][t, node])
                self.constrs['holding_capacity'][(t, node)] = constr

        return

    def set_objective(self):

        obj = 0.0

        for t in tqdm(range(self.T)):

            self.obj[t] = dict()

            obj = obj + self.cal_sku_producing_cost(t)
            obj = obj + self.cal_sku_holding_cost(t)
            obj = obj + self.cal_sku_transportation_cost(t)
            obj = obj + self.cal_sku_unfulfill_demand_cost(t)

        obj = obj + self.cal_fixed_node_cost()
        obj = obj + self.cal_fixed_edge_cost()

        obj = obj + self.cal_end_inventory_bias_cost()

        self.model.setObjective(obj, sense=COPT.MINIMIZE)

        return

    def cal_sku_producing_cost(self, t: int):

        self.obj['sku_producing_cost'] = dict()

        producing_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.PLANT:

                sku_list = get_node_sku_list(
                    node, t, self.full_sku_list)

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

        self.obj['sku_holding_cost'] = dict()

        holding_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.WAREHOUSE:

                sku_list = get_node_sku_list(
                    node, t, self.full_sku_list)

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

        self.obj['sku_transportation_cost'] = dict()

        transportation_cost = 0.0

        for e in self.network.edges:
            edge = self.network.edges[e]['object']
            sku_list = get_edge_sku_list(edge,
                                         t, self.full_sku_list)
            for k in sku_list:

                edge_sku_transportation_cost = 0.0

                if edge.transportation_sku_fixed_cost is not None:
                    edge_sku_transportation_cost = edge_sku_transportation_cost + edge.transportation_sku_fixed_cost[k] * self.vars['sku_select_edge'][t,
                                                                                                                                                       edge, k]
                if edge.transportation_sku_unit_cost is not None:
                    edge_sku_transportation_cost = edge_sku_transportation_cost + \
                        edge.transportation_sku_unit_cost[k] * \
                        self.vars['sku_flow'][t, edge, k]

                transportation_cost = transportation_cost + edge_sku_transportation_cost

                self.obj['sku_transportation_cost'][(
                    t, edge, k)] = edge_sku_transportation_cost

        return transportation_cost

    def cal_sku_unfulfill_demand_cost(self, t: int):

        self.obj['unfulfill_demand_cost'] = dict()

        unfulfill_demand_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.WAREHOUSE or node.type == CONST.CUSTOMER:
                if node.has_demand(t):

                    for k in node.demand_sku[t]:
                        unfulfill_sku_unit_cost = 0.0

                        if node.unfulfill_sku_unit_cost is not None:
                            unfulfill_sku_unit_cost = unfulfill_sku_unit_cost + \
                                node.unfulfill_sku_unit_cost[(
                                    t, k)] * self.vars['sku_demand_slack'][(t, node, k)]

                        unfulfill_demand_cost = unfulfill_demand_cost + unfulfill_sku_unit_cost

                        self.obj['unfulfill_demand_cost'][(
                            t, node, k)] = unfulfill_sku_unit_cost

        return unfulfill_demand_cost

    def cal_fixed_node_cost(self):

        self.obj['fixed_node_cost'] = dict()

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

        self.obj['fixed_edge_cost'] = dict()

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

        self.obj['end_inventory_cost'] = dict()

        end_inventory_cost = 0.0

        for node in self.network.nodes:

            if node.type == CONST.WAREHOUSE and node.end_inventory is not None:

                sku_list = get_node_sku_list(
                    node, self.T-1, self.full_sku_list)

                for k in sku_list:

                    node_end_inventory_cost = 0.0

                    end_inventory_bias = self.model.addVar(
                        lb=-COPT.INFINITY, name=f"end_I_bias_{node}")

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
    plant = Plant('1', np.array([1, 1]), 1,
                  sku_list, production_sku_unit_cost=production_sku_unit_cost)
    print(plant)

    holding_sku_unit_cost = pd.Series({sku: 1.0})
    end_inventory_bias_cost = 100
    initial_inventory = pd.Series({sku: 0.0})
    end_inventory = pd.Series({sku: 0.0})

    warehouse = Warehouse('1', np.array(
        [1, 2]), 1, initial_inventory=initial_inventory, end_inventory=end_inventory, holding_sku_unit_cost=holding_sku_unit_cost, end_inventory_bias_cost=end_inventory_bias_cost)
    print(warehouse)

    demand = pd.Series({(0, sku): 1})
    demand_sku = pd.Series({0: [sku]})
    unfulfill_sku_unit_cost = pd.Series({(0, sku): 1000})
    customer = Customer('1', np.array(
        [2, 3]), demand, demand_sku, unfulfill_sku_unit_cost=unfulfill_sku_unit_cost)
    print(customer)

    nodes = [plant, warehouse, customer]

    edges = [
        Edge('e1', plant, warehouse, 10),
        Edge('e2', warehouse, customer, 10)
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
            temp_sku_list = get_edge_sku_list(edge,
                                              t, model.full_sku_list)
            for k in temp_sku_list:
                print(
                    f"Flow on {edge}: {model.vars['sku_flow'][(t, edge, k)].x}")

        print(
            f"Unfulfill {customer} demand is {model.vars['sku_demand_slack'][(t, customer, sku)].x}")
        print(
            f"Inventory {warehouse} is {model.vars['sku_inventory'][(t, warehouse, sku)].x}")
