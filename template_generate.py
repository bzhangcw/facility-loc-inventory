from ncg.np_cg import *
import pandas as pd
from coptpy import COPT
from gurobipy import GRB
from instance_generator import *
from config.param import Param
from dnp_model import DNP
from slim_cg.slim_cg import NetworkColumnGenerationSlim as NCS
from slim_cg.slim_rmp_model import DNPSlim
import shutil
import pickle
import folium
from datetime import datetime

def template_generate(data_dir, num_periods, demand_type):
    demand_data = []
    customer_path = data_dir + 'node_info/customer.csv'
    customer_df = pd.read_csv(customer_path)
    customer_list = customer_df['id'].to_list()
    sku_path = data_dir + 'sku.csv'
    sku_df = pd.read_csv(sku_path)
    sku_list = sku_df['id'].to_list()
    base_folder = data_dir.split('/')[-2]
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    if base_folder == 'us':
        base_folder = 'us_generate'
    new_folder = 'data/' + base_folder + '_' + timestamp + '/'
    shutil.copytree(data_dir, new_folder)

    if demand_type == 1:
        for period in range(num_periods):
            for customer in customer_list:
                for sku in sku_list:
                    if int(sku[-4:]) % 3 == 0:
                        demand = int(np.random.normal(400, 5)) if period % 3 == 0 else int(np.random.normal(100,10))
                    elif int(sku[-4:]) % 2 == 0:
                        demand = np.random.randint(1, 800)
                    else:
                        demand = np.random.randint(200, 2000) if period % 10 == 0 else int(np.random.normal(100,2))
                    demand_data.append([customer, sku, demand, period])
    elif demand_type == 2:
        for period in range(num_periods):
            for customer in customer_list:
                for sku in sku_list:
                    if int(sku[-4:]) % 3 == 0:
                        demand = int(np.random.normal(2000, 10)) if period % 8 == 0 else np.random.randint(10, 300)
                    elif int(sku[-4:]) % 2 == 0:
                        demand = np.random.randint(1, 400)
                    else:
                        demand = int(np.random.normal(1000, 20)) if period % 15 == 0 else int(np.random.normal(100, 1))
                    demand_data.append([customer, sku, demand, period])

    elif demand_type == 3:
        for period in range(num_periods):
            for customer in customer_list:
                for sku in sku_list:
                    if int(sku[-4:]) % 2 == 0:
                        demand = int(np.random.normal(3000, 100)) if period % 9 == 0 else int(np.random.normal(100, 10))
                    else:
                        demand = np.random.randint(1, 200)
                    demand_data.append([customer, sku, demand, period])

    elif demand_type == 4:
        for period in range(num_periods):
            for customer in customer_list:
                for sku in sku_list:
                    if int(sku[-4:]) % 5 == 0:
                        demand = np.random.randint(1500, 3000) if period % 7 == 0 else np.random.randint(500, 800)
                    elif int(sku[-4:]) % 4 == 0:
                        demand = int(np.random.normal(600, 50))
                    elif int(sku[-4:]) % 3 == 0:
                        demand = np.random.randint(1, 400)
                    elif int(sku[-4:]) % 2 == 0:
                        demand = int(np.random.normal(500, 10)) if period % 5 == 0 else int(np.random.normal(100, 5))
                    else:
                        demand = np.random.randint(300, 600)
                    demand_data.append([customer, sku, demand, period])

    elif demand_type == 5:
        for period in range(num_periods):
            for customer in customer_list:
                for sku in sku_list:
                    if int(sku[-4:]) % 6 == 0:
                        demand = np.random.randint(100, 3000) if period % 3 == 0 else np.random.randint(50, 150)
                    elif int(sku[-4:]) % 4 == 0:
                        demand = np.random.randint(400, 1000) if period % 2 == 0 else np.random.randint(100, 300)
                    else:
                        demand = np.random.randint(50, 200)
                    demand_data.append([customer, sku, demand, period])

    demand_df = pd.DataFrame(demand_data, columns=["id", "sku", "demand", "time"])
    demand_dir = new_folder + 'demand_sku_time.csv'
    demand_df.to_csv(demand_dir)
    return new_folder


def demonstrate_graph(pickle_path):
    G = pickle.load(open(pickle_path, 'rb'))
    m = folium.Map(location=[40, -100], zoom_start=3)
    for node in G.nodes:
        if str(node).startswith('C'):
            color = 'red'
        elif str(node).startswith('T'):
            color = 'blue'
        else:
            color = 'green'
        folium.Marker(location=node.location, popup=str(node), icon=folium.Icon(color=color)).add_to(m)

    for edge in G.edges:
        folium.PolyLine(locations=[edge[0].location, edge[1].location], color='blue').add_to(m)
    save_path = pickle_path + '.html'
    m.save(save_path)
