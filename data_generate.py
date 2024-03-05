import pandas as pd
import numpy as np
from config.param import Param
from geopy.distance import geodesic
import random
from more_itertools import chunked

loc = np.array([0, 0])
np.random.seed(0)
param = Param()
arg = param.arg
num_customers = 1456
num_skus = arg.num_skus
num_periods = arg.num_periods

customers = [f"C{i + 1:04d}" for i in range(num_customers)]
skus = [f"SKU{i + 1:04d}" for i in range(1, num_skus + 1)]

demand_data = []
if arg.demand_type == 1:
    for period in range(num_periods):
        for customer in customers:
            for sku in skus:
                if int(sku[-4:]) % 3 == 0:
                    demand = random.randint(400, 500) if period % 4 == 0 else random.randint(100, 300)
                elif int(sku[-4:]) % 2 == 0:
                    demand = random.randint(1, 100)
                else:
                    demand = random.randint(200, 500)
                demand_data.append([customer, sku, demand, period])
elif arg.demand_type == 2:
    for period in range(num_periods):
        for customer in customers:
            for sku in skus:
                if int(sku[-4:]) % 4 == 0:
                    demand = random.randint(100, 600) if period % 3 == 0 else random.randint(50, 150)
                elif int(sku[-4:]) % 5 == 0:
                    demand = random.randint(1, 500) if period % 2 == 0 else random.randint(100, 200)
                else:
                    demand = random.randint(50, 20000)
                demand_data.append([customer, sku, demand, period])

elif arg.demand_type == 3:
    for period in range(num_periods):
        for customer in customers:
            for sku in skus:
                if int(sku[-4:]) % 4 == 0:
                    demand = random.randint(100, 6000) if period % 3 == 0 else random.randint(50, 150)
                elif int(sku[-4:]) % 5 == 0:
                    demand = random.randint(1, 500) if period % 2 == 0 else random.randint(100, 2000)
                else:
                    demand = random.randint(50, 200)
                demand_data.append([customer, sku, demand, period])

demand_df = pd.DataFrame(demand_data, columns=["id", "sku", "demand", "time"])
demand_df.to_csv('data/data_generate/demand_sku_time.csv')
sku_df = pd.DataFrame(skus)
sku_df.to_csv('data/data_generate/sku.csv', header=["id"])
