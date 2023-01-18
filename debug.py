# %%
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import product
from tqdm import tqdm
from pylab import *
from pyecharts import options as opts
from pyecharts.charts import Sankey
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

from data_process import data_construct

# 设置
pd.options.display.notebook_repr_html = True  # 表格显示
plt.rcParams['figure.dpi'] = 1000  # 图形分辨率
sns.set_style("darkgrid")  # 图形主题
plt.rcParams['font.sans-serif'] = ['SimHei']
# %%
model_dir = '../模型数据v1/'
phase_one_dir = 'data/phase_one/'
phase_two_dir = 'data/phase_two/'
# %% md
# 阶段一结果分析
# %%
warehouse_status = pd.read_csv(phase_one_dir + 'warehouse_status.csv')
plant_to_warehouse = pd.read_csv(phase_one_dir + 'plant_to_warehouse.csv')
warehouse_to_warehouse = pd.read_csv(phase_one_dir + 'warehouse_to_warehouse.csv')
warehouse_to_customer = pd.read_csv(phase_one_dir + 'warehouse_to_customer.csv')
peak_warehouse_demand_monthly = pd.read_csv(model_dir + 'peak_warehouse_demand_monthly.csv')
wh_outstorage_capacity_monthly = pd.read_csv(model_dir + 'wh_outstorage_capacity_monthly.csv')
warehouse_capacity_monthly = pd.read_csv(model_dir + 'warehouse_capacity_monthly.csv')
consider_inventory_days = pd.read_csv(model_dir + 'consider_inventory_days.csv')
warehouse_capacity_periodly = pd.read_csv(model_dir + 'warehouse_capacity_periodly.csv')
cus_demand_periodly = pd.read_csv(model_dir + 'cus_demand_periodly.csv')
plant_info_monthly = pd.read_csv(model_dir + 'plant_info_monthly.csv')
plant_sku_df = pd.read_csv(model_dir + 'plant_sku_df.csv')
consider_init_inventory = pd.read_csv(model_dir + 'consider_init_inventory.csv')

# %%
plant_line_product = pd.read_csv(phase_two_dir + 'plant_line_product.csv')
plant_to_warehouse = pd.read_csv(phase_two_dir + 'plant_to_warehouse.csv')
warehouse_to_warehouse = pd.read_csv(phase_two_dir + 'warehouse_to_warehouse.csv')
warehouse_to_customer = pd.read_csv(phase_two_dir + 'warehouse_to_customer.csv')
tmp6 = pd.read_csv(phase_two_dir + 'warehouse_inv_change.csv')
# %%
tmp = plant_sku_df.groupby('sku').fac_id.unique().reset_index()
outsourcing_sku = tmp[tmp.fac_id.apply(lambda x: len(x) == 1)].sku.tolist()
insourcing_sku = tmp[tmp.fac_id.apply(lambda x: len(x) > 1)].sku.tolist()

target = sys.argv[1]  # 'Y000020'
# %%
from data_process import data_construct

model_dir = '../模型数据v1/'
phase_one_dir = 'data/phase_one/'
phase_two_dir = 'data/phase_two/'

model_data = data_construct(model_dir)

mapping_category = {
    "p2w": "工厂-仓库转运",
    "w2w": "仓库-仓库转运",
    "w2c": "仓库-客户转运",
    "production": "生产"
}
df_p2w = plant_to_warehouse.query('qty > 0').assign(
    row_key=lambda df: list(zip(df['start_id'], df['end_id'], df['sku'])),
    unit_cost=lambda df: df.apply(lambda row: model_data.plant_to_warehouse_cost.get(row['row_key'], 0), axis=1),
    cost=lambda df: df['qty'] * df['unit_cost'],
    category='p2w'
)
df_w2w = warehouse_to_warehouse.query('qty > 0').assign(
    row_key=lambda df: list(zip(df['start_id'], df['end_id'], df['sku'])),
    unit_cost=lambda df: df.apply(lambda row: model_data.warehouse_transfer_cost.get(row['row_key'], 0), axis=1),
    cost=lambda df: df['qty'] * df['unit_cost'],
    category='w2w'
)
df_w2c = warehouse_to_customer.query('qty > 0').assign(
    row_key=lambda df: list(zip(df['start_id'], df['end_id'], df['sku'])),
    unit_cost=lambda df: df.apply(lambda row: model_data.warehouse_to_customer_cost.get(row['row_key'], 0), axis=1),
    cost=lambda df: df['qty'] * df['unit_cost'],
    category='w2c'
)

df_transfer_cost = pd.concat([df_p2w, df_w2w, df_w2c], sort=False)

df_prod = plant_line_product.assign(
    row_key=lambda df: list(zip(df['plant_id'], df['line_id'], df['sku'])),
    unit_cost=lambda df: df.apply(lambda row: model_data.line_prod_cost.get(row['row_key'], 0)
    if row['line_id'] != 'XXXX' else 0, axis=1),
    category='prod',
    cost=lambda df: df['qty'] * df['unit_cost'],
)

df_total_sum = pd.concat(
    [df_prod.groupby(["category", "period"]).agg({"cost": sum, "qty": sum}).reset_index().assign(
        category_chs=lambda df: df['category'].apply(mapping_category.get)),
        df_transfer_cost.groupby(["category", "period"]).agg({"cost": sum, "qty": sum}).reset_index().assign(
            category_chs=lambda df: df['category'].apply(mapping_category.get))
    ])

df_total_sum.to_csv("./cost_agg.csv")
df_transfer_cost.to_csv("./cost_transfer.csv")

columns = df_transfer_cost.columns
df_cost = pd.concat(
    [df_transfer_cost, df_prod.assign(
        end_id=lambda df: df['plant_id'],
        start_id="source"
    ).filter(columns)]
)
df_cost = df_cost.groupby(['start_id', 'end_id', 'sku', 'category']).agg({
    "qty": sum,
    "cost": sum,
    "unit_cost": "first"
}).reset_index()
dfc = df_cost.query(f"sku == '{target}'")
customers = dfc.query("category in ['p2c', 'w2c']").end_id.tolist()
import networkx as nx

# %%
data = [
    *(['source', p, sku, v, 'prod'] for (p, sku), v in model_data.plant_prod_cost.items()),
    *([i, j, sku, v, 'p2w'] for (i, j, sku), v in model_data.plant_to_warehouse_cost.items()),
    *([i, j, sku, v, 'w2c'] for (i, j, sku), v in model_data.warehouse_to_customer_cost.items()),
    *([i, j, sku, v, 'w2w'] for (i, j, sku), v in model_data.warehouse_transfer_cost.items())
]
# %%
df = pd.DataFrame(
    data=data, columns=['start_id', 'end_id', 'sku', 'unit_cost', 'category']
).query(
    f"sku == '{target}'"
)
# save all possibilities
g = nx.DiGraph()
data = df.set_index(['start_id', 'end_id']).to_dict(orient='records')
edges = [(i, j, k) for i, j, k in zip(df['start_id'], df['end_id'], data)]
g.add_edges_from(edges)
path_data = []
df_query = df.set_index(['start_id', 'end_id'])
for c in tqdm(customers):
    # path = nx.algorithms.shortest_path(g, source='source', target=c, weight='unit_cost')
    paths = nx.all_simple_paths(g, source='source', target=c, cutoff=4)
    ct = 0
    for path in paths:
        path_name = "-".join(path)
        edges = list(zip(path[:-1], path[1:]))
        costs = {df_query['category'][e]: df_query['unit_cost'][e] for e in edges}
        path_data.append(
            dict(
                start='s',
                end=c,
                **costs,
                path=path_name,
            )
        )
        ct += 1

df_final = pd.DataFrame.from_dict(path_data).fillna(0).assign(
    cost=lambda df: df['prod'] + df['p2w'] + df['w2w'] + df['w2c']
)
df_final.to_excel(f"diagnostics-{target}-all.xlsx")
