# %%
import pandas as pd
import numpy as np
from collections import namedtuple
from itertools import product
import itertools
from tqdm import tqdm
from pylab import *
from pyecharts import options as opts
from pyecharts.charts import Sankey
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys

from data_process import data_construct


import sys

if len(sys.argv) < 3:
    print(
    """usage:
        choose sku: e.g., Y000170
        choose sparsity: T (default) if you only analyze active paths (appeared in the model)
            else full
    1. cmd: 
        python *.py <sku_name> <sparse?>
    2. ipython: 
        # invoke ipython first
        In [..]: %run -i debug_this_result.py <sku_name> <sparse?>
    """
    )
    exit(0)

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

# %%
from data_process import data_construct

model_dir = '../模型数据v1/'
phase_one_dir = 'data/phase_one/'
phase_two_dir = 'data/phase_two/'
# %%
# weird_pairs = {('T0015', 'T0030'), 
#                ('T0015', 'T0003'), ('T0001', 'T0014'), ('T0026', 'T0040')}

model_data = data_construct(model_dir)
ld = model_data.warehouse_df.query("fac_type == 'level_1_inner'").fac_id.tolist()
weird_pairs = set(list(itertools.product(ld,ld)))

mapping_category = {
    "p2w": "工厂-仓库转运",
    "w2w": "仓库-仓库转运",
    "w2c": "仓库-客户转运",
    "prod": "生产"
}
df_p2w = plant_to_warehouse.assign(
    row_key=lambda df: list(zip(df['start_id'], df['end_id'], df['sku'])),
    unit_cost=lambda df: df.apply(lambda row: model_data.plant_to_warehouse_cost.get(row['row_key'], 0), axis=1),
    cost=lambda df: df['qty'] * df['unit_cost'],
    category='p2w'
)
df_w2w = warehouse_to_warehouse.assign(
    row_key=lambda df: list(zip(df['start_id'], df['end_id'], df['sku'])),
    unit_cost=lambda df: df.apply(lambda row: model_data.warehouse_transfer_cost.get(row['row_key'], 0), axis=1),
    cost=lambda df: df['qty'] * df['unit_cost'],
    category='w2w'
)
df_w2c = warehouse_to_customer.assign(
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
}).reset_index().assign(
    bool_is_weird=lambda df: 
    df.apply(lambda row: (row['start_id'], row['end_id']) in model_data.weird_pairs, axis=1)
)

second_wh = set(model_data.warehouse_df.query('fac_type=="level_2"').fac_id.tolist())


def has_2nd(path):
    for p in path:
        if p in second_wh:
            return 1
    return 0


import networkx as nx




sku_name = sys.argv[1]
sparsity = sys.argv[2]
if sku_name == 'all':
    lsa = df_cost.sku.unique()
else:
    lsa = [sku_name]
path_data = []
for sku in tqdm(lsa):
    df = df_cost.query(f"sku == '{sku}'") if sparsity == "F" else df_cost.query(f"sku == '{sku}' and qty > 0")
    # customers = df.query("category in ['p2c', 'w2c']").end_id.tolist()
    customers = df.end_id.tolist()

    g = nx.DiGraph()
    data = df.set_index(['start_id', 'end_id']).to_dict(orient='records')
    edges = [(i, j, k) for i, j, k in zip(df['start_id'], df['end_id'], data)]
    g.add_edges_from(edges)

    df_query = df.set_index(['start_id', 'end_id'])
    for c in customers:
        # path = nx.algorithms.shortest_path(g, source='source', target=c, weight='unit_cost')
        try:
            paths = nx.all_simple_paths(g, source='source', target=c, cutoff=4)
            ct = 0
            for path in paths:
                path_name = "-".join(path)
                edges = list(zip(path[:-1], path[1:]))
                costs = {df_query['category'][e]: df_query['unit_cost'][e] for e in edges}
                qty = min(df_query['qty'][e] for e in edges)
                path_data.append(
                    dict(
                        start='s',
                        end=c,
                        **costs,
                        path=path_name,
                        length=edges.__len__(),
                        sku=sku,
                        has_2nd=has_2nd(path),
                        pairs=(path[-2], path[-3]) if (edges.__len__() == 4) else "",
                        is_weird=(edges.__len__() == 4) and (
                                    ((path[-2], path[-3]) in weird_pairs) or ((path[-3], path[-2]) in weird_pairs)),
                        key=(path[-3], path[-2], sku),
                        qty=qty
                    )
                )
                ct += 1
        except:
            pass

df_final = pd.DataFrame.from_dict(path_data).fillna(0).assign(
    cost=lambda df: sum(df[kk] for kk in ['prod', 'p2w', 'w2w', 'w2c'] if kk in df.columns)
)

#
df_final.to_excel(f"diagnostics-this-{sku_name}.xlsx")
df_total_sum.to_csv("./cost_agg.csv")
df_transfer_cost.to_csv("./cost_transfer.csv")
df_cost.to_csv("./cost_analysis.csv")