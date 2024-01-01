import numpy as np
import pandas as pd
from itertools import product

import utils as utils
from dnp_model import DNP
from config.network import construct_network
from ncg.np_cg import *
from config.param import Param

data_basic = "data/basic.xlsx"
df1 = pd.read_excel(data_basic, sheet_name="wc")
df2 = pd.read_excel(data_basic, sheet_name="pw")
df3 = pd.read_excel(data_basic, sheet_name="ww")
print(df1)
print("o", list(product(df1["w"], df1["c"], df1["sku"])))
wc = pd.DataFrame(list(product(df1["w"], df1["c"], df1["sku"])))
wc.columns = ["w", "c", "sku"]
print(wc)
# wc.dropna(inplace=True)
wc.to_csv("data/wc.csv", index=False)

# pw = pd.DataFrame(list(product(df2['p'], df2['w'], df2['sku'])))
# pw.columns = ['p', 'w', 'sku']
# pw.dropna(inplace=True)
# pw.to_csv('data/pw.csv', index=False)

# ww = pd.DataFrame(list(product(df3['w'], df3['w'], df3['sku'])))
# ww.columns = ['w', 'w', 'sku']
# ww.dropna(inplace=True)
# ww.to_csv('data/ww.csv', index=False)
