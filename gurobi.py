from gurobipy import *
import gurobipy as gp
from coptpy import *
import coptpy as cp
model = cp.read('new_data_random_7_8@8@1.mps')
cp.solve()
model.optimize()
_dual_vars = model.getDuals()
print(_dual_vars)