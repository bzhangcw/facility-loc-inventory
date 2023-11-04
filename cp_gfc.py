import gurobipy as grb
import numpy as np
import const
from utils import *

def query_fractional_inter_edges(df_edges):
    df = df_edges.query("end.str.startswith('T')").query("bool_lb_vio == True")
    return df
def check_inventory_n1(model, j):
    is_n1 = False
    inv_end = 0
    edge_open = 0
    if j.start.type == const.WAREHOUSE:
        # if j.end.type == const.WAREHOUSE:
        if str(j.end).startswith('F'):
            is_n1 = True
            t = str(j.end).replace('Feak','').replace(str(j.start),'').replace('T','')
            inv_end = model.variables["sku_inventory"].sum(int(t), j.start, "*")
            edge_open = model.variables["open"][int(t), j.start]
    return is_n1, inv_end, edge_open
def check_inventory_n2(model, j):
    is_n2 = False
    last_period_inventory = 0
    edge_open = 0
    if j.end.type == const.WAREHOUSE:
        # if j.start.type == const.WAREHOUSE:
        if str(j.start).startswith('F'):
            is_n2 = True
            t = str(j.start).replace('Feak','').replace(str(j.end),'').replace('T','')
            if int(t) != -1:
                last_period_inventory = model.variables["sku_inventory"].sum(t, j.end, "*")
            elif len(j.end.initial_inventory):
                last_period_inventory = j.end.initial_inventory.sum()
            edge_open = model.variables["open"][int(t)+1, j.end]
    return is_n2, last_period_inventory, edge_open
def replace_x_y(x,y,t,j,model,version):
    if version == 1:
        is_, x_, y_ = check_inventory_n1(model, j)
    if version == 2:
        is_, x_, y_ = check_inventory_n2(model, j)
    if is_:
        if type(x_) is not float and type(x_) is not int:
            temp_x = x_.getValue()
            temp_y = y_.x
        else:
            temp_x = 0
            temp_y = 1
        # temp_y = y_.getValue()
    elif type(x.sum(t, j, "*")) is not float:
        temp_x = x.sum(t, j, "*").getValue()
        temp_y = y[t, j].x
    else:
        temp_x = 0
        temp_y = y[t, j].x
    return temp_x, temp_y

def replace_variable(x,y,t,j,model,version):
    if version == 1:
        is_, x_, y_ = check_inventory_n1(model, j)
    if version == 2:
        is_, x_, y_ = check_inventory_n2(model, j)
    if is_:
        if type(x_) is not float and type(x_) is not int:
            temp_x = x_
            temp_y = y_
        else:
            temp_x = 0
            temp_y = 1
        # temp_x = x_
        # temp_y = y_
    elif type(x.sum(t, j, "*")) is not float:
        temp_x = x.sum(t, j, "*")
        temp_y = y[t, j]
    else:
        temp_x = 0
        temp_y = 0
    return temp_x, temp_y
def seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False):
    # separation problem
    md = grb.Model("separation")
    idx1 = list(range(N1.__len__()))
    idx2 = list(range(N2.__len__()))
    alph = md.addVars(idx1, vtype=grb.GRB.BINARY)
    beta = md.addVars(idx1, vtype=grb.GRB.BINARY)
    gamm = md.addVars(idx2, vtype=grb.GRB.BINARY)
    delt = md.addVars(idx2, vtype=grb.GRB.BINARY)
    lbd = md.addVar(vtype=grb.GRB.CONTINUOUS, lb=-1e-3)
    md.addConstrs((alph[j] + beta[j] <=1) for j in idx1)
    md.addConstrs((gamm[j] + delt[j] <=1) for j in idx2)
    md.addConstr(
        sum(alph[idx] * j.capacity + beta[idx] * j.variable_lb for idx, j in enumerate(N1))
        - sum(delt[idx] * j.capacity + gamm[idx] * j.variable_lb for idx, j in enumerate(N2))
        - d
        - lbd == 0
    )
    lp1 = md.addVars(idx1)
    up1 = md.addVars(idx1)
    lp2 = md.addVars(idx2)
    up2 = md.addVars(idx2)
    ml1 = md.addVars(idx1, lb=-grb.GRB.INFINITY)
    mu2 = md.addVars(idx2, lb=-grb.GRB.INFINITY)

    for idx, j in enumerate(N1):
        md.addConstr(lp1[idx] == grb.max_(lbd, constant=j.variable_lb))
        md.addConstr(up1[idx] == grb.max_(lbd, constant=j.capacity))
        md.addConstr(ml1[idx] == grb.min_(lbd, constant=j.variable_lb))

    for idx, j in enumerate(N2):
        md.addConstr(lp2[idx] == grb.max_(lbd, constant=j.variable_lb))
        md.addConstr(up2[idx] == grb.max_(lbd, constant=j.capacity))
        md.addConstr(mu2[idx] == grb.min_(lbd, constant=j.capacity))

    e1 = 0
    e2 = 0
    for idx, j in enumerate(N1):
        temp_x, temp_y = replace_x_y(x, y, t, j, model, version=1)
        e1 += (temp_x + (up1[idx] - lbd) * (1 - temp_y)) * alph[idx] + (lp1[idx] - lbd + ml1[idx] * temp_y) * beta[idx]
    for idx, j in enumerate(N2):
        temp_x, temp_y = replace_x_y(x, y, t, j, model, version=2)
        e2 += (temp_x + j.variable_lb * (1 - temp_y)) * gamm[idx] + j.capacity * delt[idx]

    v1 = md.addVars(idx2) # correspond to L2\R
    v2 = md.addVars(idx2) # correspond to L2^R
    v3 = md.addVars(idx2) # correspond to N2\L2\C2
    v4 = md.addVars(idx2)
    for idx, j in enumerate(N2):
        temp_x, temp_y = replace_x_y(x, y, t, j, model, version=2)
        md.addConstr(
            v1[idx] == (temp_x - (lp2[idx] - lbd) * temp_y) * (1-gamm[idx]-delt[idx]))
        md.addConstr(
            v2[idx] == mu2[idx] * temp_y * (1-gamm[idx]-delt[idx])
        )
        md.addConstr(
            v3[idx] == temp_x * (1-gamm[idx]-delt[idx])
        )
        md.addConstr(
            v4[idx] == grb.min_([v1[idx], v2[idx], v3[idx]])
        )
    md.setObjective(e1 - e2 - grb.quicksum(v4) - 0, sense=grb.GRB.MAXIMIZE)
    md.setParam("NonConvex", 2)
    md.setParam("MIPGap", 0.01)
    md.optimize()
    if dump:
        md.write("md.mps")
    # print('Write mps')
    C1dR = {e for idx, e in enumerate(N1) if round(alph[idx].x) == 1}
    C1iR = {e for idx, e in enumerate(N1) if round(beta[idx].x) == 1}
    C2dR = {e for idx, e in enumerate(N2) if round(gamm[idx].x) == 1}
    C2iR = {e for idx, e in enumerate(N2) if round(delt[idx].x) == 1}
    L2C = {idx for idx, e in enumerate(N2) if (gamm[idx].x + delt[idx].x) == 0}
    l2values = np.array(
        [[v1[idx].x, v2[idx].x, v3[idx].x] for idx, e in enumerate(N2)]
    )
    vals = l2values.argmin(axis=1)
    L2iR = set()
    L2dR = set()
    ow = set()
    for idx, e in enumerate(N2):
        if idx in L2C:
            if l2values[idx][0] ==  l2values[idx][1]:
                L2iR.add(e)
            elif vals[idx] == 0:
                L2dR.add(e)
            elif vals[idx] == 1:
                L2iR.add(e)
            elif vals[idx] == 2:
                ow.add(e)
    subsets = (
        C1dR, C1iR, C2dR, C2iR,
        L2dR, L2iR, ow
    )
    return md.objval, lbd.x, subsets


# get corollary 3 type flow
def eval_cut_c3(model, x, y, t, d, *subsets, **kwargs):
    # 用Corollary 3的有效不等式加进去
    C1dR, C1iR, C2dR, C2iR, \
        L2dR, L2iR, ow, *_ = subsets
    lbd = lbdc = (
        sum(j.variable_lb for j in C1iR) 
        + sum(j.capacity for j in C1dR) 
        - sum(j.capacity for j in C2iR)
        - sum(j.variable_lb for j in C2dR)
        - d
    )
    if kwargs.get("lbdv") is not None:
        if not (abs(lbdc - kwargs.get("lbdv")) < 1e-1):
            print(lbdc, kwargs.get("lbdv"), lbdc - kwargs.get("lbdv"))
            print("this is unmatched with separation model")

            
    if lbd > 0:
        e1 = 0
        for j in C1dR:
            x_var, y_var = replace_variable(x, y, t, j, model, version=1)
            # e1 += x.sum(t, j, "*") + max(j.capacity - lbd, 0) * (1 - y[t, j])
            e1 += x_var + max(j.capacity - lbd, 0) * (1 - y_var)

        # e1 = (
        #     x.sum(t, C1dR, "*")
        #     + sum(max(j.capacity - lbd, 0) * (1 - y[t, j])
        #           for j in C1dR)
        # )
        e2 = 0
        for j in C1iR:
            _, y_var = replace_variable(x, y, t, j, model, version=1)
            # e2 += max(j.variable_lb - lbd, 0) + min(j.variable_lb, lbd) * y_var
            l = max(j.variable_lb - lbd, 0)
            m = min(j.variable_lb, lbd) * y_var
            e2 += l + m
        # e2 = (
        #     sum((max(j.variable_lb - lbd, 0) + min(j.variable_lb, lbd) * y[t, j])
        #
        #         for j in C1iR)
        #
        # )

        # e3, e4, e5 = 0 if set R = universe
        e3 = 0
        for j in C2dR:
            x_var, y_var = replace_variable(x, y, t, j, model, version=1)
            e3 += x_var + j.variable_lb * (1 - y_var)
        # e3 = (
        #    x.sum(t, C2dR, "*")
        #     + sum(j.variable_lb * (1 - y[t, j]) for j in C2dR)
        # )

        e4 = sum(j.capacity for j in C2iR)
        e5 = 0
        for j in L2dR:
            x_var, y_var = replace_variable(x, y, t, j, model, version=2)
            e5 += x_var + max(j.variable_lb - lbd, 0) * y_var
        # e5 = (
        #     x.sum(t, L2dR, "*")
        #     - sum(max(j.variable_lb - lbd, 0) * y[t, j] for j in L2dR)
        # )
        e6 = 0
        for j in L2iR:
            _, y_var = replace_variable(x, y, t, j, model, version=2)
            e6 += min(j.capacity, lbd) * y_var
        # e6 = (
        #     sum(min(j.capacity, lbd) * y[t, j] for j in L2iR)
        # )
        # e7 = (
        #     x.sum(t, ow, "*")
        # )
        e7 = 0
        for j in ow:
            x_var, _ = replace_variable(x, y, t, j, model, version=2)
            e7 += x_var
        expr = e1 + e2 - e3 - e4 - e5 - e6 - e7 - d
        try:
            cut_value = expr.getValue()
        except:
            cut_value = 0
        bool_voilate = cut_value > 0
        # print(e3)
        return expr, cut_value, lbd, bool_voilate
    return 0, 0, 0, False



