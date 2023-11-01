import gurobipy as grb
import numpy as np

from utils import *

def query_fractional_inter_edges(df_edges):
    df = df_edges.query("bool_lb_vio == True")
    return df

def seperation_gcf(model, x, y, t, N1, N2, d, dump=False, verbose=False):
    # separation problem
    md = grb.Model("separation")
    idx1 = list(range(N1.__len__()))
    idx2 = list(range(N2.__len__()))
    alph = md.addVars(idx1, vtype=grb.GRB.BINARY)
    beta = md.addVars(idx1, vtype=grb.GRB.BINARY)
    gamm = md.addVars(idx2, vtype=grb.GRB.BINARY)
    delt = md.addVars(idx2, vtype=grb.GRB.BINARY)
    # 是否要求lambda大于0
    # lbd = md.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
    # 修改点2
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


    e1 = sum(
        (x.sum(t, j, "*").getValue() + (up1[idx] - lbd) * (1 - y[t, j].x)) * alph[idx]
        + (lp1[idx] - lbd + ml1[idx] * y[t,j].x) * beta[idx]
        for idx, j in enumerate(N1)
    )

    e2 = sum(
        (x.sum(t, j, "*").getValue() + j.variable_lb * (1 - y[t, j].x)) * gamm[idx]
        + j.capacity * delt[idx]
        for idx, j in enumerate(N2)
    )

    v1 = md.addVars(idx2) # correspond to L2\R
    v2 = md.addVars(idx2) # correspond to L2^R
    v3 = md.addVars(idx2) # correspond to N2\L2\C2
    v4 = md.addVars(idx2)
    for idx, j in enumerate(N2):
        md.addConstr(
            v1[idx] == (x.sum(t, j, "*").getValue()
            - (lp2[idx] - lbd) * y[t, j].x) * (1-gamm[idx]-delt[idx])
        )
        md.addConstr(
            v2[idx] == mu2[idx] * y[t, j].x * (1-gamm[idx]-delt[idx])
        )
        md.addConstr(
            v3[idx] == x.sum(t, j, "*").getValue() * (1-gamm[idx]-delt[idx])
        )
        md.addConstr(
            v4[idx] == grb.min_([v1[idx],v2[idx],v3[idx]])
        )
    md.setObjective(e1 - e2 - grb.quicksum(v4) - 0, sense=grb.GRB.MAXIMIZE)
    md.setParam("NonConvex", 2)
    md.setParam("MIPGap", 0.01)
    md.setParam("OutputFlag", verbose)
    md.optimize()
    if dump:
        md.write("md.mps")
    # 修改点3: 修复了一些数值问题
    C1dR = {e for idx, e in enumerate(N1) if round(alph[idx].x) == 1}
    C1iR = {e for idx, e in enumerate(N1) if round(beta[idx].x) == 1}
    C2dR = {e for idx, e in enumerate(N2) if round(gamm[idx].x) == 1}
    C2iR = {e for idx, e in enumerate(N2) if round(delt[idx].x) == 1}
    L2C = {idx for idx, e in enumerate(N2) if (gamm[idx].x + delt[idx].x) == 0}
    # L1 = emptyset
    # L1dR = {e for idx, e in enumerate(N1) if alpha[idx].x == 1}
    # L1iR = {e for idx, e in enumerate(N1) if alpha[idx].x == 1}
    l2values = np.array(
        [[v1[idx].x, v2[idx].x, v3[idx].x] for idx, e in enumerate(N2)]
    )
    vals = l2values.argmin(axis=1)
    L2iR = set()
    L2dR = set()
    ow = set()
    # L2dR = {e for idx, e in enumerate(N1) if alpha[idx].x == 1}、
    # 修改点4 对照论文做了一些修改
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

    # L2dR = {e for idx, e in enumerate(N2) if vals[idx] == 0 and idx in L2C}
    # L2iR = {e for idx, e in enumerate(N2) if vals[idx] == 1 and idx in L2C}
    # ow = {e for idx, e in enumerate(N2) if vals[idx] == 2 and idx in L2C}
    # L2iR = subsets
    # 测试部分
    # temp = (
    #         sum(j.variable_lb for j in C1iR)
    #         + sum(j.capacity for j in C1dR)
    #         - sum(j.capacity for j in C2iR)
    #         - sum(j.variable_lb for j in C2dR)
    #         - d
    # )
    # print("第一次",temp)
    subsets = (
        C1dR, C1iR, C2dR, C2iR,
        L2dR, L2iR, ow
    )
    # dbg, check if the min_, max_ is tight
    # df = pd.DataFrame({
    #     "lbd": lbd.x,
    #     "l1": {k: N1[k].variable_lb for k, v in lp1.items()},
    #     "u1": {k: N1[k].capacity for k, v in lp1.items()},
    #     "lp1": {k: v.x - lbd.x for k, v in lp1.items()},
    #     "up1": {k: v.x - lbd.x for k, v in up1.items()},
    #     "ml1": {k: v.x for k, v in ml1.items()},
    # })
    return md.objval, lbd.x, subsets


# get corollary 3 type flow
def eval_cut_c3(x, y, t, d, *subsets, **kwargs):
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
            # 修改点5 这里面的才应该是精确解
            # lbd = kwargs.get("lbdv")
            
    if lbd > 0:
        e1 = (
            x.sum(t, C1dR, "*") 
            + sum(max(j.capacity - lbd, 0) * (1 - y[t, j]) 
                  for j in C1dR)
        )
        e2 = (
            sum((max(j.variable_lb - lbd, 0) + min(j.variable_lb, lbd) * y[t, j])

                for j in C1iR)

        )

        # e3, e4, e5 = 0 if set R = universe
        e3 = (
           x.sum(t, C2dR, "*") 
            + sum(j.variable_lb * (1 - y[t, j]) for j in C2dR)
        )
        e4 = sum(j.capacity for j in C2iR)
        e5 = (
            x.sum(t, L2dR, "*") 
            - sum(max(j.variable_lb - lbd, 0) * y[t, j] for j in L2dR)
        )
        e6 = (
            sum(min(j.capacity, lbd) * y[t, j] for j in L2iR)
        )
        e7 = (
            x.sum(t, ow, "*") 
        )
        expr = e1 + e2 - e3 - e4 - e5 - e6 - e7 - d
        # try:
        cut_value = expr.getValue()
        # except:
        #     cut_value = 0
        bool_voilate = cut_value > 0
        # print(e3)
        return expr, cut_value, lbd, bool_voilate
    return 0, 0, 0, False



