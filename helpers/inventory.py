"""
a submodule defining a collection of inventory constraints
"""

import json
import pickle

import coptpy
import numpy as np
from tqdm import tqdm
from coptpy import *

from utils import *


def get_l_periods_head(self, t, l):
    """
    get the time of l periods ahead of t
    :param self:
    :param t:
    :param l:
    :return:
    """
    _idx = self.data.T_n[t]
    _idx_ahead = _idx - l
    return self.data.T_t[_idx_ahead] if _idx_ahead >= 0 else None


def get_period_prev(self, t):
    return get_l_periods_head(self, t, 1)


def query_avail_prod(self):
    """
    get available production output of the day,
        on account of the `lead time`
    lead_time = q + s, q in Z, s in [0, 1), we have,
        z_p_avail[*, t] = z_p[*, t - q - 1] * s + z_p[*, t - q] * (1-s)
    e.g.,
      0: lead time = 0, q = s = 0
          z_p_avail[*, t] = z_p[*, t - 1] * 0 + z_p[*, t] * 1
      1: lead time = 1 + x, x < 1
          z_p_avail[*, t] = z_p[*, t - 2] * x + z_p[*, t - 1] * (1 - x)
      2: lead time = 1 + x, x < 1, x -> 0, lead time -> 1
          z_p_avail[*, t] -> z_p[*, t - 1]
      3: lead time = 1 + x, x < 1, x -> 1, lead time -> 2
          z_p_avail[*, t] -> z_p[*, t - 2]
    :return:
    """
    model = self.model
    (
        z_p,
        z_l,
        x_p,
        x_w,
        x_c,
        inv,
        surplus_inv,
        inv_gap,
        inv_f,
        inv_avail,
        storage_gap,
        end_inv_gap,
        *_,
    ) = self.variables

    try:
        prod_leadtime = self.data.prod_leadtime
    except:
        # undefined
        prod_leadtime = {(p, j): 1.2 for p, j, t in z_p}

    prod_leadtime_int = {k: int(v) for k, v in prod_leadtime.items()}
    prod_leadtime_res = {k: v - prod_leadtime_int[k] for k, v in prod_leadtime.items()}

    x_p_avail = {
        (p, w, j, t): x_p.get(
            (p, w, j, get_l_periods_head(self, t, prod_leadtime_int[p, j] + 1)), 0
        )
        * round(prod_leadtime_res[p, j] if prod_leadtime_res[p, j] > 0.01 else 0, 2)
        + x_p.get((p, w, j, get_l_periods_head(self, t, prod_leadtime_int[p, j])), 0)
        * round(1 - prod_leadtime_res[p, j], 2)
        for (p, w, j, t), v in x_p.items()
    }
    # sum up the upstream
    x_p_avail_sum = (
        pd.Series(x_p_avail).groupby(level=[1, 2, 3]).apply(quicksum).to_dict()
    )
    return x_p_avail, x_p_avail_sum


@timer
def query_unavail_prod(self):
    """
    get un-available production output of the day,
        on account of the `lead time`
    lead_time = q + s, q in Z, s in [0, 1), we have,
        z_p_unavail[*, t] =
        + z_p[t]
        + z_p[t-1]
        + ...
        + z_p[t-q] * s
    :return:
    """
    model = self.model
    (
        z_p,
        z_l,
        x_p,
        x_w,
        x_c,
        inv,
        surplus_inv,
        inv_gap,
        inv_f,
        inv_avail,
        storage_gap,
        end_inv_gap,
        *_,
    ) = self.variables

    try:
        prod_leadtime = self.data.prod_leadtime
    except:
        # undefined
        prod_leadtime = {(p, j): 1.2 for p, j, t in z_p}

    prod_leadtime_int = {k: int(v) for k, v in prod_leadtime.items()}
    prod_leadtime_res = {k: v - prod_leadtime_int[k] for k, v in prod_leadtime.items()}

    x_p_unavail = {
        (p, w, j, t): quicksum(
            x_p.get((p, w, j, get_l_periods_head(self, t, tau)), 0)
            for tau in range(0, prod_leadtime_int[p, j])
        )
        + x_p.get((p, w, j, get_l_periods_head(self, t, prod_leadtime_int[p, j])), 0)
        * round(prod_leadtime_res[p, j], 2)
        for (p, w, j, t), v in x_p.items()
        if prod_leadtime_int[p, j] > 0
    }
    # sum up the upstream
    x_p_unavail_sum = (
        pd.Series(x_p_unavail).groupby(level=[1, 2, 3]).apply(quicksum).to_dict()
    )
    return x_p_unavail, x_p_unavail_sum


@timer
def add_inventory_constr(self):
    """
    inventory builder
    @note:
        - variable x: (start, end, sku, time)
        - variable inv: (place, sku, time)
    :param self:
    :return:
    """
    model = self.model
    (
        z_p,
        z_l,
        x_p,
        x_w,
        x_c,
        inv,
        surplus_inv,
        inv_gap,
        inv_f,
        inv_avail,
        storage_gap,
        end_inv_gap,
        *_,
    ) = self.variables

    # auxiliary quantities
    # x* start, end, sku, time
    x_w_subset = tupledict(
        {
            (i_s, j, s, t_s): v
            for (i_s, j, s, t_s), v in x_w.items()
            if (i_s, j, s) not in self.data.X_W_inner
        }
    )
    x_p_unavailable, x_p_unavailable_sum = query_unavail_prod(self)
    x_p_available, x_p_avail_sum = query_unavail_prod(self)

    # ====== 仓库出货量约束 =====
    model.addConstrs(
        (
            quicksum(x_w_subset.select(i, "*", "*", t))
            + x_c.sum(i, "*", "*", t)
            + self.data.wh_demand_periodly_gp.get((i, t), 0)
            <= self.data.wh_outbound_cap_periodly[i]
            for i in self.data.I
            for t in self.data.T
        ),
        nameprefix="wh_outbound",
    )

    # 一段仓最多提前一个月备货
    # @note: a new formulation
    for ind, t in enumerate(self.data.T):
        if ind >= len(self.data.T) - 3:  # 最后一个月无法的值后续需求，不添加该约束，遵循期末库存量约束
            break
        i_start = ind + 1
        i_end = min(len(self.data.T), ind + 4)
        t_list = self.data.T[i_start:i_end]
        for i, s in itertools.product(self.data.I_1, self.data.S):
            if ind <= 3:
                tmp = self.data.init_inventory.get((i, s), 0)
            else:
                tmp = quicksum(
                    x_c.sum(i, "*", s, t_p)
                    + self.data.wh_demand_periodly.get((i, s, t_p), 0)
                    for t_p in t_list
                )
            model.addConstr(
                inv[i, s, t] <= tmp + surplus_inv[i, s, t],
                name=f"level1_inv-{i, s, t}",
            )

    # ====== 仓库库容约束 =====
    for i, t in enumerate(self.data.T_t[4:]):
        model.addConstrs(
            (
                inv.sum(i, "*", t)
                <= self.data.wh_storage_capacity_periodly_total[i, t]
                + storage_gap[(i, t)]
                for i in self.data.I
            ),
            nameprefix="wh_storage",
        )
        model.addConstrs(
            (
                quicksum(inv[i, s, t] for s in self.data.normal_S)
                <= self.data.wh_storage_capacity_periodly_normal[i, t]
                + storage_gap[(i, t)]
                for i in self.data.I
            ),
            nameprefix="wh_storage_normal",
        )

    # ====== 仓库库存约束 =====
    # 第0期期末库存=期初库存参数
    model.addConstrs(
        (
            inv[(i, s, self.data.T_t[0])] == self.data.init_inventory.get((i, s), 0)
            for i in self.data.I
            for s in self.data.S
        ),
        nameprefix="initial_inv",
    )
    model.addConstrs(
        (
            inv_avail[(i, s, self.data.T_t[0])] == inv[i, s, self.data.T_t[0]]
            for i in self.data.I
            for s in self.data.S
        ),
        nameprefix="initial_inv",
    )

    # 期末库存 = 前一期期末库存 + 工厂到仓库的运输量 + 仓库到仓库运输量 - 向其他仓库运输量 - 当周需求量
    _list_inventory = list(itertools.product(self.data.T_t, self.data.I, self.data.S))
    for t, i, s in tqdm(_list_inventory, desc="build-inv-flow", ncols=100):
        idx = self.data.T_n[t]
        if idx == 0:
            continue
        t_pre = self.data.T_t[idx - 1]
        xpiu = x_p_unavailable_sum.get((i, s, t), 0)  # inbound p2w, not ready to use
        xpi = x_p.sum("*", i, s, t)  # inbound p2w:  x_w.sum("*", i, s, t)
        xwi = x_w.sum("*", i, s, t)  # inbound w2w:  x_w.sum(i, "*", s, t)
        xwo = x_w.sum(i, "*", s, t)  # outbound w2w: x_c.sum(i, "*", s, t)
        xco = x_c.sum(i, "*", s, t)  # outbound w2c
        # 可用库存: inv_avail
        # @note
        # I'm not sure if this can be simplified whatsoever.
        #
        model.addConstr(
            (
                inv[i, s, t]
                == inv[i, s, t_pre]
                + xpi
                + xwi
                + inv_gap[i, s, t]  # shortage
                - xwo
                - xco
                - self.data.wh_demand_periodly.get((i, s, t), 0)
                - inv_f[i, s, t]  # perished
            ),
            name=f"inv-{i}{s}{t}",
        )
        model.addConstr(
            (inv_avail[i, s, t] == inv[i, s, t] - xpiu),
            name=f"inv_avail-{i}{s}{t}",
        )

        # now consider a simple relaxation
        #   of fresh-perishable requirements.
        # we place a simple rule as below.
        # for w2w case,
        #   the qty of `transfer` must be lower than fresh coming-in
        model.addConstr(
            (
                    x_p_avail_sum.get((i, s, t), 0) + xwi
                    >= xwo
            ),
            name=f"fresh-perishable-relaxation-{i}{s}{t}",
        )

    # ====== 期末库存约束 =====
    model.addConstrs(
        (
            inv[i, s, self.data.T[-1]] + end_inv_gap[i, s]
            >= self.data.end_inventory[i, s]
            for i, s in self.data.end_inventory
        ),
        nameprefix="end_inv",
    )
