import json
import pickle

import coptpy
import numpy as np
from coptpy import *

from utils import *

# todo
# dirty_trick
extra_cost = {
    # ('T0001', 'T0014', 'Y000001'),
    # ('T0015', 'T0030', 'Y000001'),
    # ('T0040', 'T0026', 'Y000001'),
    # ('T0001', 'T0014', 'Y000002'),
    # ('T0015', 'T0030', 'Y000002'),
    # ('T0026', 'T0040', 'Y000002'),
    # ('T0015', 'T0030', 'Y000003'),
    # ('T0001', 'T0014', 'Y000003'),
    # ('T0015', 'T0030', 'Y000004'),
    # ('T0015', 'T0030', 'Y000005'),
    # ('T0001', 'T0014', 'Y000005'),
    # ('T0040', 'T0026', 'Y000005'),
    # ('T0014', 'T0001', 'Y000006'),
    # ('T0015', 'T0030', 'Y000006'),
    # ('T0001', 'T0014', 'Y000006'),
    # ('T0026', 'T0040', 'Y000006'),
    # ('T0015', 'T0030', 'Y000007'),
    # ('T0001', 'T0014', 'Y000007'),
    # ('T0001', 'T0014', 'Y000009'),
    # ('T0015', 'T0030', 'Y000009'),
    # ('T0026', 'T0040', 'Y000009'),
    # ('T0015', 'T0030', 'Y000011'),
    # ('T0001', 'T0014', 'Y000011'),
    # ('T0026', 'T0040', 'Y000011'),
    # ('T0015', 'T0030', 'Y000012'),
    # ('T0001', 'T0014', 'Y000012'),
    # ('T0015', 'T0030', 'Y000013'),
    # ('T0001', 'T0014', 'Y000013'),
    # ('T0015', 'T0030', 'Y000014'),
    # ('T0001', 'T0014', 'Y000014'),
    # ('T0001', 'T0014', 'Y000015'),
    # ('T0015', 'T0030', 'Y000015'),
    # ('T0015', 'T0030', 'Y000016'),
    # ('T0001', 'T0014', 'Y000016'),
    # ('T0040', 'T0026', 'Y000016'),
    # ('T0001', 'T0014', 'Y000017'),
    # ('T0015', 'T0030', 'Y000017'),
    # ('T0040', 'T0026', 'Y000017'),
    ('T0015', 'T0030', 'Y000020'),
    ('T0001', 'T0014', 'Y000020'),
    ('T0040', 'T0026', 'Y000020'),
    # ('T0015', 'T0030', 'Y000021'),
    # ('T0015', 'T0030', 'Y000022'),
    # ('T0015', 'T0030', 'Y000025'),
    # ('T0001', 'T0014', 'Y000026'),
    # ('T0015', 'T0030', 'Y000026'),
    # ('T0040', 'T0026', 'Y000026'),
    # ('T0014', 'T0001', 'Y000027'),
    # ('T0001', 'T0014', 'Y000027'),
    # ('T0015', 'T0030', 'Y000027'),
    # ('T0040', 'T0026', 'Y000027'),
    # ('T0015', 'T0030', 'Y000037'),
    # ('T0014', 'T0001', 'Y000038'),
    # ('T0015', 'T0030', 'Y000038'),
    # ('T0001', 'T0014', 'Y000038'),
    # ('T0040', 'T0026', 'Y000038'),
    # ('T0015', 'T0030', 'Y000041'),
    # ('T0001', 'T0014', 'Y000041'),
    # ('T0040', 'T0026', 'Y000041'),
    # ('T0015', 'T0030', 'Y000046'),
    # ('T0015', 'T0030', 'Y000047'),
    # ('T0015', 'T0030', 'Y000048'),
    # ('T0015', 'T0030', 'Y000049'),
    # ('T0015', 'T0030', 'Y000052'),
    # ('T0015', 'T0030', 'Y000057'),
    # ('T0001', 'T0014', 'Y000057'),
    # ('T0015', 'T0030', 'Y000108'),
    # ('T0001', 'T0014', 'Y000108'),
    # ('T0001', 'T0014', 'Y000111'),
    # ('T0015', 'T0030', 'Y000111'),
    # ('T0040', 'T0026', 'Y000111'),
    # ('T0015', 'T0030', 'Y000113'),
    # ('T0001', 'T0014', 'Y000113'),
    # ('T0015', 'T0030', 'Y000114'),
    # ('T0001', 'T0014', 'Y000114'),
    # ('T0015', 'T0030', 'Y000125'),
    # ('T0001', 'T0014', 'Y000125'),
    # ('T0040', 'T0026', 'Y000125'),
    # ('T0015', 'T0030', 'Y000135'),
    # ('T0040', 'T0026', 'Y000135'),
    # ('T0015', 'T0030', 'Y000147'),
    # ('T0003', 'T0015', 'Y000043'),
    # ('T0001', 'T0014', 'Y000043'),
    # ('T0040', 'T0026', 'Y000043'),
    # ('T0040', 'T0026', 'Y000107'),
    # ('T0040', 'T0026', 'Y000145'),
    # ('T0040', 'T0026', 'Y000160'),
    # ('T0030', 'T0015', 'Y000123'),
    # ('T0026', 'T0040', 'Y000123'),
    # ('T0014', 'T0001', 'Y000162'),
    # ('T0030', 'T0015', 'Y000162'),
    # ('T0026', 'T0040', 'Y000162'),
    # ('T0003', 'T0015', 'Y000055'),
    # ('T0003', 'T0015', 'Y000079'),
    # ('T0014', 'T0001', 'Y000010'),
}


class PhaseTwo:

    def __init__(self, data, model_dir):
        self.data = data
        self.model_dir = model_dir

        self.model = None
        self.variables = []
        self.state_constrs = []
        self.integer_fix_constrs = []
        self.obj_expr_map = {}
        self.obj_expr = 0
        self.obj_map = {}

    def obj_visualize(self):
        self.obj_map = _values = {k: np.round(v.getValue(), 4) for k, v in self.obj_expr_map.items()}
        _values_repr = {
            k: f"{v:.2e}" for k, v in _values.items()
        }
        print("--- objective summary ---")
        self.obj_map_str = json.dumps(_values_repr, indent=3)
        print(self.obj_map_str)

    @timer
    def load_phasei_from_local(self, path):
        print(f"Loading Phase-I Solution @{path}")
        self.data.available_routes, self.data.warehouse_routes = pickle.load(open(path, 'rb'))

    def routing_heuristics(self, maximum_cardinality=3):
        """
        A heuristic for adding nearby routes
        :return:
        """
        df_cost = pd.Series(self.data.warehouse_to_customer_cost).reset_index()
        cus_sku_tuples = set(((i, j) for (i, j, t) in self.data.cus_demand_periodly))
        greedy_routes = df_cost.groupby(['level_1', 'level_2']).apply(
            lambda x: sorted(zip(x['level_0'], x[0]), key=lambda y: y[-1])[:maximum_cardinality])[cus_sku_tuples]
        for (j, k), vv in greedy_routes.items():  # T0025仓的运输成本是按箱计算的，即使是深圳客户，该仓也无法进入成本前三，增加该条线路
            upstream_facs = [v[0] for v in vv]
            if 'T0015' in upstream_facs or 'T0030' in upstream_facs:
                if ('T0025', j, k) in self.data.warehouse_to_customer_cost:
                    vv.append(('T0025', 1))
        return [(i, j, k) for (j, k), vv in greedy_routes.items() for i, c in vv]

    @timer
    def build(self):
        print("Phase Two Model Start ...")
        # Create COPT environment
        env = Envr()

        # Create COPT model
        model: coptpy.Model = env.createModel("phase_two")

        # ================== 添加决策变量 =====================
        z_p = model.addVars(self.data.Z_P, self.data.T, nameprefix="z_p", vtype=COPT.CONTINUOUS)  # 工厂生产量
        z_l = model.addVars(self.data.Z_L, self.data.T, nameprefix="z_l", vtype=COPT.CONTINUOUS)  # 产线生产量
        # y_l = model.addVars(self.data.Z_L, self.data.T, nameprefix="y_l", vtype=COPT.BINARY)  # 产线是否生产
        x_p = model.addVars(self.data.X_P, self.data.T, nameprefix="x_p", vtype=COPT.CONTINUOUS)  # 工厂->仓库 线路运输量
        x_w = model.addVars(self.data.X_W if DEFAULT_ALG_PARAMS.phase2_use_full_model else self.data.warehouse_routes,
                            self.data.T,
                            nameprefix="x_w", vtype=COPT.CONTINUOUS)  # 仓库->仓库 线路运输量
        if DEFAULT_ALG_PARAMS.phase2_use_full_model == 1:
            x_c = model.addVars(
                self.data.X_C,
                self.data.T,
                nameprefix="x_c",
                vtype=COPT.CONTINUOUS)  # 仓库->客户 线路运输量
        elif DEFAULT_ALG_PARAMS.phase2_use_full_model == 2:
            print("using greedy selections")
            w2c_heur = self.routing_heuristics(DEFAULT_ALG_PARAMS.phase2_greedy_range)
            w2c_routes = set(w2c_heur).union(self.data.available_routes)
            print(
                f"routing expansion:#{DEFAULT_ALG_PARAMS.phase2_greedy_range}: {len(self.data.available_routes)} => {len(w2c_routes)} from {len(w2c_heur)} ")
            w2ct_list = [(i, k, s, t) for (i, k, s) in w2c_routes for t in self.data.T if
                         (k, s, t) in self.data.cus_demand_periodly]
            x_c = model.addVars(
                w2ct_list,
                nameprefix="x_c", vtype=COPT.CONTINUOUS)
        else:
            x_c = model.addVars(
                self.data.available_routes,
                self.data.T,
                nameprefix="x_c", vtype=COPT.CONTINUOUS)  # 仓库->客户 线路运输量

        inv = model.addVars(self.data.I, self.data.S, self.data.T_t, nameprefix="inv", vtype=COPT.CONTINUOUS)  # 库存量
        surplus_inv = model.addVars(self.data.I, self.data.S, self.data.T_t, nameprefix="inv", vtype=COPT.CONTINUOUS)
        storage_gap = model.addVars(self.data.I, self.data.T_t, nameprefix="storage_gap", vtype=COPT.CONTINUOUS)
        inv_f = model.addVars(self.data.I, self.data.S, self.data.T_t, nameprefix="invf",
                              vtype=COPT.CONTINUOUS)  # 丢弃库存量
        tmp = model.addVars(self.data.I, self.data.S, self.data.T_t, nameprefix="tmp", vtype=COPT.CONTINUOUS)
        inv_gap = model.addVars(self.data.I, self.data.S, self.data.T_t, nameprefix="gap",
                                vtype=COPT.CONTINUOUS)  # 需求缺口量
        end_inv_gap = model.addVars(self.data.I, self.data.S, nameprefix="gap", vtype=COPT.CONTINUOUS)  # 期末库存缺口量
        M = self.data.warehouse_capacity_monthly.total_capacity.max() * 100  # 生成辅助变量

        self.variables = z_p, z_l, x_p, x_w, x_c, inv, surplus_inv, tmp, inv_gap, inv_f
        self.model = model

        # ================== 添加业务约束 =====================
        # ====== 变量间耦合关系 =====
        # 工厂生产量=工厂内各产线生产量之和
        model.addConstrs((z_l.sum(p, '*', s, t) == z_p[(p, s, t)] for p, s, t in z_p), nameprefix='line_product')
        # 工厂生产量=运输到各仓库量之和
        model.addConstrs((x_p.sum(p, '*', s, t) == z_p[(p, s, t)] for p, s, t in z_p), nameprefix='transport')

        # ====== 工厂生产约束 =====
        # 产线最小生产量约束
        # model.addConstrs((z_l[(p,l,s,t)] <= y_l[(p,l,s,t)] * M for p,l,s,t in z_l), nameprefix='line_turn')
        # model.addConstrs((z_l[(p,l,s,t)] >= y_l[(p,l,s,t)] * self.data.line_prod_mpq[p,l,s]
        #                   for p,l,s,t in z_l), nameprefix='line_mpq')
        # 产线产能约束
        for t in self.data.T:
            for (p, l), s_list in self.data.Ls.items():
                model.addConstr(
                    (quicksum(z_l[(p, l, s, t)] / self.data.line_prod_capcity_periodly[(p, l, s)] for s in s_list) <= 1)
                    , name='line_capacity')

        # ====== 仓库出货量约束 =====
        # todo, maybe like this,
        x_w_subset = tupledict(
            {(i_s, j, s, t_s): v for (i_s, j, s, t_s), v in x_w.items()
             if (i_s, j, s) not in self.data.X_W_inner}
        )
        model.addConstrs(
            (quicksum(x_w_subset.select(i, "*", "*", t))
             + x_c.sum(i, '*', '*', t)
             + self.data.wh_demand_periodly_gp.get((i, t), 0) <=
             self.data.wh_outbound_cap_periodly[i]
             for i in self.data.I for t in self.data.T),
            nameprefix='wh_outbound'
        )
        # model.addConstrs(
        #     (quicksum([x_w[(i_s, j, s, t_s)] for i_s, j, s, t_s in x_w
        #                if i == i_s and t == t_s and (i_s, j, s) not in self.data.X_W_inner])
        #      + x_c.sum(i, '*', '*', t)
        #      + self.data.wh_demand_periodly_gp.get((i, t), 0) <=
        #      self.data.wh_outbound_cap_periodly[i]
        #      for i in self.data.I for t in self.data.T),
        #     nameprefix='wh_outbound')

        # ====== 客户需求满足约束 =====
        for k, s, t in self.data.KST:
            model.addConstr((x_c.sum('*', k, s, t) == self.data.cus_demand_periodly.get((k, s, t), 0)),
                            name='cus_demand')

        # ====== 仓库库存约束 =====
        # 第0期期末库存=期初库存参数
        model.addConstrs(
            (inv[(i, s, self.data.T_t[0])] == self.data.init_inventory.get((i, s), 0) for i in self.data.I for s in
             self.data.S),
            nameprefix='inital_inv')

        # 期末库存 = 前一期期末库存 + 工厂到仓库的运输量 + 仓库到仓库运输量 - 向其他仓库运输量 - 当周需求量
        # todo, maybe like this?
        # for t in self.data.T_t[1:]
        for i, t in enumerate(self.data.T_t):
            if i >= 1:
                t_pre = self.data.T_t[i - 1]
                model.addConstrs(
                    (inv[i, s, t] == inv[i, s, t_pre] + inv_gap[i, s, t] + x_p.sum('*', i, s, t) + x_w.sum('*', i, s, t) \
                     - x_w.sum(i, '*', s, t) - x_c.sum(i, '*', s, t) - self.data.wh_demand_periodly.get((i, s, t), 0) \
                     - inv_f[i, s, t]
                     for i in self.data.I for s in self.data.S), nameprefix='inv')

        # 一段仓最多提前一个月备货
        # @note: a new formulation
        for ind, t in enumerate(self.data.T):
            if ind >= len(self.data.T) - 3:  # 最后一个月无法的值后续需求，不添加该约束，遵循期末库存量约束
                break
            i_start = ind + 1
            i_end = min(len(self.data.T), ind + 4)
            t_list = self.data.T[i_start: i_end]
            if ind <= 3:
                model.addConstrs(
                    tmp[i, s, t] == self.data.init_inventory.get((i, s), 0)
                    for i in self.data.I_1 for s in self.data.S)
            else:
                model.addConstrs(
                    tmp[i, s, t] == quicksum(x_c.sum(i, '*', s, t_p) + self.data.wh_demand_periodly.get((i, s, t_p), 0)
                                             for t_p in t_list)
                    for i in self.data.I_1 for s in self.data.S)
            # model.addConstrs(
            #     tmp[i, s, t] >= self.data.init_inventory.get((i, s), 0) for i in self.data.I_1 for s in self.data.S)

            model.addConstrs((inv[i, s, t] <= tmp[i, s, t] + surplus_inv[i, s, t]  # 添加松弛变量
                              for i in self.data.I_1 for s in self.data.S), nameprefix='level1_inv')
        # for ind, t in enumerate(self.data.T):
        #     if ind >= len(self.data.T) - 3:  # 最后一个月无法的值后续需求，不添加该约束，遵循期末库存量约束
        #         break
        #     i_start = ind + 1
        #     i_end = min(len(self.data.T), ind + 4)
        #     t_list = self.data.T[i_start: i_end]
        #     model.addConstrs(tmp[i, s, t] == quicksum(x_c.sum(i, '*', s, t_p) +
        #                                               self.data.wh_demand_periodly.get((i, s, t_p), 0) for t_p in
        #                                               t_list)
        #                      for i in self.data.I_1 for s in self.data.S)
        #     # model.addConstrs(
        #     #     tmp[i, s, t] >= self.data.init_inventory.get((i, s), 0) for i in self.data.I_1 for s in self.data.S)
        #
        #     model.addConstrs((inv[i, s, t] <= tmp[i, s, t] + surplus_inv[i, s, t]  # 添加松弛变量
        #                       for i in self.data.I_1 for s in self.data.S), nameprefix='level1_inv')
        # # 0类商品不留库存
        # model.addConstrs((inv[i,s,t] <= max(0, self.data.init_inventory.get((i,s), 0))
        #                   for i in self.data.I for s in self.data.S_0 for t in self.data.T), nameprefix='cate0_inv')

        # ====== 仓库库容约束 =====
        for i, t in enumerate(self.data.T_t):
            if i >= 4:
                t_pre = self.data.T_t[i - 1]
                model.addConstrs((inv.sum(i, '*', t_pre) + x_p.sum('*', i, '*', t) + x_w.sum('*', i, '*', t) -
                                  1 * (x_w.sum(i, '*', '*', t) + x_c.sum(i, '*', '*', t) +
                                       self.data.wh_demand_periodly_gp.get((i, t), 0))
                                  <= self.data.wh_storage_capacity_periodly_total[i, t] + storage_gap[(i, t)]
                                  for i in self.data.I), nameprefix='wh_storage')

        for i, t in enumerate(self.data.T_t):
            if i >= 4:
                t_pre = self.data.T_t[i - 1]
                model.addConstrs((quicksum([inv[i, s, t_pre] + x_p.sum('*', i, s, t) + x_w.sum('*', i, s, t) \
                                            - 1 * (x_w.sum(i, '*', s, t) + x_c.sum(i, '*', s, t)) for s in
                                            self.data.normal_S])
                                  <= self.data.wh_storage_capacity_periodly_normal[i, t] + storage_gap[(i, t)]
                                  for i in self.data.I),
                                 nameprefix='wh_storage')

        # ====== 期末库存约束 =====
        model.addConstrs((inv[i, s, self.data.T[-1]] + end_inv_gap[i, s] >= self.data.end_inventory[i, s] for i, s in
                          self.data.end_inventory),
                         nameprefix='end_inv')
        # model.addConstrs(x_c[i, 'C00244033', 'Y000020', t] == 0 for t in self.data.T for i in self.data.I \
        #                  if (i, 'C00244033', 'Y000020', t) in x_c and i != 'T0015')
        # model.addConstr(inv['T0025', 'Y000020', 'period01'] <= 60000)

        # ================== 添加目标函数 =====================
        self.obj_expr_map = dict(
            cost_p2w=quicksum(x_p[p, i, s, t] * (
                    self.data.added_warehouse_cost.get(i, 0) / 1e3 + self.data.plant_to_warehouse_cost[p, i, s]) for
                              p, i, s, t in x_p),
            cost_w2w=quicksum(x_w[i, j, s, t] * (
                    self.data.added_warehouse_cost.get(i, 0) / 1e3 + self.data.warehouse_transfer_cost[i, j, s]
            ) for i, j, s, t in x_w if (i, j, s) not in extra_cost),
            cost_w2w_weird=5000 * quicksum(x_w[i, j, s, t] * (
                    self.data.added_warehouse_cost.get(i, 0) / 1e3 + self.data.warehouse_transfer_cost[i, j, s]
            ) for i, j, s, t in x_w if (i, j, s) in extra_cost),
            cost_w2c=quicksum(x_c[i, k, s, t] * (
                    self.data.added_warehouse_cost.get(i, 0) / 1e3 + self.data.warehouse_to_customer_cost[(i, k, s)])
                              # todo, why
                              for i, k, s, t in x_c),
            cost_prod=quicksum(z_l[p, l, s, t] * self.data.line_prod_cost[p, l, s] for p, l, s, t in z_l),
            cost_inv_gap=quicksum(inv_gap[i, s, t] * 5000 for i, s, t in inv_gap),
            cost_inv_gap_end=quicksum(end_inv_gap[i, s] * 5000 for i, s in end_inv_gap),
            cost_surplus=quicksum(
                surplus_inv[i, s, t] * 100000
                for (idx, t) in enumerate(self.data.T_t)
                for i in self.data.I for s in self.data.S
                if idx >= 4
            ),
            cost_storage=quicksum(storage_gap[i, t] * 500 for i, t in storage_gap),
            cost_inv=quicksum(inv[i, s, t] for i, s, t in inv),
            cost_giveup=200 * quicksum(inv_f[i, s, t] for i, s, t in inv_f)
        )
        self.obj_expr = obj = sum(self.obj_expr_map.values())
        # 期末库存不足惩罚/多余备货惩罚/库存成本，防止提前备货

        model.setObjective(obj, sense=COPT.MINIMIZE)

        model.setParam(COPT.Param.TimeLimit, 1200.0)
        model.setParam(COPT.Param.RelGap, 0.001)
        model.write('phase_two_lp_2.lp')

    def query_invalid_qty(self, zl):

        invalid = {(p, l, s, t): v / self.data.line_prod_mpq[p, l, s]
                   for (p, l, s, t), v in zl.items() if 0 < v < self.data.line_prod_mpq[p, l, s]}
        empty = {(p, l, s, t): v
                 for (p, l, s, t), v in zl.items() if v == 0}
        return invalid, empty

    def clear_integer_fixings(self, z_l):
        if len(self.integer_fix_constrs) > 0:
            self.model.remove(self.integer_fix_constrs)
        for _, v in z_l.items():
            v.setInfo(COPT.Info.LB, 0)
            v.setInfo(COPT.Info.UB, COPT.INFINITY)

    @timer
    def qty_fix_heuristic(self):
        model: coptpy.Model = self.model
        _logs = []
        z_p, z_l, x_p, x_w, x_c, inv, surplus_inv, tmp, inv_gap, storage_gap, end_inv_gap = self.variables
        # --- forward iteration ---
        # try to increase production qty
        zl = model.getInfo(COPT.Info.Value, z_l)
        zl_invalid, zl_empty = self.query_invalid_qty(zl)
        _logs.append(f"original, invalid size {len(zl_invalid)}")

        for (p, l, s, t), v in zl_empty.items():
            z_l[p, l, s, t].setInfo(COPT.Info.UB, 0)
        # if v is large, we make it as large as possible (up to min_prod).
        for (p, l, s, t), v in zl_invalid.items():
            z_l[p, l, s, t].setInfo(COPT.Info.UB, self.data.line_prod_mpq[p, l, s])

        model.setObjective(self.obj_expr - 1e3 * quicksum(
            z_l[p, l, s, t] * v for (p, l, s, t), v in zl_invalid.items()
        ), COPT.MINIMIZE)
        if DEFAULT_ALG_PARAMS.phase2_qty_heur_reset:
            model.reset()
        model.solve()

        # --- backward iteration ---
        # - set all invalid to 0
        # - set others x >= min_prod
        zl = model.getInfo(COPT.Info.Value, z_l)
        zl_invalid, zl_empty = self.query_invalid_qty(zl)
        _logs.append(f"after forward, invalid size {len(zl_invalid)}")

        for (p, l, s, t), v in z_l.items():
            if (p, l, s, t) in zl_invalid:
                v.setInfo(COPT.Info.UB, 0)
            elif (p, l, s, t) in zl_empty:
                v.setInfo(COPT.Info.UB, 0)
            elif self.data.line_prod_mpq[p, l, s] > 0:
                v.setInfo(COPT.Info.UB, COPT.INFINITY)
                v.setInfo(COPT.Info.LB, self.data.line_prod_mpq[p, l, s])
            else:
                pass

        model.setObjective(self.obj_expr, COPT.MINIMIZE)
        if DEFAULT_ALG_PARAMS.phase2_qty_heur_reset:
            model.reset()
        model.solve()
        zl = model.getInfo(COPT.Info.Value, z_l)
        zl_invalid, zl_empty = self.query_invalid_qty(zl)
        _logs.append(f"after backward, invalid size {len(zl_invalid)}")
        print("--- primal fixing heuristic ---")
        [print(f"- {l}") for l in _logs]

    @timer
    def run(self, finalize=True):
        model = self.model
        z_p, z_l, x_p, x_w, x_c, inv, surplus_inv, tmp, inv_gap, *_ = self.variables

        model.write(self.model_dir + "phase_two.mps.gz")
        model.solve()

        # ================== 结果输出 =====================
        if model.status == COPT.INFEASIBLE:
            print("Phase Two Model Infeasible")
            model.computeIIS()
            model.writeIIS(self.model_dir + "iis_two.ilp")

        # if it is not the last iterate, no need to dump results.
        if not finalize:
            return

        # otherwise it is the last iterate,
        # we fix the solution to match min_qty and dump results.
        if DEFAULT_ALG_PARAMS.phase2_use_qty_heur:
            self.qty_fix_heuristic()

        self.obj_visualize()
        # ================== 结果输出 =====================
        if model.status == COPT.INFEASIBLE:
            print("Phase Two Model Infeasible")
            model.computeIIS()
            model.writeIIS(self.model_dir + 'iis_two.ilp')
        else:
            print("Phase Two Model Optimal")
            plant_product = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, z_p))).reset_index()
            plant_product.columns = ['plant_id', 'sku', 'period', 'qty']
            plant_product['qty'] = plant_product['qty'].apply(lambda x: round(x, 3))
            plant_line_product = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, z_l))).reset_index()
            plant_line_product.columns = ['plant_id', 'line_id', 'sku', 'period', 'qty']
            plant_line_product['qty'] = plant_line_product['qty'].apply(lambda x: round(x, 3))
            # @note: added by C.Z, only keep > 0 qty
            plant_line_product = plant_line_product.query("qty > 0")
            plant_line_product['min_prod'] = plant_line_product.apply(
                lambda row: self.data.line_prod_mpq.get((row['plant_id'], row['line_id'], row['sku']), 0), axis=1)

            # if_plant_line_product = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, y_l))).reset_index()
            # if_plant_line_product.columns=['plant_id', 'line_id', 'sku', 'period', 'if_product']

            plant_to_warehouse = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, x_p))).reset_index()
            plant_to_warehouse.columns = ['start_id', 'end_id', 'sku', 'period', 'qty']
            plant_to_warehouse['qty'] = plant_to_warehouse['qty'].apply(lambda x: round(x, 3))
            warehouse_to_warehouse = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, x_w))).reset_index()
            warehouse_to_warehouse.columns = ['start_id', 'end_id', 'sku', 'period', 'qty']
            warehouse_to_warehouse['qty'] = warehouse_to_warehouse['qty'].apply(lambda x: round(x, 3))
            warehouse_to_customer = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, x_c))).reset_index()
            warehouse_to_customer.columns = ['start_id', 'end_id', 'sku', 'period', 'qty']
            warehouse_to_customer['qty'] = warehouse_to_customer['qty'].apply(lambda x: round(x, 3))

            inventory_gap = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, inv_gap))).reset_index()
            inventory_gap.columns = ['warehouse_id', 'sku', 'period', 'inv_gap']
            inventory = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, inv))).reset_index()
            inventory.columns = ['warehouse_id', 'sku', 'period', 'inv']
            inventory['inv'] = inventory['inv'].apply(lambda x: round(x, 3))

            # ======= 统计仓库*sku*period的库存变动情况
            plant_to_warehouse_in = plant_to_warehouse.rename(columns={'end_id': 'warehouse_id', 'qty': 'prod_in'})
            plant_to_warehouse_in = plant_to_warehouse_in.groupby(
                ['warehouse_id', 'sku', 'period']).prod_in.sum().reset_index()
            warehouse_to_warehouse_in = warehouse_to_warehouse.rename(
                columns={'end_id': 'warehouse_id', 'qty': 'transfer_in'})
            warehouse_to_warehouse_in = warehouse_to_warehouse_in.groupby(
                ['warehouse_id', 'sku', 'period']).transfer_in.sum().reset_index()

            warehouse_to_warehouse_out = warehouse_to_warehouse.rename(
                columns={'start_id': 'warehouse_id', 'qty': 'transfer_out'})
            warehouse_to_warehouse_out = warehouse_to_warehouse_out.groupby(
                ['warehouse_id', 'sku', 'period']).transfer_out.sum().reset_index()

            # transfer_out_cap：仓间调拨计入出货量的部分
            warehouse_to_warehouse_tmp = warehouse_to_warehouse.merge(self.data.level1_to_level1_outer_map,
                                                                      on=['start_id', 'end_id', 'sku'], how='left')
            warehouse_to_warehouse_tmp = warehouse_to_warehouse_tmp[warehouse_to_warehouse_tmp.tag.isnull()]
            warehouse_to_warehouse_out_cap = warehouse_to_warehouse_tmp.rename(
                columns={'start_id': 'warehouse_id', 'qty': 'transfer_out_cap'})
            warehouse_to_warehouse_out_cap = warehouse_to_warehouse_out_cap.groupby(
                ['warehouse_id', 'sku', 'period']).transfer_out_cap.sum().reset_index()
            warehouse_to_customer_out = warehouse_to_customer.rename(
                columns={'start_id': 'warehouse_id', 'qty': 'cus_out'})
            warehouse_to_customer_out = warehouse_to_customer_out.groupby(
                ['warehouse_id', 'sku', 'period']).cus_out.sum().reset_index()
            warehouse_demand_out = self.data.warehouse_demand_periodly_tmp.rename(
                columns={'fac_id': 'warehouse_id', 'qty': 'wh_out',
                         'ds_id': 'period'})
            warehouse_demand_out = warehouse_demand_out.groupby(
                ['warehouse_id', 'sku', 'period']).wh_out.sum().reset_index()

            # 构造完整库存数据表
            tmp1 = inventory.merge(plant_to_warehouse_in, how='left')
            tmp2 = tmp1.merge(warehouse_to_warehouse_in, how='left')
            tmp3 = tmp2.merge(warehouse_to_warehouse_out, how='left')
            tmp4 = tmp3.merge(warehouse_to_customer_out, how='left')
            tmp4 = tmp4.merge(warehouse_to_warehouse_out_cap, how='left')
            tmp5 = tmp4.merge(warehouse_demand_out, how='left')
            tmp6 = tmp5.merge(inventory_gap, how='left')
            tmp6['inv_pre'] = tmp6.groupby(['warehouse_id', 'sku']).inv.shift(1)
            tmp6 = tmp6.fillna(0)
            tmp6['storage'] = tmp6.inv_pre.apply(lambda x: max(x, 0)) + tmp6.prod_in + tmp6.transfer_in - \
                              1 * (tmp6.transfer_out + tmp6.cus_out + tmp6.wh_out)

            plant_line_product.to_csv(self.model_dir + 'plant_line_product.csv', index=False)
            plant_to_warehouse.to_csv(self.model_dir + 'plant_to_warehouse.csv', index=False)
            warehouse_to_warehouse.to_csv(self.model_dir + 'warehouse_to_warehouse.csv', index=False)
            warehouse_to_customer.to_csv(self.model_dir + 'warehouse_to_customer.csv', index=False)
            tmp6.to_csv(self.model_dir + 'warehouse_inv_change.csv', index=False)

            # utility of warehouse
            new_wh = set(self.data.warehouse_df.query("if_current == 0")['fac_id'].unique())
            tmp6_gp = tmp6.groupby(['warehouse_id', 'period']).agg(
                dict(
                    storage=sum,
                    prod_in=sum,
                    transfer_in=sum,
                    transfer_out=sum,
                    cus_out=sum,
                    transfer_out_cap=sum,
                    wh_out=sum,
                    inv_gap=sum,
                )
            ).reset_index()
            tmp = tmp6_gp.merge(
                self.data.wh_storage_capacity_periodly_total.reset_index().rename(
                    columns={'fac_id': 'warehouse_id', "ds_id": "period"}),
                how='left', on=['warehouse_id', "period"])

            tmp = tmp.assign(
                cap_utility=lambda df: df.storage.apply(lambda x: max(x, 0)) / df.total_capacity,
                outbound=lambda df: 0.5 * (df.transfer_out + df.cus_out + df.wh_out),
                outbound_cap=lambda df: df['warehouse_id'].apply(lambda i: self.data.wh_outbound_cap_periodly[i]),
                outbound_cap_utility=lambda df: df.outbound / df.outbound_cap,
                bool_is_new=lambda df: df['warehouse_id'].apply(lambda x: x in new_wh)
            )
            tmp.to_csv(self.model_dir + 'warehouse_io_cap.csv', index=False)

            self.plant_to_warehouse = plant_to_warehouse

            self.model_metric()
        print("Phase Two Model End")
        self.clear_integer_fixings(z_l)

    def set_state(self):
        if DEFAULT_ALG_PARAMS.phase2_use_full_model == 1:
            # todo, this is not realistic.
            # since |X_W|, |X_C| is too large.
            print("using full model...")
            model: coptpy.Model = self.model
            z_p, z_l, x_p, x_w, x_c, inv, surplus_inv, tmp, inv_gap, *_ = self.variables

            # Benders state variable
            x_w_rhs = {(*k, t): 0 if k not in set(self.data.warehouse_routes) else 1e4 for k in self.data.X_W for t in
                       self.data.T}
            x_c_rhs = {(*k, t): 0 if k not in set(self.data.available_routes) else 1e4 for k in self.data.X_C for t in
                       self.data.T}

            if self.state_constrs.__len__() == 0:
                bound_constr_w = model.addConstrs((x_w[k] <= x_w_rhs[k] for k in self.data.X_W))
                bound_constr_c = model.addConstrs((x_c[k] <= x_c_rhs[k] for k in self.data.X_C))
                self.state_constrs = (bound_constr_w, bound_constr_c)
            else:
                bound_constr_w, bound_constr_c = self.state_constrs
                model.setInfo(COPT.Info.UB, bound_constr_w, x_w_rhs)
                model.setInfo(COPT.Info.UB, bound_constr_c, x_c_rhs)

    def model_metric(self):
        print(f"--- model product statistics ---")
        fake_product = set(self.plant_to_warehouse[
                               (self.plant_to_warehouse.start_id == 'P000X') & (self.plant_to_warehouse.qty > 0)]) & \
                       set(self.data.plant_sku_df[self.data.plant_sku_df.fac_id != 'P000X'].sku)
        print(f"model fake production, x_w: {len(fake_product)}")

    def debug(self):
        model = self.model
        z_p, z_l, x_p, x_w, x_c, inv, surplus_inv, tmp, inv_gap, *_ = self.variables
        expr = quicksum(x_c.select("T0015", "C00244033", "Y000020", "*"))
        self.constr_dbg = self.model.addConstr(expr == 10200)
        self.model.solve()
        print(self.obj_map_str)
        self.obj_visualize()
