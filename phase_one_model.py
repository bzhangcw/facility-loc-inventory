import pickle

import pandas as pd
from coptpy import *
from utils import *


class PhaseOne:

    def __init__(self, data, model_dir):

        self.data = data
        self.model_dir = model_dir
        self.solution_dir = os.path.join(self.model_dir, "phasei.pkl")
        self.model = None
        self.variables = []
        self.cuts = []

    @timer
    def build(self):
        print("Phase One Model Start ...")
        env = Envr()  # Create COPT environment
        model = env.createModel("phase_one")  # Create COPT model

        # ================== 添加决策变量 =====================
        # TODO 减少0-1决策变量
        w = model.addVars(self.data.W, nameprefix="w", vtype=COPT.BINARY)  # 仓库是否开设
        # ytype = COPT.BINARY if not PHASEONE_RELAX else COPT.CONTINUOUS
        # y_p = model.addVars(self.data.X_P, nameprefix="y_p", vtype=ytype)  # 工厂->仓库 线路是否开设
        # y_w = model.addVars(self.data.X_W, nameprefix="y_w", vtype=ytype)  # 仓库->仓库 线路是否开设
        # y_c = model.addVars(self.data.X_C, nameprefix="y_c", vtype=ytype)  # 仓库->客户 线路是否开设
        x_p = model.addVars(self.data.X_P, nameprefix="x_p", vtype=COPT.CONTINUOUS)  # 工厂->仓库 线路运输量  
        x_w = model.addVars(self.data.X_W, nameprefix="x_w", vtype=COPT.CONTINUOUS)  # 仓库->仓库 线路运输量   
        x_c = model.addVars(self.data.X_C, nameprefix="x_c", vtype=COPT.CONTINUOUS)  # 仓库->客户 线路运输量
        # 超过库容约束
        overstorage = model.addVars(self.data.W, nameprefix="over", vtype=COPT.CONTINUOUS)
        M = self.data.warehouse_capacity_monthly.total_capacity.max() * 100  # 生成辅助变量

        # save to class
        self.model = model
        self.variables = w, x_p, x_w, x_c,  # y_p, y_w, y_c,

        # ================== 添加业务约束 =====================
        # ==== 变量间耦合关系 =====
        # 仓库开设时，附近的线路才可以开通
        # model.addConstrs((w[i] >= y_p[(p, i, s)] for p, i, s in y_p), nameprefix='p2w_route')
        # model.addConstrs((w[i] >= y_w[(i, j, s)] for i, j, s in y_w), nameprefix='w2w_route_1')
        # model.addConstrs((w[i] >= y_w[(j, i, s)] for j, i, s in y_w), nameprefix='w2w_route_2')
        # model.addConstrs((w[i] >= y_c[(i, k, s)] for i, k, s in y_c), nameprefix='w2w_route_2')
        # 只有线路开通时，才可以运输
        # model.addConstrs((x_p[(p, i, s)] <= y_p[(p, i, s)] * M for p, i, s in x_p), nameprefix='p2w_route_qty')
        # model.addConstrs((x_w[(i, j, s)] <= y_w[(i, j, s)] * M for i, j, s in x_w), nameprefix='w2w_route_qty')
        # model.addConstrs((x_c[(i, k, s)] <= y_c[(i, k, s)] * M for i, k, s in x_c), nameprefix='w2c_route_qty')
        model.addConstrs((x_p[(p, i, s)] <= w[i] * M for p, i, s in x_p), nameprefix='p2w_route_qty')
        model.addConstrs((x_w[(i, j, s)] <= w[i] * M for i, j, s in x_w), nameprefix='w2w_route_qty')
        model.addConstrs((x_c[(i, k, s)] <= w[i] * M for i, k, s in x_c), nameprefix='w2c_route_qty')
        # 现有仓库一定开设
        model.addConstrs((w[i] == 1 for i in self.data.current_I), nameprefix='current_warehouse')
        # ==== 流量守恒关系 =====
        # 仓库入库量 = 出库量
        for i in self.data.I:
            for s in self.data.S:
                model.addConstr(x_p.sum('*', i, s) + x_w.sum('*', i, s) == x_w.sum(i, '*', s) + x_c.sum(i, '*', s) + \
                                self.data.wh_demand_monthly.get((i, s), 0))
        # 客户被服务量 = 客户需求
        for k, s in self.data.cus_demand_monthly.index:
            model.addConstr(x_c.sum('*', k, s) == self.data.cus_demand_monthly[(k, s)])
        # ==== 仓库出货量约束 =====
        model.addConstrs(


            quicksum([x_w[(i_s, j, t)] for i_s, j, t in x_w if i == i_s and (i_s, j, t) not in self.data.X_W_inner]) \
            + x_c.sum(i, '*', '*') + self.data.wh_demand_monthly_gp.get(i, 0) <= self.data.wh_outbound_cap_periodly[i]
            for i in self.data.I)
        # ==== 库容约束 ====
        # 按库存天数计算
        model.addConstrs(quicksum((x_w.sum(i, '*', s) + x_c.sum(i, '*', s) + self.data.wh_demand_monthly.get((i, s), 0)) \
                                  * self.data.inventory_days.get((i, s), 5) / 10
                                  for s in self.data.S) <= self.data.wh_storage_capacity_monthly_total[i] + overstorage[i] for i
                         in self.data.I)
        # ==== 虚拟工厂生产约束 ===
        model.addConstrs((x_p.sum('P000X', '*', s) == 0 for s in self.data.insourcing_sku), nameprefix='outsourcing')

        # ================== 添加目标函数 =====================
        obj = quicksum(x_p[p, i, s] * self.data.plant_to_warehouse_cost[p, i, s] for p, i, s in x_p) + \
              quicksum(x_w[i, j, s] * self.data.warehouse_transfer_cost[i, j, s] for i, j, s in x_w) + \
              quicksum(x_c[i, k, s] * self.data.warehouse_to_customer_cost[i, k, s] for i, k, s in x_c) + \
              quicksum(w[i] * self.data.added_warehouse_cost.get(i, 0) for i in w) + \
              quicksum(x_p[p, i, s] * self.data.plant_prod_cost[p, s] for p, i, s in x_p)+\
              quicksum(overstorage[i] for i in w) # 库容约束惩罚

        model.setObjective(obj, sense=COPT.MINIMIZE)
        model.setParam(COPT.Param.TimeLimit, 1200.0)
        model.setParam(COPT.Param.RelGap, 0.00001)

    @timer
    def run(self):
        model = self.model
        w, x_p, x_w, x_c, *_ = self.variables
        model.write(self.model_dir + 'phase_one.mps.gz')
        model.solve()

        # ================== 结果输出 =====================
        if model.status == COPT.INFEASIBLE:
            print("Phase One Model Infeasible")
            model.computeIIS()
            model.writeIIS(self.model_dir + 'iis_one.ilp')
        else:
            warehouse_status = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, w))).reset_index()
            warehouse_status.columns = ['fac_id', 'if_open']
            # if_plant_to_warehouse = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, y_p))).reset_index()
            # if_plant_to_warehouse.columns = ['start_id', 'end_id', 'sku', 'if_open']
            # if_warehouse_to_warehouse = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, y_w))).reset_index()
            # if_warehouse_to_warehouse.columns = ['start_id', 'end_id', 'sku', 'if_open']
            # if_warehouse_to_customer = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, y_c))).reset_index()
            # if_warehouse_to_customer.columns = ['start_id', 'end_id', 'sku', 'if_open']

            plant_to_warehouse = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, x_p))).reset_index()
            plant_to_warehouse.columns = ['start_id', 'end_id', 'sku', 'qty']
            plant_to_warehouse['qty'] = plant_to_warehouse['qty'].apply(lambda x: round(x, 2))
            warehouse_to_warehouse = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, x_w))).reset_index()
            warehouse_to_warehouse.columns = ['start_id', 'end_id', 'sku', 'qty']
            warehouse_to_warehouse['qty'] = warehouse_to_warehouse['qty'].apply(lambda x: round(x, 2))
            warehouse_to_customer = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, x_c))).reset_index()
            warehouse_to_customer.columns = ['start_id', 'end_id', 'sku', 'qty']
            warehouse_to_customer['qty'] = warehouse_to_customer['qty'].apply(lambda x: round(x, 2))

            warehouse_status.to_csv(self.model_dir + 'warehouse_status.csv', index=False)
            plant_to_warehouse.to_csv(self.model_dir + 'plant_to_warehouse.csv', index=False)
            warehouse_to_warehouse.to_csv(self.model_dir + 'warehouse_to_warehouse.csv', index=False)
            warehouse_to_customer.to_csv(self.model_dir + 'warehouse_to_customer.csv', index=False)

            total_transfer_qty = pd.concat([plant_to_warehouse, warehouse_to_warehouse, warehouse_to_customer])
            group = self.data.level1_to_level1_outer_map.groupby(['start_id']).end_id.unique().reset_index()
            group['group'] = group.apply(lambda x: list(set(x.end_id).union({x.start_id})), axis=1)
            group['start_id'] = group['group']
            group = group.explode('start_id')[['start_id', 'group']]

            available_routes = total_transfer_qty[(total_transfer_qty.end_id.apply(lambda x: x[0] == 'C')) &
                                                  (total_transfer_qty.qty > 0)]
            # 非华南区补充所有仓到客户的运输路线
            other_region_trad = available_routes[
                available_routes.end_id == 'C00598623'][['end_id', 'sku']].drop_duplicates()  # todo hard code
            other_region_trad = other_region_trad.merge(self.data.warehouse_df[self.data.warehouse_df.fac_type.apply(
                lambda x: 'level_1' in x)][['fac_id']], how='cross')
            other_region_ka = available_routes[
                available_routes.end_id == 'C00598656'][['end_id', 'sku']].drop_duplicates()
            other_region_ka = other_region_ka.merge(self.data.warehouse_df[self.data.warehouse_df.fac_type.apply(
                lambda x: 'level_2' in x)][['fac_id']], how='cross')
            other_region = pd.concat([other_region_trad, other_region_ka])
            other_region.columns = ['end_id', 'sku', 'start_id']
            available_routes = available_routes.merge(group, on='start_id', how='left')
            available_routes = available_routes.explode('group')
            available_routes.loc[available_routes['group'].notnull(), 'start_id'] = \
                available_routes.loc[available_routes['group'].notnull(), 'group']
            available_routes = pd.concat([available_routes, other_region])
            available_routes = available_routes[['start_id', 'end_id', 'sku']].drop_duplicates()

            warehouse_routes = total_transfer_qty[(total_transfer_qty.start_id.apply(lambda x: x[0] == 'T')) &
                                                  (total_transfer_qty.end_id.apply(lambda x: x[0] == 'T')) &
                                                  (total_transfer_qty.qty > 0)]

            # 添加一段厂内仓之间的运输路线
            warehouse_routes = pd.concat([warehouse_routes, self.data.level1_to_level1_inner])[
                ['start_id', 'end_id', 'sku']].drop_duplicates()
            warehouse_routes = warehouse_routes.merge(group, on='start_id', how='left')
            warehouse_routes = warehouse_routes.explode('group')
            warehouse_routes.loc[warehouse_routes['group'].notnull(), 'start_id'] = \
                warehouse_routes.loc[warehouse_routes['group'].notnull(), 'group']
            warehouse_routes = warehouse_routes[['start_id', 'end_id', 'sku']].drop_duplicates()
            warehouse_routes = warehouse_routes[warehouse_routes.start_id != warehouse_routes.end_id]
            self.data.available_routes = list(set(available_routes.set_index(['start_id', 'end_id', 'sku']).index.tolist()) & \
                                              set(self.data.warehouse_to_customer_cost.keys()))
            self.data.warehouse_routes = list(set(warehouse_routes.set_index(['start_id', 'end_id', 'sku']).index.tolist()) & \
                                              set(self.data.warehouse_transfer_cost.keys()))


            pickle.dump(
                (self.data.available_routes, self.data.warehouse_routes),
                open(self.solution_dir, 'wb')
            )
            self.plant_to_warehouse = plant_to_warehouse

            print("Phase One Model End")
            self.model_metric()
            return self.data

    def model_metric(self):
        print(f"--- model sparsity statistics ---")
        print(f"model sparsity, x_w: {len(self.data.warehouse_routes)}/{len(self.data.X_W)}")
        print(f"model sparsity, x_c: {len(self.data.available_routes)}/{len(self.data.X_C)}")
        print(f"---------------------------------")
        print(f"--- model product statistics ---")
        fake_product = set(self.plant_to_warehouse[
                (self.plant_to_warehouse.start_id == 'P000X') & (self.plant_to_warehouse.qty > 0)]) & \
        set(self.data.plant_sku_df[self.data.plant_sku_df.fac_id != 'P000X'].sku)
        print(f"model fake production, x_w: {len(fake_product)}")
