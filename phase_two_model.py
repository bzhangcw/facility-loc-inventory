import json
import pickle

import coptpy
import numpy as np
import pandas as pd
from coptpy import *

from utils import *
import helpers as mh


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
        self.obj_map = _values = {
            k: np.round(v.getValue(), 4) for k, v in self.obj_expr_map.items()
        }
        _values_repr = {k: f"{v:.2e}" for k, v in _values.items()}
        print("--- objective summary ---")
        self.obj_map_str = json.dumps(_values_repr, indent=3)
        print(self.obj_map_str)

    @timer
    def load_phasei_from_local(self, path):
        print(f"Loading Phase-I Solution @{path}")
        self.data.available_routes, self.data.warehouse_routes = pickle.load(
            open(path, "rb")
        )

    def routing_heuristics(self, maximum_cardinality=3):
        """
        A heuristic for adding nearby routes
        :return:
        """
        df_cost = pd.Series(self.data.warehouse_to_customer_cost).reset_index()
        cus_sku_tuples = set(((i, j) for (i, j, t) in self.data.cus_demand_periodly))
        greedy_routes = df_cost.groupby(["level_1", "level_2"]).apply(
            lambda x: sorted(zip(x["level_0"], x[0]), key=lambda y: y[-1])[
                :maximum_cardinality
            ]
        )[cus_sku_tuples]
        for (
            (j, k),
            vv,
        ) in greedy_routes.items():  # T0025仓的运输成本是按箱计算的，即使是深圳客户，该仓也无法进入成本前三，增加该条线路
            upstream_facs = [v[0] for v in vv]
            if "T0015" in upstream_facs or "T0030" in upstream_facs:
                if ("T0025", j, k) in self.data.warehouse_to_customer_cost:
                    vv.append(("T0025", 1))
        return [(i, j, k) for (j, k), vv in greedy_routes.items() for i, c in vv]

    @timer
    def build(self):
        print("Phase Two Model Start ...")
        # Create COPT environment
        env = Envr()

        # Create COPT model
        model: coptpy.Model = env.createModel("phase_two")
        self.model = model
        # ================== 添加决策变量 =====================
        z_p = model.addVars(
            self.data.Z_P, self.data.T, nameprefix="z_p", vtype=COPT.CONTINUOUS
        )  # 工厂生产量
        z_l = model.addVars(
            self.data.Z_L, self.data.T, nameprefix="z_l", vtype=COPT.CONTINUOUS
        )  # 产线生产量
        x_p = model.addVars(
            self.data.X_P, self.data.T, nameprefix="x_p", vtype=COPT.CONTINUOUS
        )  # 工厂->仓库 线路运输量
        x_w = mh.addvars_xw(self, model)
        x_c = mh.addvars_xc(self, model)

        inv = model.addVars(
            self.data.I,
            self.data.S,
            self.data.T_t,
            nameprefix="inv",
            vtype=COPT.CONTINUOUS,
        )  # 库存量
        inv_avail = model.addVars(
            self.data.I,
            self.data.S,
            self.data.T_t,
            nameprefix="inva",
            vtype=COPT.CONTINUOUS,
        )  # 可用库存量
        surplus_inv = model.addVars(
            self.data.I,
            self.data.S,
            self.data.T_t,
            nameprefix="invs",
            vtype=COPT.CONTINUOUS,
        )
        storage_gap = model.addVars(
            self.data.I, self.data.T_t, nameprefix="storage_gap", vtype=COPT.CONTINUOUS
        )
        inv_f = model.addVars(
            self.data.I,
            self.data.S,
            self.data.T_t,
            nameprefix="invf",
            vtype=COPT.CONTINUOUS,
        )  # 丢弃库存量
        inv_gap = model.addVars(
            self.data.I,
            self.data.S,
            self.data.T_t,
            nameprefix="gap",
            vtype=COPT.CONTINUOUS,
        )  # 需求缺口量
        end_inv_gap = model.addVars(
            self.data.I, self.data.S, nameprefix="gap", vtype=COPT.CONTINUOUS
        )  # 期末库存缺口量

        self.variables = (
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
        )

        # ================== 添加业务约束 =====================

        # ====== 变量间耦合关系 =====
        # 工厂生产量=工厂内各产线生产量之和
        model.addConstrs(
            (z_l.sum(p, "*", s, t) == z_p[(p, s, t)] for p, s, t in z_p),
            nameprefix="line_product",
        )

        # 工厂生产量=运输到各仓库量之和
        model.addConstrs(
            (x_p.sum(p, "*", s, t) == z_p[(p, s, t)] for p, s, t in z_p),
            nameprefix="transport",
        )

        # ====== 工厂生产约束 =====
        # 产线产能约束
        for t in self.data.T:
            for (p, l), s_list in self.data.Ls.items():
                model.addConstr(
                    (
                        quicksum(
                            z_l[(p, l, s, t)]
                            / self.data.line_prod_capcity_periodly[(p, l, s)]
                            for s in s_list
                        )
                        <= self.data.line_utilization[(p, l, t)]
                    ),
                    name="line_capacity",
                )

        # 代工厂最小生产量约束
        model.addConstrs(
            z_l.sum(p, l, "*", "*") >= self.data.min_production[(p, l)]
            for p, l in self.data.min_production
        )

        # ====== 客户需求满足约束 =====
        for k, s, t in self.data.KST:
            model.addConstr(
                (
                    x_c.sum("*", k, s, t)
                    == self.data.cus_demand_periodly.get((k, s, t), 0)
                ),
                name="cus_demand",
            )

        # @todo,
        # why do we even need this.
        model.addConstr(
            quicksum(
                x_w[i, j, s, t]
                for i, j, s, t in x_w
                if (i, j) in self.data.weird_pairs 
            ) <= 2e5
        )
        
        mh.add_inventory_constr(self)

        # ================== 添加目标函数 =====================
        self.obj_expr_map = dict(
            cost_p2w=quicksum(
                x_p[p, i, s, t]
                * (
                    self.data.added_warehouse_cost.get(i, 0)
                    * DEFAULT_ALG_PARAMS.phase2_new_fac_penalty
                    + self.data.plant_to_warehouse_cost[p, i, s]
                )
                for p, i, s, t in x_p
            ),
            cost_w2w=quicksum(
                x_w[i, j, s, t]
                * (
                    self.data.added_warehouse_cost.get(i, 0)
                    * DEFAULT_ALG_PARAMS.phase2_new_fac_penalty
                    + self.data.warehouse_transfer_cost[i, j, s]
                )
                for i, j, s, t in x_w
                if (i, j) not in self.data.weird_pairs
            ),
            cost_w2w_weird=DEFAULT_ALG_PARAMS.phase2_inner_transfer_penalty
            * quicksum(
                x_w[i, j, s, t]
                * (
                    self.data.added_warehouse_cost.get(i, 0)
                    * DEFAULT_ALG_PARAMS.phase2_new_fac_penalty
                    + self.data.warehouse_transfer_cost[i, j, s]
                )
                for i, j, s, t in x_w
                if (i, j) in self.data.weird_pairs 
            ),
            cost_w2c=quicksum(
                x_c[i, k, s, t]
                * (
                    self.data.added_warehouse_cost.get(i, 0)
                    * DEFAULT_ALG_PARAMS.phase2_new_fac_penalty
                    + self.data.warehouse_to_customer_cost[(i, k, s)]
                )
                # todo, why
                for i, k, s, t in x_c
            ),
            cost_prod=quicksum(
                z_l[p, l, s, t] * self.data.line_prod_cost[p, l, s]
                for p, l, s, t in z_l
            ),
            cost_inv_gap=quicksum(inv_gap[i, s, t] * 5000 for i, s, t in inv_gap),
            cost_inv_gap_end=quicksum(end_inv_gap[i, s] * 5000 for i, s in end_inv_gap),
            cost_surplus=quicksum(
                surplus_inv[i, s, t] * 100000
                for (idx, t) in enumerate(self.data.T_t)
                for i in self.data.I
                for s in self.data.S
                if idx >= 4
            ),
            cost_storage=quicksum(storage_gap[i, t] * 500 for i, t in storage_gap),
            cost_inv=quicksum(inv[i, s, t] for i, s, t in inv),
            cost_giveup=200 * quicksum(inv_f[i, s, t] for i, s, t in inv_f),
        )
        self.obj_expr = obj = sum(self.obj_expr_map.values())

        # 期末库存不足惩罚/多余备货惩罚/库存成本，防止提前备货
        model.setObjective(obj, sense=COPT.MINIMIZE)

        model.setParam(COPT.Param.TimeLimit, DEFAULT_ALG_PARAMS.timelimit)
        model.setParam(COPT.Param.LpMethod, DEFAULT_ALG_PARAMS.phase2_lpmethod)
        model.setParam(COPT.Param.FeasTol, 1e-5)
        model.setParam(COPT.Param.DualTol, 1e-5)
        model.write("phase_two_lp_2.lp")

    @timer
    def run(self, finalize=True):
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
        ) = self.variables

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
            mh.qty_fix_heuristic(self)

        self.obj_visualize()
        # ================== 结果输出 =====================
        if model.status == COPT.INFEASIBLE:
            print("Phase Two Model Infeasible")
            model.computeIIS()
            model.writeIIS(self.model_dir + "iis_two.ilp")
        else:
            print("Phase Two Model Optimal")
            plant_product = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, z_p))
            ).reset_index()
            plant_product.columns = ["plant_id", "sku", "period", "qty"]
            plant_product["qty"] = plant_product["qty"].apply(lambda x: round(x, 3))
            plant_line_product = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, z_l))
            ).reset_index()
            plant_line_product.columns = ["plant_id", "line_id", "sku", "period", "qty"]
            plant_line_product["qty"] = plant_line_product["qty"].apply(
                lambda x: round(x, 3)
            )
            # @note: added by C.Z, only keep > 0 qty
            # plant_line_product = plant_line_product.query("qty > 0")
            plant_line_product["min_prod"] = plant_line_product.apply(
                lambda row: self.data.line_prod_mpq.get(
                    (row["plant_id"], row["line_id"], row["sku"]), 0
                ),
                axis=1,
            )

            # if_plant_line_product = pd.DataFrame(pd.Series(model.getInfo(COPT.Info.Value, y_l))).reset_index()
            # if_plant_line_product.columns=['plant_id', 'line_id', 'sku', 'period', 'if_product']

            plant_to_warehouse = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, x_p))
            ).reset_index()
            plant_to_warehouse.columns = ["start_id", "end_id", "sku", "period", "qty"]
            plant_to_warehouse["qty"] = plant_to_warehouse["qty"].apply(
                lambda x: round(x, 3)
            )
            warehouse_to_warehouse = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, x_w))
            ).reset_index()
            warehouse_to_warehouse.columns = [
                "start_id",
                "end_id",
                "sku",
                "period",
                "qty",
            ]
            warehouse_to_warehouse["qty"] = warehouse_to_warehouse["qty"].apply(
                lambda x: round(x, 3)
            )
            warehouse_to_customer = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, x_c))
            ).reset_index()
            warehouse_to_customer.columns = [
                "start_id",
                "end_id",
                "sku",
                "period",
                "qty",
            ]
            warehouse_to_customer["qty"] = warehouse_to_customer["qty"].apply(
                lambda x: round(x, 3)
            )

            inventory_gap = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, inv_gap))
            ).reset_index()
            inventory_gap.columns = ["warehouse_id", "sku", "period", "inv_gap"]
            inventory = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, inv))
            ).reset_index()
            inventory.columns = ["warehouse_id", "sku", "period", "inv"]
            inventory["inv"] = inventory["inv"].apply(lambda x: round(x, 3))
            inventory_dropped = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, inv_f))
            ).reset_index()
            inventory_dropped.columns = ["warehouse_id", "sku", "period", "inv_dropped"]
            inventory_dropped["inv_dropped"] = inventory_dropped["inv_dropped"].apply(
                lambda x: round(x, 3)
            )
            inventory_avail = pd.DataFrame(
                pd.Series(model.getInfo(COPT.Info.Value, inv_avail))
            ).reset_index()
            inventory_avail.columns = ["warehouse_id", "sku", "period", "inv_avail"]
            inventory_avail["inv_avail"] = inventory_avail["inv_avail"].apply(
                lambda x: round(x, 3)
            )

            # ======= 统计仓库*sku*period的库存变动情况
            plant_to_warehouse_in = plant_to_warehouse.rename(
                columns={"end_id": "warehouse_id", "qty": "prod_in"}
            )
            plant_to_warehouse_in = (
                plant_to_warehouse_in.groupby(["warehouse_id", "sku", "period"])
                .prod_in.sum()
                .reset_index()
            )
            warehouse_to_warehouse_in = warehouse_to_warehouse.rename(
                columns={"end_id": "warehouse_id", "qty": "transfer_in"}
            )
            warehouse_to_warehouse_in = (
                warehouse_to_warehouse_in.groupby(["warehouse_id", "sku", "period"])
                .transfer_in.sum()
                .reset_index()
            )

            warehouse_to_warehouse_out = warehouse_to_warehouse.rename(
                columns={"start_id": "warehouse_id", "qty": "transfer_out"}
            )
            warehouse_to_warehouse_out = (
                warehouse_to_warehouse_out.groupby(["warehouse_id", "sku", "period"])
                .transfer_out.sum()
                .reset_index()
            )

            # transfer_out_cap：仓间调拨计入出货量的部分
            warehouse_to_warehouse_tmp = warehouse_to_warehouse.merge(
                self.data.level1_to_level1_outer_map,
                on=["start_id", "end_id", "sku"],
                how="left",
            )
            warehouse_to_warehouse_tmp = warehouse_to_warehouse_tmp[
                warehouse_to_warehouse_tmp.tag.isnull()
            ]
            warehouse_to_warehouse_out_cap = warehouse_to_warehouse_tmp.rename(
                columns={"start_id": "warehouse_id", "qty": "transfer_out_cap"}
            )
            warehouse_to_warehouse_out_cap = (
                warehouse_to_warehouse_out_cap.groupby(
                    ["warehouse_id", "sku", "period"]
                )
                .transfer_out_cap.sum()
                .reset_index()
            )
            warehouse_to_customer_out = warehouse_to_customer.rename(
                columns={"start_id": "warehouse_id", "qty": "cus_out"}
            )
            warehouse_to_customer_out = (
                warehouse_to_customer_out.groupby(["warehouse_id", "sku", "period"])
                .cus_out.sum()
                .reset_index()
            )
            warehouse_demand_out = self.data.warehouse_demand_periodly_tmp.rename(
                columns={"fac_id": "warehouse_id", "qty": "wh_out", "ds_id": "period"}
            )
            warehouse_demand_out = (
                warehouse_demand_out.groupby(["warehouse_id", "sku", "period"])
                .wh_out.sum()
                .reset_index()
            )

            # 构造完整库存数据表
            tmp1 = inventory.merge(plant_to_warehouse_in, how="left")
            tmp2 = tmp1.merge(warehouse_to_warehouse_in, how="left")
            tmp3 = tmp2.merge(warehouse_to_warehouse_out, how="left")
            tmp4 = tmp3.merge(warehouse_to_customer_out, how="left")
            tmp4 = tmp4.merge(warehouse_to_warehouse_out_cap, how="left")
            tmp5 = tmp4.merge(warehouse_demand_out, how="left")
            tmp6 = tmp5.merge(inventory_gap, how="left")
            tmp6 = tmp6.merge(inventory_dropped, how="left")
            tmp6 = tmp6.merge(inventory_avail, how="left")
            tmp6["inv_pre"] = tmp6.groupby(["warehouse_id", "sku"]).inv.shift(1)
            tmp6 = tmp6.fillna(0)
            tmp6["storage"] = (
                tmp6.inv_pre.apply(lambda x: max(x, 0))
                + tmp6.prod_in
                + tmp6.transfer_in
                - 1 * (tmp6.transfer_out + tmp6.cus_out + tmp6.wh_out)
            )

            plant_line_product.to_csv(
                self.model_dir + "plant_line_product.csv", index=False
            )
            plant_to_warehouse.to_csv(
                self.model_dir + "plant_to_warehouse.csv", index=False
            )
            warehouse_to_warehouse.to_csv(
                self.model_dir + "warehouse_to_warehouse.csv", index=False
            )
            warehouse_to_customer.to_csv(
                self.model_dir + "warehouse_to_customer.csv", index=False
            )
            tmp6.to_csv(self.model_dir + "warehouse_inv_change.csv", index=False)

            # utility of warehouse
            new_wh = set(
                self.data.warehouse_df.query("if_current == 0")["fac_id"].unique()
            )
            tmp6_gp = (
                tmp6.groupby(["warehouse_id", "period"])
                .agg(
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
                )
                .reset_index()
            )
            tmp = tmp6_gp.merge(
                self.data.wh_storage_capacity_periodly_total.reset_index().rename(
                    columns={"fac_id": "warehouse_id", "ds_id": "period"}
                ),
                how="left",
                on=["warehouse_id", "period"],
            )

            tmp = tmp.assign(
                cap_utility=lambda df: df.storage.apply(lambda x: max(x, 0))
                / df.total_capacity,
                outbound=lambda df: 0.5 * (df.transfer_out + df.cus_out + df.wh_out),
                outbound_cap=lambda df: df["warehouse_id"].apply(
                    lambda i: self.data.wh_outbound_cap_periodly[i]
                ),
                outbound_cap_utility=lambda df: df.outbound / df.outbound_cap,
                bool_is_new=lambda df: df["warehouse_id"].apply(lambda x: x in new_wh),
            )
            tmp.to_csv(self.model_dir + "warehouse_io_cap.csv", index=False)

            self.plant_to_warehouse = plant_to_warehouse

            self.model_metric()
        print("Phase Two Model End")
        mh.clear_integer_fixings(self, z_l)

    def model_metric(self):
        print(f"--- model product statistics ---")
        fake_product = set(
            self.plant_to_warehouse[
                (self.plant_to_warehouse.start_id == "P000X")
                & (self.plant_to_warehouse.qty > 0)
            ]
        ) & set(self.data.plant_sku_df[self.data.plant_sku_df.fac_id != "P000X"].sku)
        print(f"model fake production, x_w: {len(fake_product)}")
