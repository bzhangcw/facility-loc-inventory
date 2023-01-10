import pandas as pd
from collections import namedtuple


def data_construct(model_dir):
    print("Data construct start ...")
    # ==================== 数据读取 ============================
    # 集合数据
    plant_df = pd.read_csv(model_dir + 'plant_df.csv')
    warehouse_df = pd.read_csv(model_dir + 'warehouse_df.csv')
    customer_df = pd.read_csv(model_dir + 'customer_df.csv')
    sku_df = pd.read_csv(model_dir + 'sku_df.csv')
    plant_sku_df = pd.read_csv(model_dir + 'plant_sku_df.csv')
    # 参数数据
    peak_cus_demand_monthly = pd.read_csv(model_dir + 'peak_cus_demand_monthly.csv')
    cus_demand_periodly = pd.read_csv(model_dir + 'cus_demand_periodly.csv')
    peak_warehouse_demand_monthly = pd.read_csv(model_dir + 'peak_warehouse_demand_monthly.csv')
    warehouse_demand_periodly = pd.read_csv(model_dir + 'warehouse_demand_periodly.csv')
    warehouse_capacity_monthly = pd.read_csv(model_dir + 'warehouse_capacity_monthly.csv')
    warehouse_capacity_periodly = pd.read_csv(model_dir + 'warehouse_capacity_periodly.csv')
    wh_outstorage_capacity_monthly = pd.read_csv(model_dir + 'wh_outstorage_capacity_monthly.csv')
    plant_info_monthly = pd.read_csv(model_dir + 'plant_info_monthly.csv')
    plant_to_warehouse_cost = pd.read_csv(model_dir + 'plant_to_warehouse_cost.csv')
    level1_to_level1_inner_cost = pd.read_csv(model_dir + 'level1_to_level1_inner_cost.csv')
    level1_to_level1_outer_cost = pd.read_csv(model_dir + 'level1_to_level1_outer_cost.csv')
    level1_to_level2_cost = pd.read_csv(model_dir + 'level1_to_level2_cost.csv')
    warehouse_to_customer_cost = pd.read_csv(model_dir + 'warehouse_to_customer_cost.csv')
    added_warehouse_cost = pd.read_csv(model_dir + 'added_warehouse_cost.csv')
    consider_end_inventory = pd.read_csv(model_dir + 'consider_end_inventory.csv')
    consider_init_inventory = pd.read_csv(model_dir + 'consider_init_inventory.csv')
    consider_inventory_days = pd.read_csv(model_dir + 'consider_inventory_days.csv')
    # 决策变量数据
    plant_to_warehouse = pd.read_csv(model_dir + 'plant_to_warehouse.csv')
    level1_to_level1_inner = pd.read_csv(model_dir + 'level1_to_level1_inner.csv')
    level1_to_level1_outer = pd.read_csv(model_dir + 'level1_to_level1_outer.csv')
    level1_to_level1_outer_map = pd.read_csv(model_dir + 'level1_to_level1_outer_map.csv')
    level1_to_level2 = pd.read_csv(model_dir + 'level1_to_level2.csv')
    warehouse_to_customer = pd.read_csv(model_dir + 'warehouse_to_customer.csv')

    # ================= 数据构造 =========================
    data = namedtuple("data", ['demand'])
    data.cus_demand_monthly = peak_cus_demand_monthly.set_index(['ka_id', 'sku'])['qty']
    data.wh_demand_monthly = peak_warehouse_demand_monthly.set_index(['fac_id', 'sku'])['qty']
    data.wh_demand_monthly_gp = peak_warehouse_demand_monthly.groupby('fac_id').qty.sum().reset_index(
    ).set_index(['fac_id'])['qty']
    data.wh_storage_capacity_monthly_total = warehouse_capacity_monthly.set_index(['fac_id'])['total_capacity']
    data.wh_storage_capacity_monthly_normal = warehouse_capacity_monthly.set_index(['fac_id'])['normal_capacity']
    data.wh_outbound_cap_monthly = wh_outstorage_capacity_monthly.set_index(['fac_id'])['capacity_max_monthly']
    data.wh_outbound_cap_periodly = wh_outstorage_capacity_monthly.set_index(['fac_id'])['capacity_max_periodly']
    data.line_prod_capcity_monthly = plant_info_monthly.set_index(['fac_id', 'line_id', 'sku'])[
        'capacity_monthly'].to_dict()
    data.line_prod_capcity_periodly = plant_info_monthly.set_index(['fac_id', 'line_id', 'sku'])[
        'capacity_periodly'].to_dict()
    data.line_prod_mpq = plant_info_monthly.set_index(['fac_id', 'line_id', 'sku'])[
        'min_size'].to_dict()

    # 成本参数
    data.plant_to_warehouse_cost = plant_to_warehouse_cost.set_index(['start_id', 'end_id', 'sku'])['total_cost']
    data.warehouse_transfer_cost = pd.concat([level1_to_level1_inner_cost, level1_to_level1_outer_cost,
                                             level1_to_level2_cost]).set_index(['start_id', 'end_id', 'sku'])['total_cost']
    data.warehouse_to_customer_cost = warehouse_to_customer_cost.set_index(['start_id', 'end_id', 'sku'])[
        'total_cost']
    data.line_prod_cost = plant_info_monthly.set_index(['fac_id', 'line_id', 'sku'])[
        'unit_cost'].to_dict()
    data.plant_prod_cost = plant_info_monthly.groupby(['fac_id', 'sku']).unit_cost.min().reset_index().set_index(
        ['fac_id', 'sku'])['unit_cost']
    data.added_warehouse_cost = added_warehouse_cost.set_index(['fac_id'])['rental_cost']
    data.end_inventory = consider_end_inventory.set_index(['fac_id', 'sku'])['qty'].to_dict()
    data.init_inventory = consider_init_inventory.set_index(['fac_id', 'sku'])['qty'].to_dict()
    tmp = consider_init_inventory.groupby(['fac_id']).qty.sum().reset_index()
    data.init_inventory_wh = tmp.set_index(['fac_id'])['qty'].to_dict()
    data.inventory_days = consider_inventory_days.set_index(['fac_id','sku']).inv_days.to_dict()
    # 时间参数
    ds_df = cus_demand_periodly[['ds']].drop_duplicates().sort_values(by='ds').reset_index(drop=True)
    ds_df['ds_id'] = ['period' + str(i).rjust(2, '0') for i in range(1, len(ds_df) + 1)]
    # 产线生产sku列表
    tmp = plant_info_monthly.groupby(['fac_id', 'line_id']).agg({'sku': lambda x: x.unique().tolist()}).reset_index()
    data.Ls = tmp.set_index(['fac_id', 'line_id'])['sku']
    # 各旬需求数据生成
    warehouse_demand_periodly_tmp = warehouse_demand_periodly.merge(ds_df, how='left', on='ds')
    cus_demand_periodly_tmp = cus_demand_periodly.merge(ds_df, how='left', on='ds')
    cus_demand_periodly_tmp = cus_demand_periodly_tmp[cus_demand_periodly_tmp.qty > 0]
    data.wh_demand_periodly = warehouse_demand_periodly_tmp.set_index(['fac_id', 'sku', 'ds_id'])['qty']
    data.wh_demand_periodly_gp = warehouse_demand_periodly_tmp.groupby(['fac_id', 'ds_id']).qty.sum().reset_index(
    ).set_index(['fac_id', 'ds_id'])['qty']
    data.cus_demand_periodly = cus_demand_periodly_tmp.set_index(['ka_id', 'sku', 'ds_id'])['qty']
    warehouse_capacity_periodly_tmp = warehouse_capacity_periodly.merge(ds_df, how='left', on='ds')
    data.wh_storage_capacity_periodly_total = warehouse_capacity_periodly_tmp.set_index(
        ['fac_id', 'ds_id'])['total_capacity']
    data.wh_storage_capacity_periodly_normal = warehouse_capacity_periodly_tmp.set_index(
        ['fac_id', 'ds_id'])['normal_capacity']

    # 集合数据
    data.P = plant_sku_df.fac_id.unique().tolist()
    data.I = warehouse_df.fac_id.unique().tolist()
    data.current_I = warehouse_df[warehouse_df.if_current == 1].fac_id.unique().tolist()
    data.added_I = warehouse_df[warehouse_df.if_current == 0].fac_id.unique().tolist()
    data.K = customer_df.ka_id.unique().tolist()
    data.S = sku_df.sku.unique().tolist()
    data.normal_S = list(set(sku_df.sku.unique()) - {'Y000168', 'Y000169', 'Y000170'})
    data.T = ds_df.ds_id.unique().tolist()
    data.T_t = ['period00'] + data.T
    data.I_1 = warehouse_df[warehouse_df.fac_type.apply(lambda x: 'level_1' in x)].fac_id.unique().tolist()
    data.I_2 = warehouse_df[warehouse_df.fac_type.apply(lambda x: 'level_2' in x)].fac_id.unique().tolist()
    data.S_0 = sku_df[sku_df.sku_category == '0'].sku.unique().tolist()
    # 决策变量数据
    warehouse_transfer_df = pd.concat([level1_to_level1_inner, level1_to_level1_outer, level1_to_level2])
    data.W = warehouse_df.set_index(['fac_id']).index.tolist()
    data.X_P = plant_to_warehouse.set_index(['start_id', 'end_id', 'sku']).index.tolist()
    data.X_W = warehouse_transfer_df.set_index(['start_id', 'end_id', 'sku']).index.tolist()
    data.X_W_inner = level1_to_level1_outer_map.set_index(['start_id', 'end_id', 'sku']).index.tolist()
    data.X_C = warehouse_to_customer.set_index(['start_id', 'end_id', 'sku']).index.tolist()
    data.Z_L = plant_info_monthly.set_index(['fac_id', 'line_id', 'sku']).index.tolist()
    data.Z_P = plant_info_monthly.set_index(['fac_id', 'sku']).index.tolist()

    data.warehouse_capacity_monthly = warehouse_capacity_monthly
    data.warehouse_demand_periodly_tmp = warehouse_demand_periodly_tmp
    data.level1_to_level1_outer_map = level1_to_level1_outer_map
    print("Data construct end !")
    return data
