python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/us_generate_202403151725/' &> mps_log/us/ncs_e-4_us_generate_202403151725_7@8@0.log

python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/us_generate_202403152049/' &> mps_log/us/ncs_e-4_us_generate_202403152049_7@8@0.log

python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/sechina_202403152133/' &> mps_log/china/ncs_e-4_sechina_202403152133_7@8@0_without_warm.log

python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/sechina_202403130110/' &> mps_log/china/ncs_e-4_sechina_202403130110_7@8@0_without_warm.log
data/sechina_202403130110/

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403151725_7_7@8@0.mps &> mps_log/us/gurocl_new_guro_us_generate_202403151725_7_7@8@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403152049_7_7@8@0.mps &> mps_log/us/gurocl_new_guro_us_generate_202403152049_7_7@8@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_sechina_202403152133_7_7@8@0.mps &> mps_log/china/gurocl_new_guro_sechina_202403152133_7_7@8@0.mps.log


data/us_generate_202403151725/
python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/sechina_202403152155/' &> mps_log/china/ncs_e-4_sechina_202403152155_7@8@0.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_sechina_202403152155_7_7@8@0.mps &> mps_log/china/gurocl_new_guro_sechina_202403152155_7_7@8@0.mps.log

# python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/sechina_202403142305/' &> mps_log/us/ncs_e-4_sechina_202403142305_7@8@0.log

# python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/sechina_202403142315/' &> mps_log/us/ncs_e-4_sechina_202403142315_7@8@0.log

# python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/sechina_202403142326/' &> mps_log/us/ncs_e-4_sechina_202403142326_7@8@0.log


# python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/us_generate_202403151725/' &> mps_log/us/ncs_e-4_us_generate_202403151725_7@8@0.log


# new_guro_us_generate_202403151909_7_7@8@0.mps

# python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/us_generate_202403151909/' &> mps_log/us/new_guro_us_generate_202403151909_7_7@8@0.mps.log &

# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403151909_7_7@8@0.mps &> mps_log/us/new_guro_us_generate_202403151909_7_7@8@0.mps.log &


# 1955

# python facility-loc-inventory-template/ncs_for_template.py --fpath 'data/us_generate_202403151955/' &> mps_log/us/new_guro_us_generate_202403151955_7_7@8@0.mps.log &

# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403151955_7_7@8@0.mps &> mps_log/us/new_guro_us_generate_202403151955_7_7@8@0.mps.log &