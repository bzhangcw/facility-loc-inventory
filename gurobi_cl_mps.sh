# pick_instance: 5/7/8/9
# conf_label: 1/2/3/5/6/8
# data_now:1219
# backorder : 0

# gurobi_cl TimeLimit=7200 Threads=4 MIPGap=0.01 Method=5 mps_log/us/new_guro_us_generate_202403122342_7_7@8@0.mps &> mps_log/us/gurobi_cl_new_guro_us_generate_202403122342_7_7@8@0.mps.log
# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_sechina_202403142305_7_7@8@0.mps &> mps_log/china/new_guro_sechina_202403142305_7_7@8@0.mps.log
# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_sechina_202403142315_7_7@8@0.mps &> mps_log/china/new_guro_sechina_202403142315_7_7@8@0.mps.log
# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_sechina_202403142315_7_7@8@0.mps &> mps_log/china/new_guro_sechina_202403142326_7_7@8@0.mps.log

# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403151324_7_7@8@0.mps &> mps_log/us/new_guro_us_generate_202403151324_7_7@8@0.mps.log

# gurobi_cl TimeLimit=20000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403151356_7_7@8@0.mps &> mps_log/us/new_guro_us_generate_202403151356_7_7@8@0.mps.log


# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_sechina_202403142326_7_7@8@0.mps &> mps_log/china/new_guro_sechina_202403142326_7_7@8@0.mps.log
# export CG_RMP_USE_WS=0;

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403122342_7_7@8@50@0.mps &> cus_test/new_guro_us_generate_202403122342_7_7@8@50@0.mps.log

# python facility-loc-inventory-template/template_run.py --fpath 'data/us_generate_202403122342/' --pick_instance 17 &> cus_test/ncs_e-5_us_generate_202403122342_7_7@8@50@0_withoutwarm.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403122342_7_7@8@100@0.mps &> cus_test/new_guro_us_generate_202403122342_7_7@8@100@0.mps.log

# python facility-loc-inventory-template/template_run.py --fpath 'data/us_generate_202403122342/' --pick_instance 18 &> cus_test/ncs_e-5_us_generate_202403122342_7_7@8@100@0_withoutwarm.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403122342_7_7@8@150@0.mps &> cus_test/new_guro_us_generate_202403122342_7_7@8@150@0.mps.log

# python facility-loc-inventory-template/template_run.py --fpath 'data/us_generate_202403122342/' --pick_instance 19 &> cus_test/ncs_e-5_us_generate_202403122342_7_7@8@200@0_withoutwarm.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 new_guro_us_generate_202403122342_7_7@8@150@0.mps &> cus_test/new_guro_us_generate_202403122342_7_7@8@150@0.mps.log

# python facility-loc-inventory-template/template_run.py --fpath 'data/us_generate_202403122342/' --pick_instance 20 &> cus_test/ncs_e-5_us_generate_202403122342_7_7@8@150@0_withoutwarm.mps.log


# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@17@50@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@17@50@0.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@18@100@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@18@100@0.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@19@150@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@19@150@0.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@20@200@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@20@200@0.mps.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@21@250@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@21@250@0.mps.log

# gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@22@300@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@22@300@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@23@350@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@23@350@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@24@400@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@24@400@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@25@450@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@25@450@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_sechina_202403142326_7_8@26@500@0.mps &> cus_test/new_guro_sechina_202403142326_7_8@26@500@0.mps.log

gurobi_cl TimeLimit=10000 Threads=4 MIPGap=0.01 Method=5 mps/new_guro_us_generate_202403122342_7_7@25@500@0.mps &> cus_test/new_guro_us_generate_202403122342_7_8@26@500@0.mps.log