#python test_final.py --lowerbound 0 --cp_lowerbound 0 --add_in_upper 0 --node_cost 0 --edge_cost 0 --nodelb 0

#python test_final.py --cp_lowerbound 0 --add_in_upper 0 --node_cost 0 --edge_cost 0 --nodelb 0
python test_final.py --cus_num 519 --cp_lowerbound 1 --add_in_upper 0
gurobi_cl TimeLimit=3600 finaltest/C519_L1_CL1_NU0_Fn0_Fe0_NL0.mps &> finaltest/C519_L1_CL1_NU0_Fn0_Fe0_NL0.mps.log
#python test_final.py --cp_lowerbound 1 --add_in_upper 1
#python test_final.py --cp_lowerbound 1 --add_in_upper 1 --node_cost 1
#python test_final.py --cp_lowerbound 1 --add_in_upper 1 --edge_cost 1
#python test_final.py --cp_lowerbound 1 --add_in_upper 1 --node_cost 1 --edge_cost 1
#python test_final.py --cp_lowerbound 1 --add_in_upper 1 --node_cost 1 --edge_cost 1 --add_in_upper 1 --node_cost 1 --edge_cost 1 --nodelb 1
#gurobi_cl TimeLimit=3600 finaltest/CL1_NU0_Fn0_Fe0.mps &> finaltest/CL1_NU0_Fn0_Fe0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/CL1_NU1_Fn0_Fe0.mps &> finaltest/CL1_NU1_Fn0_Fe0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/CL1_NU1_Fn1_Fe0.mps &> finaltest/CL1_NU1_Fn1_Fe0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/CL1_NU1_Fn0_Fe1.mps &> finaltest/CL1_NU1_Fn0_Fe1.mps.log
#gurobi_cl TimeLimit=3600 finaltest/CL1_NU1_Fn1_Fe1.mps &> finaltest/CL1_NU1_Fn1_Fe1_NL1.mps.log
#gurobi_cl TimeLimit=3600 finaltest/CL0_NU0_Fn0_Fe0_NL0.mps &> finaltest/CL0_NU0_Fn0_Fe0_NL0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/L0_CL0_NU0_Fn0_Fe0_NL0.mps &> finaltest/L0_CL0_NU0_Fn0_Fe0_NL0.mps.log

#python test_final.py --cus_num 100 --cp_lowerbound 1 --add_in_upper 1  --node_cost 0 --edge_cost 0
#python test_final.py --cus_num 100 --cp_lowerbound 1 --add_in_upper 1  --node_cost 1 --edge_cost 1
#python test_final.py --cus_num 100 --cp_lowerbound 1 --add_in_upper 1  --node_cost 1 --edge_cost 1 --nodelb 1
#python test_final.py --cus_num 200 --cp_lowerbound 1 --add_in_upper 1  --node_cost 0 --edge_cost 0
#python test_final.py --cus_num 200 --cp_lowerbound 1 --add_in_upper 1  --node_cost 1 --edge_cost 1
#python test_final.py --cus_num 200 --cp_lowerbound 1 --add_in_upper 1  --node_cost 1 --edge_cost 1 --nodelb 1

#gurobi_cl TimeLimit=3600 finaltest/C100_L1_CL1_NU1_Fn0_Fe0_NL0.mps &> finaltest/C100_L1_CL1_NU1_Fn0_Fe0_NL0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/C100_L1_CL1_NU1_Fn1_Fe1_NL0.mps &> finaltest/C100_L1_CL1_NU1_Fn1_Fe1_NL0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/C100_L1_CL1_NU1_Fn1_Fe1_NL1.mps &> finaltest/C100_L1_CL1_NU1_Fn1_Fe1_NL1.mps.log
#gurobi_cl TimeLimit=3600 finaltest/C200_L1_CL1_NU1_Fn0_Fe0_NL0.mps &> finaltest/C200_L1_CL1_NU1_Fn0_Fe0_NL0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/C200_L1_CL1_NU1_Fn1_Fe1_NL0.mps &> finaltest/C200_L1_CL1_NU1_Fn1_Fe1_NL0.mps.log
#gurobi_cl TimeLimit=3600 finaltest/C200_L1_CL1_NU1_Fn1_Fe1_NL1.mps &> finaltest/C200_L1_CL1_NU1_Fn1_Fe1_NL1.mps.log