#python test_final.py --lowerbound 0 --cp_lowerbound 0 --add_in_upper 0 --node_cost 0 --edge_cost 0 --nodelb 0

#python test_final.py --cp_lowerbound 0 --add_in_upper 0 --node_cost 0 --edge_cost 0 --nodelb 0
# python test_final.py --cus_num 519 --cp_lowerbound 1 --add_in_upper 0
# gurobi_cl TimeLimit=3600 finaltest/C519_L1_CL1_NU0_Fn0_Fe0_NL0.mps &> finaltest/C519_L1_CL1_NU0_Fn0_Fe0_NL0.mps.log
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

# python main_ncs_np.py --conf_label 1 --pick_instance 1 &> feasibility/c1p1.log
# python main_ncs_np.py --conf_label 1 --pick_instance 2 &> feasibility/c1p2.log
# python main_ncs_np.py --conf_label 1 --pick_instance 3 &> feasibility/c1p3.log
# python main_ncs_np.py --conf_label 1 --pick_instance 4 &> feasibility/c1p4.log
# python main_ncs_np.py --conf_label 1 --pick_instance 5 &> feasibility/c1p5.log
# python main_ncs_np.py --conf_label 1 --pick_instance 6 &> feasibility/c1p6.log

# python main_ncs_np.py --conf_label 2 --pick_instance 1 &> feasibility/c2p1.log
# python main_ncs_np.py --conf_label 2 --pick_instance 2 &> feasibility/c2p2.log
# python main_ncs_np.py --conf_label 2 --pick_instance 3 &> feasibility/c2p3.log
# python main_ncs_np.py --conf_label 2 --pick_instance 4 &> feasibility/c2p4.log
# python main_ncs_np.py --conf_label 2 --pick_instance 5 &> feasibility/c2p5.log
# python main_ncs_np.py --conf_label 2 --pick_instance 5 &> feasibility/c2p6.log

# python main_ncs_np.py --conf_label 3 --pick_instance 1 &> feasibility/c3p1.log
# python main_ncs_np.py --conf_label 3 --pick_instance 2 &> feasibility/c3p2.log
# python main_ncs_np.py --conf_label 3 --pick_instance 3 &> feasibility/c3p3.log
# python main_ncs_np.py --conf_label 3 --pick_instance 4 &> feasibility/c3p4.log
# python main_ncs_np.py --conf_label 3 --pick_instance 5 &> feasibility/c3p5.log
# python main_ncs_np.py --conf_label 3 --pick_instance 5 &> feasibility/c3p6.log

# python main_ncs_np.py --conf_label 4 --pick_instance 1 &> feasibility/c4p1.log
# python main_ncs_np.py --conf_label 4 --pick_instance 2 &> feasibility/c4p2.log
# python main_ncs_np.py --conf_label 4 --pick_instance 3 &> feasibility/c4p3.log
# python main_ncs_np.py --conf_label 4 --pick_instance 4 &> feasibility/c4p4.log
# python main_ncs_np.py --conf_label 4 --pick_instance 5 &> feasibility/c4p5.log
# python main_ncs_np.py --conf_label 4 --pick_instance 5 &> feasibility/c4p6.log

# python main_ncs_np.py --conf_label 5 --pick_instance 1 &> feasibility/c5p1.log
# python main_ncs_np.py --conf_label 5 --pick_instance 2 &> feasibility/c5p2.log
# python main_ncs_np.py --conf_label 5 --pick_instance 3 &> feasibility/c5p3.log
# python main_ncs_np.py --conf_label 5 --pick_instance 4 &> feasibility/c5p4.log
# python main_ncs_np.py --conf_label 5 --pick_instance 5 &> feasibility/c5p5.log
# python main_ncs_np.py --conf_label 5 --pick_instance 5 &> feasibility/c5p6.log

python main_ncs_np.py --conf_label 6 --pick_instance 6 &> feasibility/c6p6.log

# python main_ncs_np.py &> feasibility/c8p6_cg.log