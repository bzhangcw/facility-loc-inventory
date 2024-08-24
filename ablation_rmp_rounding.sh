#### RMP Algorithm ###

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403151725/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403151725_0.log 
# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142326/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding4/sechina_202403142326_rounding4_V2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403152049/' --del_col_alg 0 &>  ncs_us_generate_202403152049_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403122258/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403122258_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403122342/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403122342_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403130034/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403130034_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403152133/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403152133_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403152155/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403152155_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403142315/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403142315_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403142326/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403142326_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403130110/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403130110_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403130301/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &> ablation_study_rmp/algorithm1/sechina_202403130301_0.log

#### Rounding Algorithm ###

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/us_generate_202403151725_rounding4_V2.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403152049/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/us_generate_202403152049_rounding4_V2.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122258/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/us_generate_202403122258_rounding4.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122342/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/us_generate_202403122342_rounding4.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403130034/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/us_generate_202403130034_rounding4_recovered.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152133/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/sechina_202403152133_rounding4_recovered.log
# # 上次停在这
# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152155/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/sechina_202403152155_rounding4_recovered2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142315/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/sechina_202403142315_rounding4_recovered2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142326/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/sechina_202403142326_rounding4_recovered2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130110/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/sechina_202403130110_rounding4.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130301/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding5/sechina_202403130301_rounding4_recovered.log

#### RMP Algorithm 2

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403151725/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403151725_0.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403152049/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403152049_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403122258/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403122258_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403122342/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 --check_num 3 &>  ablation_study_rmp/algorithm2/us_generate_202403122342_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403130034/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403130034_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403152133/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403152133_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403152155/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403152155_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403142315/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403142315_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403142326/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403142326_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403130110/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403130110_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403130301/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &> ablation_study_rmp/algorithm2/sechina_202403130301_0.log


## Parameter

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403151725/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &>  ablation_study_rmp/algorithm1/us_generate_202403151725_0_2.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403152049/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &> ablation_study_rmp/algorithm1/us_generate_202403152049_0_2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403122258/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &>  ablation_study_rmp/algorithm1/us_generate_202403122258_0_2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403122342/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &>  ablation_study_rmp/algorithm1/us_generate_202403122342_0_2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403130034/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &>  ablation_study_rmp/algorithm1/us_generate_202403130034_0_2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403152133/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &>  ablation_study_rmp/algorithm1/sechina_202403152133_0_2.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/sechina_202403152155/' --del_col_alg 1 --if_del_col 1 --del_col_freq 2 &>  ablation_study_rmp/algorithm1/sechina_202403152155_0_2.log


export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403151725/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/us_generate_202403151725_0.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test.py --fpath 'data/us_generate_202403152049/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403152049_0.log