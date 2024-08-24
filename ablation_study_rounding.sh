# export CG_RMP_USE_WS=0;

# export CG_RMP_METHOD=4;

export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --rounding_heuristic_4 1 &> ablation_study_rounding/rounding4/us_generate_202403151725_rounding4.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --rounding_heuristic_2 1 &> ablation_study_rounding/rounding2/us_generate_202403151725_2.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --cg_mip_recover 1 --cg_rmp_mip_iter 20 --cg_method_mip_heuristic 0 &> ablation_study_rounding/direct/us_generate_202403151725_0.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --rounding_heuristic_4 1 &> ablation_study_rounding/us_generate_202403151725_4.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403152049/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/us_generate_202403152049_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122258/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/us_generate_202403122258_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122342/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/us_generate_202403122342_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403130034/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/us_generate_202403130034_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152133/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/sechina_202403152133_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152155/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/sechina_202403152155_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142315/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/sechina_202403142315_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142326/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/sechina_202403142326_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130110/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &> ablation_study_rmp/algorithm4/sechina_202403130110_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130301/' --del_col_alg 4 --if_del_col 1 --column_pool_len 3 &>ablation_study_rmp/algorithm4/sechina_202403130301_0.log

# #### Algorithm 1

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403151725_0.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403152049/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403152049_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122258/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403122258_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122342/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403122342_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403130034/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/us_generate_202403130034_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152133/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403152133_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152155/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403152155_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142315/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403142315_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142326/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403142326_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130110/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm1/sechina_202403130110_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130301/' --del_col_alg 1 --if_del_col 1 --del_col_freq 3 &> ablation_study_rmp/algorithm1/sechina_202403130301_0.log

# #### Algorithm 2


# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403151725_0.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403152049/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403152049_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122258/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403122258_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122342/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403122342_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403130034/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/us_generate_202403130034_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152133/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403152133_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152155/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403152155_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142315/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403142315_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142326/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403142326_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130110/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm2/sechina_202403130110_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130301/' --del_col_alg 2 --if_del_col 1 --del_col_freq 3 &> ablation_study_rmp/algorithm2/sechina_202403130301_0.log

# #### Algorithm 3

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403151725/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/us_generate_202403151725_0.log 

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403152049/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/us_generate_202403152049_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122258/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/us_generate_202403122258_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403122342/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/us_generate_202403122342_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/us_generate_202403130034/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/us_generate_202403130034_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152133/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/sechina_202403152133_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403152155/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/sechina_202403152155_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142315/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/sechina_202403142315_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403142326/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/sechina_202403142326_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130110/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &>  ablation_study_rmp/algorithm3/sechina_202403130110_0.log

# export CG_RMP_USE_WS=0; export CG_RMP_METHOD=0; python -u facility-loc-inventory/ablation_test_rounding.py --fpath 'data/sechina_202403130301/' --del_col_alg 3 --if_del_col 1 --del_col_freq 3 &> ablation_study_rmp/algorithm3/sechina_202403130301_0.log
