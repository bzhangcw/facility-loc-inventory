# 1 2 3 7 5; 13 14 15
export CG_RMP_USE_WS=0;
python template_run.py --if_del_col 0 &>./out/del_cols_log/0.log
python template_run.py --if_del_col 1 --del_col_freq 3 &>./out/del_cols_log/131.log
python template_run.py --if_del_col 1 --del_col_freq 2 &>./out/del_cols_log/121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403151725/' --if_del_col 0 &>output/del_cols_log/us_generate_202403151725_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403151725/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/us_generate_202403151725_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403151725/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/us_generate_202403151725_121.log

# 2049解不出来？
python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403152049/' --if_del_col 0 &>output/del_cols_log/us_generate_202403152049_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403152049/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/us_generate_202403152049_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403152049/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/us_generate_202403152049_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403122258/' --if_del_col 0 &>output/del_cols_log/us_generate_202403122258_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403122258/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/us_generate_202403122258_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403122258/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/us_generate_202403122258_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403122342/' --if_del_col 0 &>output/del_cols_log/us_generate_202403122342_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403122342/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/us_generate_202403122342_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403122342/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/us_generate_202403122342_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403130034/' --if_del_col 0 &>output/del_cols_log/us_generate_202403130034_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403130034/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/us_generate_202403130034_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/us_generate_202403130034/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/us_generate_202403130034_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403152133/' --if_del_col 0 &>output/del_cols_log/sechina_202403152133_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403152133/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/sechina_202403152133_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403152133/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/sechina_202403152133_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403152155/' --if_del_col 0 &>output/del_cols_log/sechina_202403152155_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403152155/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/sechina_202403152155_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403152155/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/sechina_202403152155_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403142315/' --if_del_col 0 &>output/del_cols_log/sechina_202403142315_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403142315/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/sechina_202403142315_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403142315/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/sechina_202403142315_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403142326/' --if_del_col 0 &>output/del_cols_log/sechina_202403142326_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403142326/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/sechina_202403142326_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403142326/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/sechina_202403142326_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403130110/' --if_del_col 0 &>output/del_cols_log/sechina_202403130110_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403130110/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/sechina_202403130110_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403130110/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/sechina_202403130110_121.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403130301/' --if_del_col 0 &>output/del_cols_log/sechina_202403130301_0.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403130301/' --if_del_col 1 --del_col_freq 3 &>output/del_cols_log/sechina_202403130301_131.log

python facility-loc-inventory/ncs_for_template.py --fpath 'data/sechina_202403130301/'--if_del_col 1 --del_col_freq 2 &>output/del_cols_log/sechina_202403130301_121.log
