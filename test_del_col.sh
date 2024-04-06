# 1 2 3 7 5; 13 14 15
export CG_RMP_USE_WS=0;
python template_run.py --if_del_col 0 &>./out/del_cols_log/0.log
python template_run.py --if_del_col 1 --del_col_freq 3 &>./out/del_cols_log/131.log
python template_run.py --if_del_col 1 --del_col_freq 2 &>./out/del_cols_log/121.log

