

python -u main.py --phase1_resolve 0 --phase2_use_qty_heur 0 --phase2_use_full_model 3 &> 0221.log
python debug_this_result.py Y000169 T
python analyze_cost.py
