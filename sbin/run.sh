export resolve=$2
export target=$1
cmd="python -u main.py --phase1_resolve $resolve \
	--phase2_use_qty_heur 1 \
	--phase2_qty_heur_reset 1 \
	--phase2_use_full_model 3 \
	--result_dir $target \
	--phase2_new_fac_penalty 1e-6 \
	--phase2_inner_transfer_penalty 1 \
	--phase2_greedy_range 10000000 &> $target/run.log &"
echo $cmd;
# eval $cmd


