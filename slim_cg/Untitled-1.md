[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:33:44,563: -- The FACINV python package --
[facinv] 2023-12-14 19:33:44,563:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:33:44,563: :solution      to ./out
[facinv] 2023-12-14 19:33:44,563: :data          to ./data
[facinv] 2023-12-14 19:33:44,563: :logs and tmps to ./tmp
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 5, in <module>
    from dnp_model import DNP
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 16, in <module>
    import update_constr
ModuleNotFoundError: No module named 'update_constr'

[Done] exited with code=1 in 1.176 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:34:13,977: -- The FACINV python package --
[facinv] 2023-12-14 19:34:13,978:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:34:13,978: :solution      to ./out
[facinv] 2023-12-14 19:34:13,978: :data          to ./data
[facinv] 2023-12-14 19:34:13,978: :logs and tmps to ./tmp
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 5, in <module>
    from dnp_model import DNP
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 18, in <module>
    from update_constr import *
ModuleNotFoundError: No module named 'update_constr'

[Done] exited with code=1 in 0.992 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:34:36,358: -- The FACINV python package --
[facinv] 2023-12-14 19:34:36,358:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:34:36,358: :solution      to ./out
[facinv] 2023-12-14 19:34:36,358: :data          to ./data
[facinv] 2023-12-14 19:34:36,358: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:34:36,651: generating the signature of this problem
[facinv] 2023-12-14 19:34:36,651: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:34:36,651: current data has been generated before
[facinv] 2023-12-14 19:34:36,651: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:34:36,682: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:34:36,703: initialization complete, start generating columns...
[facinv] 2023-12-14 19:34:36,704: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 67%|██████▋   | 10/15 [00:00<00:00, 91.46it/s]
100%|██████████| 15/15 [00:00<00:00, 91.49it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 0.995 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:34:51,490: -- The FACINV python package --
[facinv] 2023-12-14 19:34:51,490:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:34:51,490: :solution      to ./out
[facinv] 2023-12-14 19:34:51,490: :data          to ./data
[facinv] 2023-12-14 19:34:51,490: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:34:51,738: generating the signature of this problem
[facinv] 2023-12-14 19:34:51,738: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:34:51,738: current data has been generated before
[facinv] 2023-12-14 19:34:51,738: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:34:51,762: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:34:51,776: initialization complete, start generating columns...
[facinv] 2023-12-14 19:34:51,776: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 67%|██████▋   | 10/15 [00:00<00:00, 96.47it/s]
100%|██████████| 15/15 [00:00<00:00, 94.35it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/ncg/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 0.938 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/ncg/tempCodeRunnerFile.py"
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/ncg/tempCodeRunnerFile.py", line 1, in <module>
    c
NameError: name 'c' is not defined

[Done] exited with code=1 in 0.018 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:35:11,754: -- The FACINV python package --
[facinv] 2023-12-14 19:35:11,755:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:35:11,755: :solution      to ./out
[facinv] 2023-12-14 19:35:11,755: :data          to ./data
[facinv] 2023-12-14 19:35:11,755: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:35:12,128: generating the signature of this problem
[facinv] 2023-12-14 19:35:12,128: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:35:12,128: current data has been generated before
[facinv] 2023-12-14 19:35:12,128: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:35:12,160: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:35:12,181: initialization complete, start generating columns...
[facinv] 2023-12-14 19:35:12,181: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 67%|██████▋   | 10/15 [00:00<00:00, 94.87it/s]
100%|██████████| 15/15 [00:00<00:00, 94.90it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/ncg/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 1.347 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:35:35,991: -- The FACINV python package --
[facinv] 2023-12-14 19:35:35,991:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:35:35,991: :solution      to ./out
[facinv] 2023-12-14 19:35:35,991: :data          to ./data
[facinv] 2023-12-14 19:35:35,991: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:35:36,362: generating the signature of this problem
[facinv] 2023-12-14 19:35:36,362: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:35:36,362: current data has been generated before
[facinv] 2023-12-14 19:35:36,363: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:35:36,393: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:35:36,415: initialization complete, start generating columns...
[facinv] 2023-12-14 19:35:36,415: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 60%|██████    | 9/15 [00:00<00:00, 87.88it/s]
100%|██████████| 15/15 [00:00<00:00, 89.16it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/hyper_cg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/hyper_cg/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/hyper_cg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 1.325 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py"
[facinv] 2023-12-14 19:35:40,605: -- The FACINV python package --
[facinv] 2023-12-14 19:35:40,605:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:35:40,605: :solution      to ./out
[facinv] 2023-12-14 19:35:40,605: :data          to ./data
[facinv] 2023-12-14 19:35:40,605: :logs and tmps to ./tmp
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py", line 7, in <module>
    from slim_cg.slim_rmp_model import DNPSlim
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_rmp_model.py", line 6, in <module>
    class DNPSlim(DNP):
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_rmp_model.py", line 32, in DNPSlim
    customer_list: List[Customer] = None,
NameError: name 'Customer' is not defined

[Done] exited with code=1 in 0.729 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py"
[facinv] 2023-12-14 19:36:23,179: -- The FACINV python package --
[facinv] 2023-12-14 19:36:23,180:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:36:23,180: :solution      to ./out
[facinv] 2023-12-14 19:36:23,180: :data          to ./data
[facinv] 2023-12-14 19:36:23,180: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:36:23,224: generating the signature of this problem
[facinv] 2023-12-14 19:36:23,224: {
  "data_dir": "data/data_0401_V4.xlsx",
  "one_period": true
}
[facinv] 2023-12-14 19:36:23,224: current data has been generated before
[facinv] 2023-12-14 19:36:23,224: reading from cache: ./out/data_0401_V4-True.pk
----------DCG Model------------
[facinv] 2023-12-14 19:36:23,798: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
[facinv] 2023-12-14 19:36:24,544: initialization complete, start generating columns...
[facinv] 2023-12-14 19:36:24,544: generating column oracles

  0%|          | 0/472 [00:00<?, ?it/s]use COPT to build and solve model
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  1%|          | 5/472 [00:00<00:09, 48.57it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  3%|▎         | 13/472 [00:00<00:07, 61.84it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  4%|▍         | 21/472 [00:00<00:06, 69.50it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  6%|▌         | 28/472 [00:00<00:08, 52.81it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  7%|▋         | 35/472 [00:00<00:07, 57.63it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 10%|▉         | 45/472 [00:00<00:06, 67.74it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 11%|█         | 53/472 [00:00<00:06, 68.79it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 13%|█▎        | 61/472 [00:00<00:05, 70.21it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 15%|█▌        | 71/472 [00:01<00:05, 78.09it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 17%|█▋        | 79/472 [00:01<00:06, 63.30it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 18%|█▊        | 86/472 [00:01<00:05, 64.64it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 20%|█▉        | 94/472 [00:01<00:05, 68.14it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 23%|██▎       | 107/472 [00:01<00:04, 81.95it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 25%|██▍       | 116/472 [00:01<00:04, 81.30it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 27%|██▋       | 127/472 [00:01<00:03, 87.54it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 29%|██▉       | 136/472 [00:01<00:03, 86.19it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 31%|███       | 145/472 [00:02<00:05, 63.11it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 33%|███▎      | 155/472 [00:02<00:04, 67.14it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 35%|███▍      | 164/472 [00:02<00:04, 70.98it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 37%|███▋      | 173/472 [00:02<00:04, 72.89it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 38%|███▊      | 181/472 [00:02<00:04, 71.63it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 40%|████      | 189/472 [00:02<00:04, 68.71it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 42%|████▏     | 197/472 [00:02<00:04, 67.32it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 44%|████▍     | 207/472 [00:03<00:04, 56.71it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 46%|████▌     | 215/472 [00:03<00:04, 60.21it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 47%|████▋     | 222/472 [00:03<00:04, 61.67it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 49%|████▊     | 229/472 [00:03<00:03, 62.26it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 50%|█████     | 237/472 [00:03<00:03, 64.66it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 52%|█████▏    | 244/472 [00:03<00:03, 63.81it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 53%|█████▎    | 251/472 [00:03<00:03, 63.03it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 55%|█████▍    | 258/472 [00:03<00:03, 61.00it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 56%|█████▌    | 265/472 [00:03<00:03, 61.18it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 58%|█████▊    | 272/472 [00:04<00:03, 61.15it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 59%|█████▉    | 279/472 [00:04<00:04, 46.16it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 61%|██████    | 286/472 [00:04<00:03, 51.20it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 63%|██████▎   | 298/472 [00:04<00:02, 66.97it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 65%|██████▍   | 306/472 [00:04<00:02, 64.76it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 67%|██████▋   | 314/472 [00:04<00:02, 65.10it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 68%|██████▊   | 321/472 [00:04<00:02, 64.89it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 69%|██████▉   | 328/472 [00:05<00:02, 63.69it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 71%|███████   | 335/472 [00:05<00:02, 62.20it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 72%|███████▏  | 342/472 [00:05<00:02, 60.84it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 75%|███████▍  | 353/472 [00:05<00:01, 73.39it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 77%|███████▋  | 364/472 [00:05<00:01, 82.87it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 79%|███████▉  | 374/472 [00:05<00:01, 84.91it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 82%|████████▏ | 385/472 [00:05<00:00, 91.21it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 84%|████████▎ | 395/472 [00:05<00:01, 65.12it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 86%|████████▌ | 406/472 [00:06<00:00, 73.75it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 88%|████████▊ | 415/472 [00:06<00:00, 76.77it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 90%|█████████ | 425/472 [00:06<00:00, 81.14it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 92%|█████████▏| 434/472 [00:06<00:00, 83.00it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 94%|█████████▍| 443/472 [00:06<00:00, 80.65it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 96%|█████████▌| 454/472 [00:06<00:00, 86.94it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 98%|█████████▊| 463/472 [00:06<00:00, 86.71it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

100%|██████████| 472/472 [00:06<00:00, 69.91it/s]
use COPT to build and solve model
removed initial skeleton
Setting parameter 'Logging' to 1
[facinv] 2023-12-14 19:36:35,185: initialization of restricted master finished
[facinv] 2023-12-14 19:36:35,186: solving the first rmp
Setting parameter 'LpMethod' to 2
Setting parameter 'Crossover' to 0
Model fingerprint: 2fd76794

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing an LP problem

The original problem has:
    58473 rows, 105499 columns and 197918 non-zero elements
The folded problem has:
    10536 rows, 44478 columns and 82400 non-zero elements
The presolved problem has:
    2563 rows, 24292 columns and 46021 non-zero elements

Starting barrier solver using 8 threads

Problem info:
Dualized in presolve:            No
Range of matrix coefficients:    [1e+00,2e+00]
Range of rhs coefficients:       [1e+00,4e+05]
Range of bound coefficients:     [0e+00,0e+00]
Range of cost coefficients:      [3e-02,2e+02]

Factor info:
Number of free columns:          424
Number of dense columns:         0
Number of matrix entries:        2.186e+04
Number of factor entries:        2.413e+04
Number of factor flops:          2.782e+05

Iter       Primal.Obj         Dual.Obj      Compl  Primal.Inf  Dual.Inf    Time
   0  +9.98513557e+09  +3.44252468e+07   1.47e+11    5.70e+06  2.00e+01   0.08s
   1  +5.85972542e+09  +1.03003191e+07   4.04e+10    9.22e+05  7.98e+00   0.08s
   2  +4.88415390e+09  +2.67256453e+07   5.86e+09    3.34e+04  3.27e-01   0.08s
   3  +9.82600666e+08  +2.96496779e+07   9.83e+08    5.82e+03  2.93e-02   0.09s
   4  +3.76220701e+08  +3.53942198e+07   3.46e+08    2.09e+03  4.33e-03   0.09s
   5  +3.38061484e+08  +3.84833377e+07   3.04e+08    1.85e+03  2.95e-03   0.09s
   6  +2.87898386e+08  +3.95540752e+07   2.52e+08    1.54e+03  2.37e-03   0.09s
   7  +2.54171035e+08  +4.06391386e+07   2.16e+08    1.33e+03  2.28e-03   0.09s
   8  +1.79094039e+08  +4.16903455e+07   1.39e+08    8.54e+02  1.57e-03   0.09s
   9  +1.27234447e+08  +4.27491038e+07   8.55e+07    5.24e+02  1.10e-03   0.10s
  10  +9.29995612e+07  +4.37388511e+07   4.98e+07    3.03e+02  8.08e-04   0.10s
  11  +7.13221328e+07  +4.50829404e+07   2.65e+07    1.60e+02  5.06e-04   0.10s
  12  +6.57193396e+07  +4.58103662e+07   2.01e+07    1.22e+02  3.85e-04   0.10s
  13  +6.21596580e+07  +4.65023646e+07   1.58e+07    9.77e+01  2.59e-04   0.10s
  14  +5.96113848e+07  +4.67819606e+07   1.30e+07    8.06e+01  1.99e-04   0.11s
  15  +5.79616298e+07  +4.70573165e+07   1.10e+07    6.95e+01  1.54e-04   0.11s
  16  +5.63992736e+07  +4.72230015e+07   9.27e+06    5.90e+01  1.26e-04   0.11s
  17  +5.51561307e+07  +4.73410473e+07   7.90e+06    5.07e+01  1.02e-04   0.11s
  18  +5.41560066e+07  +4.74194443e+07   6.80e+06    4.40e+01  8.13e-05   0.12s
  19  +5.36247989e+07  +4.75006342e+07   6.19e+06    4.05e+01  6.37e-05   0.12s
  20  +5.25181090e+07  +4.75357826e+07   5.03e+06    3.30e+01  5.42e-05   0.12s
  21  +5.16663351e+07  +4.75800462e+07   4.13e+06    2.72e+01  4.22e-05   0.13s
  22  +5.05981777e+07  +4.76168070e+07   3.01e+06    1.99e+01  3.30e-05   0.13s
  23  +4.99943007e+07  +4.76503622e+07   2.37e+06    1.57e+01  2.46e-05   0.13s
  24  +4.93950737e+07  +4.76733470e+07   1.74e+06    1.16e+01  1.90e-05   0.14s
  25  +4.91183536e+07  +4.76964129e+07   1.44e+06    9.64e+00  1.39e-05   0.14s
  26  +4.87002302e+07  +4.77097268e+07   1.00e+06    6.71e+00  1.11e-05   0.14s
  27  +4.85838282e+07  +4.77249094e+07   8.67e+05    5.89e+00  7.71e-06   0.14s
  28  +4.84832254e+07  +4.77351867e+07   7.55e+05    5.18e+00  5.38e-06   0.15s
  29  +4.81947259e+07  +4.77414025e+07   4.58e+05    3.13e+00  4.07e-06   0.15s
  30  +4.81576073e+07  +4.77450926e+07   4.16e+05    2.87e+00  3.28e-06   0.15s
  31  +4.80199468e+07  +4.77488389e+07   2.74e+05    1.89e+00  2.38e-06   0.16s
  32  +4.79432528e+07  +4.77518420e+07   1.93e+05    1.33e+00  1.90e-06   0.16s
  33  +4.78869085e+07  +4.77543768e+07   1.34e+05    9.17e-01  1.54e-06   0.16s
  34  +4.78424382e+07  +4.77564365e+07   8.69e+04    5.92e-01  1.17e-06   0.16s
  35  +4.78332133e+07  +4.77578163e+07   7.61e+04    5.24e-01  9.03e-07   0.16s
  36  +4.78132816e+07  +4.77590802e+07   5.47e+04    3.79e-01  6.83e-07   0.17s
  37  +4.77965938e+07  +4.77600435e+07   3.69e+04    2.56e-01  4.90e-07   0.17s
  38  +4.77827208e+07  +4.77605788e+07   2.24e+04    1.53e-01  3.89e-07   0.17s
  39  +4.77726649e+07  +4.77609931e+07   1.18e+04    7.81e-02  2.95e-07   0.17s
  40  +4.77684431e+07  +4.77614336e+07   7.09e+03    4.67e-02  1.89e-07   0.17s
  41  +4.77631759e+07  +4.77617119e+07   1.50e+03    1.00e-02  1.14e-07   0.18s
  42  +4.77627614e+07  +4.77618108e+07   9.71e+02    6.52e-03  7.11e-08   0.18s
  43  +4.77622354e+07  +4.77619251e+07   3.17e+02    2.54e-03  2.38e-08   0.18s
  44  +4.77622098e+07  +4.77619281e+07   2.83e+02    2.34e-03  4.73e-10   0.18s
  45  +4.77619826e+07  +4.77619749e+07   7.76e+00    1.05e-05  1.63e-09   0.19s
  46  +4.77619806e+07  +4.77619796e+07   1.02e+00    8.12e-06  1.75e-09   0.19s
  47  +4.77619796e+07  +4.77619796e+07   2.59e-06    1.70e-09  1.07e-10   0.19s

Barrier status:                  OPTIMAL
Primal objective:                4.77619796e+07
Dual objective:                  4.77619796e+07
Duality gap (abs/rel):           1.38e-06 / 2.89e-14
Primal infeasibility (abs/rel):  1.70e-09 / 3.87e-15
Dual infeasibility (abs/rel):    1.07e-10 / 5.47e-13
Postsolving
Unfolding solutions

Solving finished
Status: Optimal  Objective: 4.7761979596e+07  Iterations: 0  Time: 0.19s
1

0it [00:00, ?it/s]
24it [00:00, 233.64it/s]
50it [00:00, 248.34it/s]
80it [00:00, 270.78it/s]
108it [00:00, 184.43it/s]
138it [00:00, 213.82it/s]
163it [00:00, 214.68it/s]
187it [00:00, 205.47it/s]
212it [00:00, 215.59it/s]
235it [00:01, 198.64it/s]
256it [00:01, 195.72it/s]
277it [00:01, 184.60it/s]
299it [00:01, 193.53it/s]
321it [00:01, 199.15it/s]
342it [00:01, 151.13it/s]
361it [00:01, 157.90it/s]
389it [00:01, 185.37it/s]
413it [00:02, 199.04it/s]
439it [00:02, 213.39it/s]
466it [00:02, 228.88it/s]
                         
[facinv] 2023-12-14 19:36:37,997: k:     1 / 50 f: 4.776198e+07, c': 6.5125e-03
1

0it [00:00, ?it/s]
1it [00:01,  1.22s/it]
2it [00:03,  1.62s/it]
3it [00:05,  1.93s/it]
4it [00:08,  2.37s/it]
5it [00:09,  2.03s/it]
6it [00:12,  2.09s/it]
7it [00:13,  1.81s/it]
8it [00:15,  1.98s/it]
9it [00:18,  2.12s/it]
10it [00:19,  1.92s/it]
11it [00:21,  2.01s/it]
12it [00:23,  1.80s/it]
13it [00:25,  1.99s/it]
14it [00:26,  1.76s/it]
15it [00:29,  1.91s/it]
16it [00:30,  1.71s/it]
17it [00:32,  2.01s/it]
18it [00:34,  1.79s/it]
19it [00:36,  1.94s/it]
20it [00:38,  2.03s/it]
21it [00:40,  1.97s/it]
22it [00:43,  2.13s/it]
23it [00:44,  1.90s/it]
24it [00:46,  1.90s/it]
25it [00:48,  2.07s/it]
26it [00:51,  2.11s/it]
27it [00:53,  2.30s/it]
28it [00:55,  2.04s/it]
29it [00:56,  1.79s/it]
30it [00:58,  2.01s/it]
31it [01:01,  2.09s/it]
32it [01:02,  1.82s/it]
33it [01:04,  1.92s/it]
34it [01:07,  2.14s/it]
35it [01:09,  2.10s/it]
36it [01:10,  1.86s/it]
37it [01:12,  1.95s/it]
38it [01:13,  1.75s/it]
39it [01:16,  1.85s/it]
40it [01:18,  1.99s/it]
41it [01:20,  1.99s/it]
42it [01:21,  1.76s/it]
43it [01:23,  1.80s/it]
44it [01:25,  1.86s/it]
45it [01:27,  1.98s/it]
46it [01:29,  1.91s/it]
47it [01:32,  2.16s/it]
48it [01:34,  2.11s/it]
49it [01:36,  2.10s/it]
50it [01:38,  2.06s/it]
51it [01:40,  2.22s/it]
52it [01:43,  2.26s/it]
53it [01:44,  1.98s/it]
54it [01:46,  2.00s/it]
55it [01:48,  1.97s/it]
56it [01:50,  1.93s/it]
57it [01:52,  2.01s/it]
58it [01:54,  2.15s/it]
59it [01:57,  2.16s/it]
60it [01:58,  1.91s/it]
61it [02:00,  1.94s/it]
62it [02:01,  1.73s/it]
63it [02:03,  1.86s/it]/Users/xue/miniforge3/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

[Done] exited with code=null in 142.767 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:38:48,114: -- The FACINV python package --
[facinv] 2023-12-14 19:38:48,114:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:38:48,114: :solution      to ./out
[facinv] 2023-12-14 19:38:48,114: :data          to ./data
[facinv] 2023-12-14 19:38:48,114: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:38:48,431: generating the signature of this problem
[facinv] 2023-12-14 19:38:48,431: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:38:48,431: current data has been generated before
[facinv] 2023-12-14 19:38:48,431: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 53, in <module>
    ) = utils.get_data_from_cfg(cfg)
  File "/Users/xue/github/facility-loc-inventory/utils.py", line 206, in get_data_from_cfg
    package = pickle.load(open(fp, "rb"))
ModuleNotFoundError: No module named 'entity'

[Done] exited with code=1 in 1.048 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:39:01,983: -- The FACINV python package --
[facinv] 2023-12-14 19:39:01,983:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:39:01,983: :solution      to ./out
[facinv] 2023-12-14 19:39:01,983: :data          to ./data
[facinv] 2023-12-14 19:39:01,983: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:39:02,241: generating the signature of this problem
[facinv] 2023-12-14 19:39:02,241: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:39:02,241: current data has been generated before
[facinv] 2023-12-14 19:39:02,241: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:39:02,266: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:39:02,279: initialization complete, start generating columns...
[facinv] 2023-12-14 19:39:02,279: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 67%|██████▋   | 10/15 [00:00<00:00, 99.55it/s]
100%|██████████| 15/15 [00:00<00:00, 97.90it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/ncg/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 0.961 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py"
[facinv] 2023-12-14 19:39:06,658: -- The FACINV python package --
[facinv] 2023-12-14 19:39:06,658:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:39:06,658: :solution      to ./out
[facinv] 2023-12-14 19:39:06,658: :data          to ./data
[facinv] 2023-12-14 19:39:06,658: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:39:06,710: generating the signature of this problem
[facinv] 2023-12-14 19:39:06,710: {
  "data_dir": "data/data_0401_V4.xlsx",
  "one_period": true
}
[facinv] 2023-12-14 19:39:06,710: current data has been generated before
[facinv] 2023-12-14 19:39:06,710: reading from cache: ./out/data_0401_V4-True.pk
----------DCG Model------------
[facinv] 2023-12-14 19:39:07,303: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
[facinv] 2023-12-14 19:39:08,067: initialization complete, start generating columns...
[facinv] 2023-12-14 19:39:08,067: generating column oracles

  0%|          | 0/472 [00:00<?, ?it/s]use COPT to build and solve model
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved


  0%|          | 0/472 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py", line 65, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_cg.py", line 332, in run
    self.oracles[customer] = self.construct_oracle(customer)
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_cg.py", line 195, in construct_oracle
    oracle.modeling(customer)
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 250, in modeling
    self.modeling_milp(model_name, gap, limit, threads, logging,customer)
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 280, in modeling_milp
    self.add_vars()
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 372, in add_vars
    for node in self._iterate_nodes():
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 298, in _iterate_nodes
    if n.type != config.const.CUSTOMER:
NameError: name 'config' is not defined

[Done] exited with code=1 in 2.358 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py"
[facinv] 2023-12-14 19:39:24,574: -- The FACINV python package --
[facinv] 2023-12-14 19:39:24,574:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:39:24,574: :solution      to ./out
[facinv] 2023-12-14 19:39:24,575: :data          to ./data
[facinv] 2023-12-14 19:39:24,575: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:39:24,626: generating the signature of this problem
[facinv] 2023-12-14 19:39:24,626: {
  "data_dir": "data/data_0401_V4.xlsx",
  "one_period": true
}
[facinv] 2023-12-14 19:39:24,626: current data has been generated before
[facinv] 2023-12-14 19:39:24,626: reading from cache: ./out/data_0401_V4-True.pk
----------DCG Model------------
[facinv] 2023-12-14 19:39:25,226: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
[facinv] 2023-12-14 19:39:25,975: initialization complete, start generating columns...
[facinv] 2023-12-14 19:39:25,975: generating column oracles

  0%|          | 0/472 [00:00<?, ?it/s]use COPT to build and solve model
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved


  0%|          | 0/472 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py", line 65, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_cg.py", line 332, in run
    self.oracles[customer] = self.construct_oracle(customer)
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_cg.py", line 195, in construct_oracle
    oracle.modeling(customer)
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 250, in modeling
    self.modeling_milp(model_name, gap, limit, threads, logging,customer)
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 280, in modeling_milp
    self.add_vars()
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 372, in add_vars
    for node in self._iterate_nodes():
  File "/Users/xue/github/facility-loc-inventory/slim_cg/slim_pricing.py", line 298, in _iterate_nodes
    if n.type != config.const.CUSTOMER:
NameError: name 'config' is not defined

[Done] exited with code=1 in 2.755 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py"
[facinv] 2023-12-14 19:40:25,330: -- The FACINV python package --
[facinv] 2023-12-14 19:40:25,330:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:40:25,330: :solution      to ./out
[facinv] 2023-12-14 19:40:25,331: :data          to ./data
[facinv] 2023-12-14 19:40:25,331: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:40:25,373: generating the signature of this problem
[facinv] 2023-12-14 19:40:25,373: {
  "data_dir": "data/data_0401_V4.xlsx",
  "one_period": true
}
[facinv] 2023-12-14 19:40:25,373: current data has been generated before
[facinv] 2023-12-14 19:40:25,373: reading from cache: ./out/data_0401_V4-True.pk
----------DCG Model------------
[facinv] 2023-12-14 19:40:25,970: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
[facinv] 2023-12-14 19:40:26,719: initialization complete, start generating columns...
[facinv] 2023-12-14 19:40:26,719: generating column oracles

  0%|          | 0/472 [00:00<?, ?it/s]use COPT to build and solve model
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  1%|▏         | 6/472 [00:00<00:09, 50.92it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  3%|▎         | 14/472 [00:00<00:06, 66.42it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  5%|▍         | 22/472 [00:00<00:06, 67.63it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  6%|▌         | 29/472 [00:00<00:07, 55.89it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  7%|▋         | 35/472 [00:00<00:07, 56.93it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 10%|▉         | 45/472 [00:00<00:06, 68.11it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 11%|█         | 53/472 [00:00<00:06, 69.83it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 13%|█▎        | 61/472 [00:00<00:05, 71.61it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 15%|█▌        | 72/472 [00:01<00:04, 82.49it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 17%|█▋        | 81/472 [00:01<00:06, 63.47it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 19%|█▉        | 89/472 [00:01<00:05, 67.18it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 21%|██        | 100/472 [00:01<00:04, 77.36it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 23%|██▎       | 110/472 [00:01<00:04, 82.91it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 25%|██▌       | 119/472 [00:01<00:04, 82.26it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 28%|██▊       | 131/472 [00:01<00:03, 90.27it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 30%|██▉       | 141/472 [00:02<00:04, 66.95it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 32%|███▏      | 151/472 [00:02<00:04, 70.92it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 34%|███▍      | 160/472 [00:02<00:04, 74.96it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 36%|███▌      | 169/472 [00:02<00:03, 76.75it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 38%|███▊      | 178/472 [00:02<00:03, 76.55it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 39%|███▉      | 186/472 [00:02<00:03, 73.13it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 41%|████      | 194/472 [00:02<00:04, 69.39it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 43%|████▎     | 205/472 [00:02<00:03, 76.86it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 45%|████▌     | 213/472 [00:03<00:04, 53.05it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 47%|████▋     | 220/472 [00:03<00:04, 55.52it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 48%|████▊     | 227/472 [00:03<00:04, 58.12it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 50%|████▉     | 235/472 [00:03<00:03, 61.60it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 51%|█████▏    | 242/472 [00:03<00:03, 63.68it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 53%|█████▎    | 249/472 [00:03<00:03, 62.38it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 54%|█████▍    | 256/472 [00:03<00:03, 60.37it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 56%|█████▌    | 263/472 [00:03<00:03, 60.82it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 57%|█████▋    | 270/472 [00:04<00:03, 60.11it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 59%|█████▊    | 277/472 [00:04<00:03, 62.29it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 60%|██████    | 284/472 [00:04<00:03, 48.02it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 62%|██████▏   | 292/472 [00:04<00:03, 55.20it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 64%|██████▍   | 303/472 [00:04<00:02, 67.17it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 66%|██████▌   | 311/472 [00:04<00:02, 65.42it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 67%|██████▋   | 318/472 [00:04<00:02, 64.13it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 69%|██████▉   | 325/472 [00:04<00:02, 63.69it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 70%|███████   | 332/472 [00:05<00:02, 61.54it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 72%|███████▏  | 339/472 [00:05<00:02, 60.48it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 73%|███████▎  | 346/472 [00:05<00:02, 62.43it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 76%|███████▌  | 359/472 [00:05<00:01, 77.79it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 78%|███████▊  | 369/472 [00:05<00:01, 81.50it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 80%|████████  | 379/472 [00:05<00:01, 84.25it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 82%|████████▏ | 388/472 [00:05<00:01, 62.63it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 84%|████████▍ | 396/472 [00:05<00:01, 67.13it/s]

[Done] exited with code=15 in 8.775 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:40:34,357: -- The FACINV python package --
[facinv] 2023-12-14 19:40:34,358:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:40:34,358: :solution      to ./out
[facinv] 2023-12-14 19:40:34,358: :data          to ./data
[facinv] 2023-12-14 19:40:34,358: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:40:34,671: generating the signature of this problem
[facinv] 2023-12-14 19:40:34,671: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:40:34,671: current data has been generated before
[facinv] 2023-12-14 19:40:34,671: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:40:34,695: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:40:34,707: initialization complete, start generating columns...
[facinv] 2023-12-14 19:40:34,707: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 67%|██████▋   | 10/15 [00:00<00:00, 96.68it/s]
100%|██████████| 15/15 [00:00<00:00, 96.94it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/ncg/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 1.202 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:40:55,776: -- The FACINV python package --
[facinv] 2023-12-14 19:40:55,776:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:40:55,776: :solution      to ./out
[facinv] 2023-12-14 19:40:55,776: :data          to ./data
[facinv] 2023-12-14 19:40:55,776: :logs and tmps to ./tmp
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 53, in <module>
    ) = config.utils.get_data_from_cfg(cfg)
NameError: name 'config' is not defined

[Done] exited with code=1 in 0.712 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:41:08,755: -- The FACINV python package --
[facinv] 2023-12-14 19:41:08,755:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:41:08,755: :solution      to ./out
[facinv] 2023-12-14 19:41:08,755: :data          to ./data
[facinv] 2023-12-14 19:41:08,755: :logs and tmps to ./tmp
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 53, in <module>
    ) = config.utils.get_data_from_cfg(cfg)
NameError: name 'config' is not defined

[Done] exited with code=1 in 0.704 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/tempCodeRunnerFile.py"
  File "/Users/xue/github/facility-loc-inventory/tempCodeRunnerFile.py", line 1
    config.
           ^
SyntaxError: invalid syntax

[Done] exited with code=1 in 0.033 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncg_np.py"
[facinv] 2023-12-14 19:41:34,889: -- The FACINV python package --
[facinv] 2023-12-14 19:41:34,889:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:41:34,889: :solution      to ./out
[facinv] 2023-12-14 19:41:34,889: :data          to ./data
[facinv] 2023-12-14 19:41:34,889: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:41:35,170: generating the signature of this problem
[facinv] 2023-12-14 19:41:35,170: {
  "customer_num": 15,
  "data_dir": "data/data_0401_0inv.xlsx",
  "plant_num": 1,
  "sku_num": 1,
  "warehouse_num": 25
}
[facinv] 2023-12-14 19:41:35,170: current data has been generated before
[facinv] 2023-12-14 19:41:35,170: reading from cache: ./out/data_0401_0inv-15-1-1-25.pk
[facinv] 2023-12-14 19:41:35,195: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-12-14 19:41:35,208: initialization complete, start generating columns...
[facinv] 2023-12-14 19:41:35,208: generating column oracles

  0%|          | 0/15 [00:00<?, ?it/s]
 67%|██████▋   | 10/15 [00:00<00:00, 96.16it/s]
100%|██████████| 15/15 [00:00<00:00, 96.68it/s]
Traceback (most recent call last):
  File "/Users/xue/github/facility-loc-inventory/main_ncg_np.py", line 104, in <module>
    np_cg.run()
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 342, in run
    cg_init.primal_sweeping_method(self)
  File "/Users/xue/github/facility-loc-inventory/ncg/cg_init.py", line 92, in primal_sweeping_method
    self.subproblem(_this_customer, col_ind)
  File "/Users/xue/github/facility-loc-inventory/ncg/np_cg.py", line 290, in subproblem
    new_col = oracle.query_columns()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 353, in query_columns
    new_col = self.eval_helper()
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 322, in eval_helper
    _vals = {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 323, in <dictcomp>
    attr: {
  File "/Users/xue/github/facility-loc-inventory/dnp_model.py", line 326, in <dictcomp>
    for k, v in self.columns_helpers[attr][t].items()
TypeError: 'NoneType' object is not subscriptable

[Done] exited with code=1 in 0.951 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_dnp.py"
[facinv] 2023-12-14 19:41:40,816: -- The FACINV python package --
[facinv] 2023-12-14 19:41:40,816:   LLLGZ, 2023 (c)   
[facinv] 2023-12-14 19:41:40,816: :solution      to ./out
[facinv] 2023-12-14 19:41:40,816: :data          to ./data
[facinv] 2023-12-14 19:41:40,816: :logs and tmps to ./tmp
[facinv] 2023-12-14 19:41:40,868: generating the signature of this problem
[facinv] 2023-12-14 19:41:40,868: {
  "data_dir": "data/data_0401_V4.xlsx",
  "one_period": true
}
[facinv] 2023-12-14 19:41:40,868: current data has been generated before
[facinv] 2023-12-14 19:41:40,868: reading from cache: ./out/data_0401_V4-True.pk
----------DCG Model------------
[facinv] 2023-12-14 19:41:41,514: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
[facinv] 2023-12-14 19:41:42,312: initialization complete, start generating columns...
[facinv] 2023-12-14 19:41:42,312: generating column oracles

  0%|          | 0/472 [00:00<?, ?it/s]use COPT to build and solve model
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  1%|▏         | 6/472 [00:00<00:08, 53.58it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  3%|▎         | 15/472 [00:00<00:06, 66.58it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  5%|▍         | 23/472 [00:00<00:06, 72.04it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  7%|▋         | 31/472 [00:00<00:08, 53.17it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

  8%|▊         | 40/472 [00:00<00:06, 62.16it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 10%|█         | 48/472 [00:00<00:06, 66.52it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 12%|█▏        | 56/472 [00:00<00:06, 65.76it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 14%|█▎        | 64/472 [00:00<00:05, 69.16it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 15%|█▌        | 73/472 [00:01<00:06, 59.08it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 17%|█▋        | 82/472 [00:01<00:06, 64.46it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 19%|█▉        | 90/472 [00:01<00:05, 65.96it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 22%|██▏       | 105/472 [00:01<00:04, 82.86it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 24%|██▍       | 114/472 [00:01<00:04, 77.92it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 26%|██▌       | 123/472 [00:01<00:04, 80.46it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 28%|██▊       | 134/472 [00:01<00:03, 84.84it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 30%|███       | 143/472 [00:02<00:05, 63.03it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 32%|███▏      | 151/472 [00:02<00:04, 66.55it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 34%|███▍      | 160/472 [00:02<00:04, 71.57it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 36%|███▌      | 169/472 [00:02<00:04, 74.25it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 38%|███▊      | 177/472 [00:02<00:04, 72.91it/s]use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model
use COPT to build and solve model

 38%|███▊      | 180/472 [00:02<00:04, 69.47it/s]

[Done] exited with code=15 in 5.361 seconds

[Running] bash "/Users/xue/github/facility-loc-inventory/test.sh"

[Done] exited with code=1 in 26.257 seconds

[Running] bash "/Users/xue/github/facility-loc-inventory/test.sh"

[Done] exited with code=null in 1033.922 seconds

[Running] bash "/Users/xue/github/facility-loc-inventory/test.sh"

[Done] exited with code=0 in 22850.215 seconds

[Running] bash "/Users/xue/github/facility-loc-inventory/test.sh"

[Done] exited with code=null in 64.877 seconds

[Running] bash "/Users/xue/github/facility-loc-inventory/test.sh"

[Done] exited with code=null in 3547.052 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_np.py"
[facinv] 2023-12-15 11:07:21,171: -- The FACINV python package --
[facinv] 2023-12-15 11:07:21,171:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:07:21,171: :solution      to ./out
[facinv] 2023-12-15 11:07:21,171: :data          to ./data
[facinv] 2023-12-15 11:07:21,171: :logs and tmps to ./tmp
[facinv] 2023-12-15 11:07:21,218: generating the signature of this problem
[facinv] 2023-12-15 11:07:21,219: {
  "data_dir": "data/data_0401_0inv_V2.xlsx",
  "one_period": true
}
[facinv] 2023-12-15 11:07:21,219: current data has not been generated before
[facinv] 2023-12-15 11:07:21,219: creating a temporary cache @./out/data_0401_0inv_V2-True.pk
[facinv] 2023-12-15 11:08:22,718: dumping a temporary cache @./out/data_0401_0inv_V2-True.pk
----------DCG Model------------
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'Threads' to 8
Setting parameter 'TimeLimit' to 3600
Model fingerprint: 2b6331d3

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    287205 rows, 491872 columns and 1457070 non-zero elements
    235768 binaries

Presolving the problem

The presolved problem has:
    24815 rows, 231478 columns and 633914 non-zero elements
    2330 binaries

Starting the MIP solver with 8 threads and 32 tasks

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --       0  3.467077e+06            --    Inf  4.92s
H        0         1      --       0  3.467077e+06  3.700525e+06  6.31%  4.92s
         0         1      --       0  3.700525e+06  3.700525e+06  0.00% 20.00s
         1         0     0.0       0  3.700525e+06  3.700525e+06  0.00% 20.00s
         1         0     0.0       0  3.700525e+06  3.700525e+06  0.00% 20.00s

Best solution   : 3700524.596295394
Best bound      : 3700524.596295394
Best gap        : 0.0000%
Solve time      : 20.01
Solve node      : 1
MIP status      : solved
Solution status : integer optimal (relative gap limit 0.0001)

Violations      :     absolute     relative
  bounds        :            0            0
  rows          :            0            0
  integrality   :            0

[Done] exited with code=0 in 111.622 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_np.py"
[facinv] 2023-12-15 11:09:36,049: -- The FACINV python package --
[facinv] 2023-12-15 11:09:36,049:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:09:36,049: :solution      to ./out
[facinv] 2023-12-15 11:09:36,049: :data          to ./data
[facinv] 2023-12-15 11:09:36,049: :logs and tmps to ./tmp
[facinv] 2023-12-15 11:09:36,084: generating the signature of this problem
[facinv] 2023-12-15 11:09:36,084: {
  "data_dir": "data/data_0401_0inv_V2.xlsx",
  "one_period": true
}
[facinv] 2023-12-15 11:09:36,084: current data has been generated before
[facinv] 2023-12-15 11:09:36,085: reading from cache: ./out/data_0401_0inv_V2-True.pk
----------DCG Model------------
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'Threads' to 8
Setting parameter 'TimeLimit' to 3600
Model fingerprint: 2b6331d3

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    287205 rows, 491872 columns and 1457070 non-zero elements
    235768 binaries

Presolving the problem

The presolved problem has:
    24815 rows, 231478 columns and 633914 non-zero elements
    2330 binaries

Starting the MIP solver with 8 threads and 32 tasks

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --       0  3.467077e+06            --    Inf  4.76s
H        0         1      --       0  3.467077e+06  3.700525e+06  6.31%  4.77s
         0         1      --       0  3.700525e+06  3.700525e+06  0.00% 19.30s
         1         0     0.0       0  3.700525e+06  3.700525e+06  0.00% 19.30s
         1         0     0.0       0  3.700525e+06  3.700525e+06  0.00% 19.30s

Best solution   : 3700524.596295394
Best bound      : 3700524.596295394
Best gap        : 0.0000%
Solve time      : 19.31
Solve node      : 1
MIP status      : solved
Solution status : integer optimal (relative gap limit 0.0001)

Violations      :     absolute     relative
  bounds        :            0            0
  rows          :            0            0
  integrality   :            0

[Done] exited with code=0 in 49.133 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_np.py"
[facinv] 2023-12-15 11:13:19,055: -- The FACINV python package --
[facinv] 2023-12-15 11:13:19,056:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:13:19,056: :solution      to ./out
[facinv] 2023-12-15 11:13:19,056: :data          to ./data
[facinv] 2023-12-15 11:13:19,056: :logs and tmps to ./tmp
[facinv] 2023-12-15 11:13:19,102: generating the signature of this problem
[facinv] 2023-12-15 11:13:19,102: {
  "customer_num": 519,
  "data_dir": "data/data_0401_0inv_V2.xlsx",
  "one_period": false,
  "plant_num": 20,
  "sku_num": 140,
  "warehouse_num": 20
}
[facinv] 2023-12-15 11:13:19,102: current data has been generated before
[facinv] 2023-12-15 11:13:19,102: reading from cache: ./out/data_0401_0inv_V2-519-False-20-140-20.pk
setting T0001_T0014 to 144.2181796
setting T0001_T0019 to 98.31385537
setting T0015_T0014 to 394.0329401
setting T0015_T0021 to 408.0426661
setting T0015_T0030 to 752.1390094000001
setting T0030_T0003 to 351.76162220000003
setting T0030_T0014 to 58.435917159999995
setting T0030_T0019 to 41.699958779999996
setting T0030_T0021 to 91.30670755999999
setting T0030_T0022 to 51.846567900000004
setting T0030_T0026 to 83.73739974
setting T0030_T0028 to 38.51742725
setting T0030_T0029 to 187.7694078
setting T0030_T0031 to 141.5849477
setting T0030_T0041 to 75.34098564
----------DCG Model------------
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'Threads' to 8
Setting parameter 'TimeLimit' to 3600
Model fingerprint: deef2028

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    986192 rows, 1575018 columns and 4704546 non-zero elements
    748384 binaries

Presolving the problem

Best solution   : +inf
Best bound      : +inf
Best gap        : 0.0000%
Solve time      : 1.49
Solve node      : 0
MIP status      : solved
Solution status : infeasible

[Done] exited with code=0 in 73.13 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_np.py"
[facinv] 2023-12-15 11:19:37,263: -- The FACINV python package --
[facinv] 2023-12-15 11:19:37,263:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:19:37,263: :solution      to ./out
[facinv] 2023-12-15 11:19:37,263: :data          to ./data
[facinv] 2023-12-15 11:19:37,264: :logs and tmps to ./tmp
[facinv] 2023-12-15 11:19:37,303: generating the signature of this problem
[facinv] 2023-12-15 11:19:37,304: {
  "customer_num": 519,
  "data_dir": "data/data_0401_0inv_V2.xlsx",
  "one_period": false,
  "plant_num": 20,
  "sku_num": 140,
  "warehouse_num": 20
}
[facinv] 2023-12-15 11:19:37,304: current data has been generated before
[facinv] 2023-12-15 11:19:37,304: reading from cache: ./out/data_0401_0inv_V2-519-False-20-140-20.pk
setting T0001_T0014 to 144.2181796
setting T0001_T0019 to 98.31385537
setting T0015_T0014 to 394.0329401
setting T0015_T0021 to 408.0426661
setting T0015_T0030 to 752.1390094000001
setting T0030_T0003 to 351.76162220000003
setting T0030_T0014 to 58.435917159999995
setting T0030_T0019 to 41.699958779999996
setting T0030_T0021 to 91.30670755999999
setting T0030_T0022 to 51.846567900000004
setting T0030_T0026 to 83.73739974
setting T0030_T0028 to 38.51742725
setting T0030_T0029 to 187.7694078
setting T0030_T0031 to 141.5849477
setting T0030_T0041 to 75.34098564
----------DCG Model------------
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved


[Done] exited with code=15 in 3.346 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_np.py"
[facinv] 2023-12-15 11:20:12,310: -- The FACINV python package --
[facinv] 2023-12-15 11:20:12,311:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:20:12,311: :solution      to ./out
[facinv] 2023-12-15 11:20:12,311: :data          to ./data
[facinv] 2023-12-15 11:20:12,311: :logs and tmps to ./tmp
[facinv] 2023-12-15 11:20:12,351: generating the signature of this problem
[facinv] 2023-12-15 11:20:12,351: {
  "customer_num": 519,
  "data_dir": "data/data_0401_0inv_V2.xlsx",
  "one_period": false,
  "plant_num": 20,
  "sku_num": 140,
  "warehouse_num": 20
}
[facinv] 2023-12-15 11:20:12,351: current data has been generated before
[facinv] 2023-12-15 11:20:12,351: reading from cache: ./out/data_0401_0inv_V2-519-False-20-140-20.pk
setting T0001_T0014 to 144.2181796
setting T0001_T0019 to 98.31385537
setting T0015_T0014 to 394.0329401
setting T0015_T0021 to 408.0426661
setting T0015_T0030 to 752.1390094000001
setting T0030_T0003 to 351.76162220000003
setting T0030_T0014 to 58.435917159999995
setting T0030_T0019 to 41.699958779999996
setting T0030_T0021 to 91.30670755999999
setting T0030_T0022 to 51.846567900000004
setting T0030_T0026 to 83.73739974
setting T0030_T0028 to 38.51742725
setting T0030_T0029 to 187.7694078
setting T0030_T0031 to 141.5849477
setting T0030_T0041 to 75.34098564
----------DCG Model------------
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'Threads' to 8
Setting parameter 'TimeLimit' to 3600
Model fingerprint: deef2028

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    986192 rows, 1575018 columns and 4704546 non-zero elements
    748384 binaries

Presolving the problem

Best solution   : +inf
Best bound      : +inf
Best gap        : 0.0000%
Solve time      : 1.51
Solve node      : 0
MIP status      : solved
Solution status : infeasible
initial column of RMP is infeasible
Model fingerprint: deef2028

Start the IIS computation for a MIP

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
         0           0      986192           0     1575018      0.00s
Warning: model contains 1014 empty rows which are redundant
         2           0        5796           0       62306     19.65s
         3           0        5732           0       55850     19.68s
         4           0        5668           0       51489     19.72s
         5           0        5604           0       48863     19.75s
         6           0        5540           0       47264     19.78s
         7           0        5476           0       45874     19.81s
         8           0        5412           0       44715     19.84s
         9           0        5348           0       43883     19.85s
        10           0        5284           0       43012     19.86s
        11           0        5220           0       42093     19.88s
        12           0        5156           0       41250     19.89s
        13           0        5092           0       40414     19.90s
        14           0        5028           0       39578     19.91s
        15           0        4964           0       38719     19.92s
        16           0        4900           0       37849     19.93s
        17           0        4836           0       36966     19.94s
        18           0        4772           0       36205     19.95s
        19           0        4708           0       35442     19.96s
        20           0        4644           0       34679     19.97s

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
        21           0        4580           0       33909     19.98s
        22           0        4516           0       33132     19.99s
        23           0        4452           0       32337     20.00s
        24           0        4388           0       31539     20.01s
        25           0        4324           0       30755     20.02s
        26           0        4260           0       29957     20.04s
        27           0        4196           0       29156     20.05s
        28           0        4132           0       28353     20.06s
        29           0        4068           0       27564     20.07s
        30           0        4004           0       26850     20.08s
        31           0        3940           0       26113     20.09s
        32           0        3876           0       25380     20.10s
        33           0        3812           0       24638     20.11s
        34           0        3748           0       23904     20.12s
        35           0        3684           0       23171     20.13s
        36           0        3620           0       22442     20.14s
        37           0        3556           0       21711     20.15s
        38           0        3492           0       20974     20.16s
        39           0        3428           0       20240     20.17s
        40           0        3364           0       19493     20.18s

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
        41           0        3300           0       18745     20.19s
        42           0        3236           0       18006     20.20s
        43           0        3172           0       17251     20.21s
        44           0        3108           0       16489     20.22s
        45           0        3044           0       15804     20.23s
        46           0        2980           0       15136     20.24s
        47           0        2916           0       14473     20.25s
        48           0        2852           0       13814     20.26s
        49           0        2788           0       13130     20.27s
        50           0        2724           0       12445     20.28s
        51           0        2660           0       11762     20.29s
        52           0        2596           0       11076     20.30s
        53           0        2532           0       10388     20.31s
        54           0        2468           0        9747     20.32s
        55           0        2404           0        9129     20.33s
        56           0        2340           0        8511     20.34s
        57           0        2276           0        7877     20.35s
        58           0        2212           0        7242     20.36s
        59           0        2148           0        6704     20.37s
        60           0        2084           0        6146     20.37s

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
        61           0        2020           0        5628     20.38s
        62           0        1956           0        5304     20.39s
        63           0        1892           0        5151     20.40s
        64           0        1828           0        4985     20.41s
        65           0        1764           0        4810     20.42s
        66           0        1700           0        4620     20.43s
        67           0        1636           0        4432     20.44s
        68           0        1572           0        4246     20.45s
        69           0        1508           0        4055     20.46s
        70           0        1444           0        3848     20.47s
        71           0        1380           0        3643     20.48s
        72           0        1316           0        3438     20.49s
        73           0        1252           0        3221     20.50s
        74           0        1188           0        2999     20.51s
        75           0        1124           0        2773     20.52s
        76           0        1060           0        2538     20.53s
        77           0         996           0        2384     20.54s
        78           0         932           0        2243     20.55s
        79           0         868           0        2089     20.56s
        80           0         804           0        1929     20.57s

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
        81           0         740           0        1782     20.58s
        82           0         676           0        1633     20.59s
        83           0         612           0        1465     20.59s
        84           0         548           0        1308     20.60s
        97           1         485           0        1144     20.72s
        98           1         421           0         969     20.73s
        99           1         357           0         800     20.74s
       100           1         293           0         619     20.75s
       113           2         230           0         457     20.87s
       114           2         166           0         330     20.88s
       115           2         102           0         202     20.89s
       128           3          39           0          76     21.01s
       139           4           8           0          14     21.11s
       140           4           4           0           6     21.12s
       141           4           4           0           5     21.13s
       142           4           4           0           4     21.14s
       143           4           4           0           3     21.15s
       144           4           4           0           2     21.16s
       145           4           4           0           1     21.17s
       146           4           4           0           0     21.18s

IIS summary: 4 rows, 0 bounds of columns
IIS computation finished (21.208s)
Writing IIS problem to /Users/xue/github/facility-loc-inventory/iis/dnp.iis

[Done] exited with code=0 in 94.504 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/dnp_model.py"
[facinv] 2023-12-15 11:30:52,468: -- The FACINV python package --
[facinv] 2023-12-15 11:30:52,468:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:30:52,468: :solution      to ./out
[facinv] 2023-12-15 11:30:52,468: :data          to ./data
[facinv] 2023-12-15 11:30:52,468: :logs and tmps to ./tmp
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'RelGap' to 0.0001
Setting parameter 'TimeLimit' to 3600
Model fingerprint: 29a5b649

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    281433 rows, 494873 columns and 1171263 non-zero elements
    242689 binaries

Presolving the problem

The presolved problem has:
    19860 rows, 226534 columns and 604346 non-zero elements
    2489 binaries

Starting the MIP solver with 8 threads and 32 tasks

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --       0  4.668310e+06            --    Inf  6.23s
         0         1      --     342  2.021055e+07            --    Inf 19.19s
         0         1      --     317  2.021492e+07            --    Inf 20.38s
         0         1      --     268  2.022466e+07            --    Inf 22.48s
         0         1      --     243  2.022839e+07            --    Inf 23.95s
         0         1      --     235  2.023044e+07            --    Inf 25.23s
         0         1      --     227  2.023159e+07            --    Inf 26.34s
         0         1      --     256  2.023262e+07            --    Inf 27.30s
         0         1      --     257  2.023381e+07            --    Inf 28.53s
         0         1      --     230  2.023503e+07            --    Inf 29.37s
         0         1      --     213  2.023675e+07            --    Inf 33.27s
         0         1      --     156  2.023792e+07            --    Inf 36.34s
         0         1      --     149  2.023840e+07            --    Inf 38.50s
         0         1      --     136  2.023972e+07            --    Inf 41.18s
         0         1      --     117  2.024009e+07            --    Inf 43.21s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --     102  2.024033e+07            --    Inf 45.00s
         0         1      --      97  2.024067e+07            --    Inf 46.41s
         0         1      --      90  2.024122e+07            --    Inf 48.54s
         0         1      --      88  2.024183e+07            --    Inf 51.13s
         0         1      --      81  2.024201e+07            --    Inf 52.66s
         0         1      --      73  2.024214e+07            --    Inf 54.19s
         0         1      --      78  2.024234e+07            --    Inf 55.44s
H        0         0      --      78  2.024550e+07  3.366291e+07  39.9% 87.50s
H        0         0      --      78  2.024550e+07  3.366291e+07  39.9% 99.76s
         1         2   25069      78  2.024550e+07  3.366291e+07  39.9% 99.76s
         2         2   13294     127  2.024801e+07  3.366291e+07  39.9%   117s
         3         4    9741     166  2.024801e+07  3.366291e+07  39.9%   117s
         4         2    9311     228  2.025414e+07  3.366291e+07  39.8%   142s
         5         4    7917     131  2.025414e+07  3.366291e+07  39.8%   142s
         6         6    7562     187  2.025414e+07  3.366291e+07  39.8%   142s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         7         8    6868      85  2.025414e+07  3.366291e+07  39.8%   142s
         8         2    6169     239  2.025899e+07  3.366291e+07  39.8%   155s
         9         4    5924     176  2.025899e+07  3.366291e+07  39.8%   155s
        10         6    5450     194  2.025899e+07  3.366291e+07  39.8%   155s
        20        10    4033     219  2.026959e+07  3.366291e+07  39.8%   179s
        30        30    3899     181  2.026959e+07  3.366291e+07  39.8%   179s

[Done] exited with code=null in 278.128 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/main_ncs_np.py"
[facinv] 2023-12-15 11:35:30,893: -- The FACINV python package --
[facinv] 2023-12-15 11:35:30,894:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:35:30,894: :solution      to ./out
[facinv] 2023-12-15 11:35:30,894: :data          to ./data
[facinv] 2023-12-15 11:35:30,894: :logs and tmps to ./tmp
[facinv] 2023-12-15 11:35:30,944: generating the signature of this problem
[facinv] 2023-12-15 11:35:30,945: {
  "customer_num": 519,
  "data_dir": "data/data_0401_0inv_V2.xlsx",
  "one_period": false,
  "plant_num": 20,
  "sku_num": 140,
  "warehouse_num": 20
}
[facinv] 2023-12-15 11:35:30,945: current data has been generated before
[facinv] 2023-12-15 11:35:30,945: reading from cache: ./out/data_0401_0inv_V2-519-False-20-140-20.pk
setting T0001_T0014 to 144.2181796
setting T0001_T0019 to 98.31385537
setting T0015_T0014 to 394.0329401
setting T0015_T0021 to 408.0426661
setting T0015_T0030 to 752.1390094000001
setting T0030_T0003 to 351.76162220000003
setting T0030_T0014 to 58.435917159999995
setting T0030_T0019 to 41.699958779999996
setting T0030_T0021 to 91.30670755999999
setting T0030_T0022 to 51.846567900000004
setting T0030_T0026 to 83.73739974
setting T0030_T0028 to 38.51742725
setting T0030_T0029 to 187.7694078
setting T0030_T0031 to 141.5849477
setting T0030_T0041 to 75.34098564
----------DCG Model------------
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'Threads' to 8
Setting parameter 'TimeLimit' to 3600
Model fingerprint: 6c446e43

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    986192 rows, 1575018 columns and 4704546 non-zero elements
    748384 binaries

Presolving the problem

Best solution   : +inf
Best bound      : +inf
Best gap        : 0.0000%
Solve time      : 1.42
Solve node      : 0
MIP status      : solved
Solution status : infeasible
initial column of RMP is infeasible
Model fingerprint: 6c446e43

Start the IIS computation for a MIP

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
         0           0      986192           0     1575018      0.00s
Warning: model contains 1014 empty rows which are redundant
         4           0        1476           0       41286     17.69s
         5           0        1412           0       27863     17.71s
         6           0        1348           0       20357     17.72s
         7           0        1284           0       15416     17.74s
         8           0        1220           0       12670     17.75s
         9           0        1156           0       10775     17.76s
        10           0        1092           0        9281     17.78s
        11           0        1028           0        8304     17.79s
        12           0         964           0        7446     17.79s
        25           1         901           0        6630     17.88s
        26           1         837           0        5839     17.88s
        27           1         773           0        5034     17.89s
        28           1         709           0        4304     17.89s
        29           1         645           0        3560     17.90s
        30           1         581           0        2819     17.91s
        31           1         517           0        2141     17.91s
        32           1         453           0        1507     17.92s
        45           2         390           0        1113     17.98s
        46           2         326           0         941     17.99s

 Iteration  Min RowBnd  Max RowBnd  Min ColBnd  Max ColBnd      Time
        47           2         262           0         735     17.99s
        48           2         198           0         514     18.00s
        49           2         134           0         345     18.00s
        50           2          70           0         170     18.01s
        73           4           8           0          31     18.13s
        74           4           4           0          24     18.13s
        89           4           4           8          24     18.20s
       102           4           4          12          20     18.26s
       109           4           4          12          12     18.29s

IIS summary: 4 rows, 12 bounds of columns
IIS computation finished (18.315s)
Writing IIS problem to /Users/xue/github/facility-loc-inventory/iis/dnp.iis

[Done] exited with code=0 in 93.729 seconds

[Running] python -u "/Users/xue/github/facility-loc-inventory/dnp_model.py"
[facinv] 2023-12-15 11:40:34,691: -- The FACINV python package --
[facinv] 2023-12-15 11:40:34,691:   LLLGZ, 2023 (c)   
[facinv] 2023-12-15 11:40:34,691: :solution      to ./out
[facinv] 2023-12-15 11:40:34,691: :data          to ./data
[facinv] 2023-12-15 11:40:34,692: :logs and tmps to ./tmp
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'RelGap' to 0.0001
Setting parameter 'TimeLimit' to 3600
Model fingerprint: db854bd7

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    281433 rows, 494873 columns and 1171263 non-zero elements
    242689 binaries

Presolving the problem

The presolved problem has:
    19860 rows, 226534 columns and 604346 non-zero elements
    2489 binaries

Starting the MIP solver with 8 threads and 32 tasks

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --       0  4.668310e+06            --    Inf  6.33s
         0         1      --     342  2.021055e+07            --    Inf 18.98s
         0         1      --     317  2.021492e+07            --    Inf 20.20s
         0         1      --     268  2.022466e+07            --    Inf 22.46s
         0         1      --     243  2.022839e+07            --    Inf 23.90s
         0         1      --     235  2.023044e+07            --    Inf 25.14s
         0         1      --     227  2.023159e+07            --    Inf 26.23s
         0         1      --     256  2.023262e+07            --    Inf 27.17s
         0         1      --     257  2.023381e+07            --    Inf 28.37s
         0         1      --     230  2.023503e+07            --    Inf 29.17s
         0         1      --     213  2.023675e+07            --    Inf 32.81s
         0         1      --     156  2.023792e+07            --    Inf 35.78s
         0         1      --     149  2.023840e+07            --    Inf 37.95s
         0         1      --     136  2.023972e+07            --    Inf 40.59s
         0         1      --     117  2.024009e+07            --    Inf 42.50s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --     102  2.024033e+07            --    Inf 44.21s
         0         1      --      97  2.024067e+07            --    Inf 45.56s
         0         1      --      90  2.024122e+07            --    Inf 47.64s
         0         1      --      88  2.024183e+07            --    Inf 50.17s
         0         1      --      81  2.024201e+07            --    Inf 51.68s
         0         1      --      73  2.024214e+07            --    Inf 53.17s
         0         1      --      78  2.024234e+07            --    Inf 54.39s
H        0         0      --      78  2.024550e+07  3.366291e+07  39.9% 85.44s
H        0         0      --      78  2.024550e+07  3.366291e+07  39.9% 97.79s
         1         2   25069      78  2.024550e+07  3.366291e+07  39.9% 97.79s
         2         2   13294     127  2.024801e+07  3.366291e+07  39.9%   115s
         3         4    9741     166  2.024801e+07  3.366291e+07  39.9%   115s
         4         2    9311     228  2.025414e+07  3.366291e+07  39.8%   139s
         5         4    7917     131  2.025414e+07  3.366291e+07  39.8%   139s
         6         6    7562     187  2.025414e+07  3.366291e+07  39.8%   139s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         7         8    6868      85  2.025414e+07  3.366291e+07  39.8%   139s
         8         2    6169     239  2.025899e+07  3.366291e+07  39.8%   152s
         9         4    5924     176  2.025899e+07  3.366291e+07  39.8%   152s
        10         6    5450     194  2.025899e+07  3.366291e+07  39.8%   152s
        20        10    4033     219  2.026959e+07  3.366291e+07  39.8%   172s
        30        30    3899     181  2.026959e+07  3.366291e+07  39.8%   172s
        40        18    3236     225  2.027020e+07  3.366291e+07  39.8%   195s
        50        38    2774     194  2.027020e+07  3.366291e+07  39.8%   195s
        60        58    2486     233  2.027020e+07  3.366291e+07  39.8%   195s
        70        46    2215     217  2.027020e+07  3.366291e+07  39.8%   210s
        80        66    1993     166  2.027020e+07  3.366291e+07  39.8%   210s
        90        86    1840     226  2.027020e+07  3.366291e+07  39.8%   210s
       100        74    1749     213  2.027020e+07  3.366291e+07  39.8%   227s
       110        94    1658     141  2.027020e+07  3.366291e+07  39.8%   227s
       120       114    1600     223  2.027020e+07  3.366291e+07  39.8%   227s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
       130       102    1554     175  2.027020e+07  3.366291e+07  39.8%   259s
       140       122    1607     222  2.027020e+07  3.366291e+07  39.8%   259s
       150       142    1668     160  2.027020e+07  3.366291e+07  39.8%   259s
       160       130    1694     146  2.027020e+07  3.366291e+07  39.8%   280s
       170       150    1613     215  2.027020e+07  3.366291e+07  39.8%   280s
       180       170    1555     160  2.027020e+07  3.366291e+07  39.8%   280s
       190       190    1526     135  2.027020e+07  3.366291e+07  39.8%   280s
       200       178    1627     171  2.027020e+07  3.366291e+07  39.8%   318s
       300       282    1655     169  2.027020e+07  3.366291e+07  39.8%   365s
       400       386    1754     167  2.027020e+07  3.366291e+07  39.8%   427s
       500       490    1460     164  2.027020e+07  3.366291e+07  39.8%   442s
       600        52    1531     188  2.027152e+07  3.366291e+07  39.8%   662s
       700       156    1621     159  2.027152e+07  3.366291e+07  39.8%   728s
       800       228    1589     151  2.027152e+07  3.366291e+07  39.8%   785s
       900       332    1477     208  2.027152e+07  3.366291e+07  39.8%   842s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
      1000       436    1406     142  2.027152e+07  3.366291e+07  39.8%   869s
H     1088       514    1345     163  2.027152e+07  2.053870e+07  1.30%   902s
      1100       540    1336     101  2.027152e+07  2.053870e+07  1.30%   902s
      1200       644    1296     190  2.027232e+07  2.053870e+07  1.30%   955s
      1300       748    1311     110  2.027321e+07  2.053870e+07  1.29%  1008s
      1400       852    1291     107  2.027321e+07  2.053870e+07  1.29%  1044s
      1500       956    1261     107  2.027321e+07  2.053870e+07  1.29%  1080s
      1600      1028    1221      41  2.027321e+07  2.053870e+07  1.29%  1109s
      1700      1132    1184      79  2.027321e+07  2.053870e+07  1.29%  1133s
H     1760      1186    1163      37  2.027321e+07  2.028268e+07  0.05%  1146s
      1800        78    1150      45  2.027333e+07  2.028268e+07  0.05%  1154s
      1900       166    1117     121  2.027406e+07  2.028268e+07  0.04%  1175s
      2000       262    1083      45  2.027407e+07  2.028268e+07  0.04%  1191s
      3000      1244   794.0       6  2.027415e+07  2.028268e+07  0.04%  1340s
H     3065      1306   781.1       7  2.027416e+07  2.027625e+07  0.01%  1345s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
H     3073       700   779.6       1  2.027416e+07  2.027522e+07  0.01%  1348s
H     3080       708   777.8       6  2.027416e+07  2.027514e+07  0.00%  1348s
      3102       336   772.9       7  2.027416e+07  2.027514e+07  0.00%  1348s

Best solution   : 20275142.373650800
Best bound      : 20274161.462090638
Best gap        : 0.0048%
Solve time      : 1349.01
Solve node      : 3102
MIP status      : solved
Solution status : integer optimal (relative gap limit 0.0001)

Violations      :     absolute     relative
  bounds        :  6.97787e-10  6.97787e-10
  rows          :  2.71177e-10  2.71177e-10
  integrality   :            0
[facinv] 2023-12-15 12:04:38,573: table warehouse_total_avg_inventory failed
[facinv] 2023-12-15 12:04:38,690: saving finished
0:24:03.998247

[Done] exited with code=0 in 1446.079 seconds

