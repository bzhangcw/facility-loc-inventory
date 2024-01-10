/Users/xue/miniforge3/bin/python3.10 /Users/xue/github/facility-loc-inventory/rounding.py
[facinv] 2024-01-10 15:06:41,863: -- The FACINV python package --
[facinv] 2024-01-10 15:06:41,863:   LLLGZ, 2023 (c)
[facinv] 2024-01-10 15:06:41,863: :solution      to ./out
[facinv] 2024-01-10 15:06:41,863: :data          to ./data
[facinv] 2024-01-10 15:06:41,863: :logs and tmps to ./tmp
save mps name allinone_data_0401_0inv_7_2@4.mps
[facinv] 2024-01-10 15:06:41,878: time scale 7
[facinv] 2024-01-10 15:06:41,878: generating the signature of this problem
[facinv] 2024-01-10 15:06:41,878: {
  "customer_num": 10,
  "data_dir": "data/data_0401_0inv.xlsx",
  "one_period": false,
  "plant_num": 20,
  "sku_num": 30,
  "warehouse_num": 20
}
[facinv] 2024-01-10 15:06:41,878: current data has been generated before
[facinv] 2024-01-10 15:06:41,878: reading from cache: ./out/data_0401_0inv-10-False-20-30-20.pk
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Setting parameter 'Threads' to 8
Setting parameter 'TimeLimit' to 3600
Setting parameter 'Crossover' to 0
Writing problem to /Users/xue/github/facility-loc-inventory/allinone_data_0401_0inv_7_2@4.mps
Set parameter Username
Academic license - for non-commercial use only - expires 2024-07-05
Read MPS format model from file allinone_data_0401_0inv_7_2@4.mps
Reading time = 0.06 seconds
COPTPROB: 27607 rows, 107267 columns, 237563 nonzeros
----------DNP Model(MIP)------------
Gurobi Optimizer version 10.0.2 build v10.0.2rc0 (mac64[arm])

CPU model: Apple M1 Pro
Thread count: 8 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 27607 rows, 107267 columns and 237563 nonzeros
Model fingerprint: 0x77e50b06
Variable types: 55526 continuous, 51741 integer (0 binary)
Coefficient statistics:
  Matrix range     [1e+00, 1e+08]
  Objective range  [7e-02, 5e+02]
  Bounds range     [1e+00, 4e+04]
  RHS range        [1e+00, 7e+05]
Found heuristic solution: objective 2.501904e+08
Presolve removed 21580 rows and 57624 columns
Presolve time: 0.12s
Presolved: 6027 rows, 49643 columns, 130107 nonzeros
Variable types: 49519 continuous, 124 integer (124 binary)
Deterministic concurrent LP optimizer: primal and dual simplex
Showing first log only...

Concurrent spin time: 0.00s

Solved with dual simplex

Root relaxation: objective 5.279606e+06, 1539 iterations, 0.06 seconds (0.09 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 5279605.91    0    8 2.5019e+08 5279605.91  97.9%     -    0s
     0     0 5279605.91    0    8 2.5019e+08 5279605.91  97.9%     -    0s
     0     0 5279605.91    0    8 2.5019e+08 5279605.91  97.9%     -    0s
H    0     0                    5280805.7615 5279627.83  0.02%     -    0s
     0     0 5280137.71    0    8 5280805.76 5280137.71  0.01%     -    0s
     0     0 5280196.08    0    7 5280805.76 5280196.08  0.01%     -    0s
     0     0 5280665.42    0    4 5280805.76 5280665.42  0.00%     -    0s

Cutting planes:
  Gomory: 4
  Cover: 1
  Implied bound: 10
  MIR: 5
  Flow cover: 6
  Relax-and-lift: 1

Explored 1 nodes (3122 simplex iterations) in 0.58 seconds (0.68 work units)
Thread count was 8 (of 8 available processors)

Solution count 2: 5.28081e+06 2.5019e+08

Optimal solution found (tolerance 1.00e-04)
Best objective 5.280805761508e+06, best bound 5.280665420412e+06, gap 0.0027%
----------DNP Model(LP)------------
Gurobi Optimizer version 10.0.2 build v10.0.2rc0 (mac64[arm])

CPU model: Apple M1 Pro
Thread count: 8 physical cores, 8 logical processors, using up to 8 threads

Optimize a model with 27607 rows, 107267 columns and 237563 nonzeros
Model fingerprint: 0xb383506c
Coefficient statistics:
  Matrix range     [1e+00, 1e+08]
  Objective range  [7e-02, 5e+02]
  Bounds range     [1e+00, 4e+04]
  RHS range        [1e+00, 7e+05]
Presolve removed 20196 rows and 56611 columns
Presolve time: 0.05s
Presolved: 7411 rows, 50656 columns, 147370 nonzeros

Concurrent LP optimizer: primal simplex, dual simplex, and barrier
Showing barrier log only...

Ordering time: 0.05s

Barrier performed 0 iterations in 0.12 seconds (0.17 work units)
Barrier solve interrupted - model solved by another algorithm


Solved with dual simplex
Iteration    Objective       Primal Inf.    Dual Inf.      Time
    1927    6.8927011e+05   0.000000e+00   0.000000e+00      0s

Solved in 1927 iterations and 0.14 seconds (0.17 work units)
Optimal objective  6.892701066e+05
----------DCS Model------------
[facinv] 2024-01-10 15:06:44,347: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
[facinv] 2024-01-10 15:06:44,351: initialization complete, start generating columns...
[facinv] 2024-01-10 15:06:44,351: generating column oracles
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

100%|██████████| 10/10 [00:00<00:00, 30.78it/s]
removed initial skeleton
Setting parameter 'Logging' to 1
[facinv] 2024-01-10 15:06:45,879: initialization of restricted master finished
[facinv] 2024-01-10 15:06:45,879: solving the first rmp
Setting parameter 'Crossover' to 0
Model fingerprint: 778c8d4b

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing an LP problem

The original problem has:
    19890 rows, 59090 columns and 205490 non-zero elements
The presolved problem has:
    4228 rows, 44688 columns and 162138 non-zero elements

Starting the simplex solver using up to 8 threads

Method   Iteration           Objective  Primal.NInf   Dual.NInf        Time
Dual             0    2.5019036000e+08            0           0       0.09s
Postsolving
Dual             0    2.5019036000e+08            0           0       0.09s

Solving finished
Status: Optimal  Objective: 2.5019036000e+08  Iterations: 0  Time: 0.09s
[facinv] 2024-01-10 15:06:46,342: k:     1 / 1000 f: 2.501904e+08, c': 1.8470e+04
[facinv] 2024-01-10 15:06:46,969: k:     2 / 1000 f: 5.354827e+06, c': -1.2513e+08
-----Solve MIP_RMP-----
2 MIP_RMP 5354827.125552319 GAP 0.0
Reset Over
[facinv] 2024-01-10 15:06:49,968: k:     3 / 1000 f: 5.353796e+06, c': -4.3373e+04
[facinv] 2024-01-10 15:06:50,736: k:     4 / 1000 f: 5.310230e+06, c': -1.1492e+05
-----Solve MIP_RMP-----
4 MIP_RMP 5313125.372144524 GAP 2895.1425784584135
Reset Over
[facinv] 2024-01-10 15:06:54,634: k:     5 / 1000 f: 5.310199e+06, c': -9.3089e+04
[facinv] 2024-01-10 15:06:55,384: k:     6 / 1000 f: 5.310124e+06, c': -1.2534e+05
-----Solve MIP_RMP-----
6 MIP_RMP 5313098.391199382 GAP 2974.3835443658754
Reset Over
[facinv] 2024-01-10 15:06:59,272: k:     7 / 1000 f: 5.310124e+06, c': -1.2091e+05
[facinv] 2024-01-10 15:06:59,980: k:     8 / 1000 f: 5.310124e+06, c': -1.4428e+05
-----Solve MIP_RMP-----
8 MIP_RMP 5313098.391199835 GAP 2974.383544818498
Reset Over
[facinv] 2024-01-10 15:07:03,987: k:     9 / 1000 f: 5.308931e+06, c': -5.0133e+05
[facinv] 2024-01-10 15:07:04,714: k:    10 / 1000 f: 5.308863e+06, c': -1.0508e+05
-----Solve MIP_RMP-----
10 MIP_RMP 5312040.74529079 GAP 3177.2692022398114
Reset Over
[facinv] 2024-01-10 15:07:08,729: k:    11 / 1000 f: 5.308632e+06, c': -8.3350e+04
[facinv] 2024-01-10 15:07:09,437: k:    12 / 1000 f: 5.308238e+06, c': -8.3146e+04
-----Solve MIP_RMP-----
12 MIP_RMP 5311750.317115729 GAP 3512.1673788735643
Reset Over
[facinv] 2024-01-10 15:07:13,769: k:    13 / 1000 f: 5.300060e+06, c': -1.3098e+05
[facinv] 2024-01-10 15:07:14,556: k:    14 / 1000 f: 5.300060e+06, c': -6.2852e+04
-----Solve MIP_RMP-----
14 MIP_RMP 5301628.386377602 GAP 1568.8504018271342
Reset Over
[facinv] 2024-01-10 15:07:19,208: k:    15 / 1000 f: 5.300060e+06, c': -9.7761e+04
[facinv] 2024-01-10 15:07:20,084: k:    16 / 1000 f: 5.299049e+06, c': -3.0649e+05
-----Solve MIP_RMP-----
16 MIP_RMP 5300735.315818769 GAP 1686.5596495382488
Reset Over
[facinv] 2024-01-10 15:07:25,216: k:    17 / 1000 f: 5.299023e+06, c': -1.3607e+05
[facinv] 2024-01-10 15:07:25,997: k:    18 / 1000 f: 5.298468e+06, c': -1.5398e+05
-----Solve MIP_RMP-----
18 MIP_RMP 5299171.997454025 GAP 704.4936317186803
Reset Over
[facinv] 2024-01-10 15:07:31,432: k:    19 / 1000 f: 5.298423e+06, c': -7.4275e+04
[facinv] 2024-01-10 15:07:32,373: k:    20 / 1000 f: 5.298423e+06, c': -7.8633e+04
-----Solve MIP_RMP-----
20 MIP_RMP 5299171.997443894 GAP 748.618692102842
Reset Over
[facinv] 2024-01-10 15:07:36,903: k:    21 / 1000 f: 5.298423e+06, c': -1.1176e+05
[facinv] 2024-01-10 15:07:37,702: k:    22 / 1000 f: 5.298423e+06, c': -5.6596e+04
-----Solve MIP_RMP-----
22 MIP_RMP 5299171.997451162 GAP 748.6186993503943
Reset Over
[facinv] 2024-01-10 15:07:42,833: k:    23 / 1000 f: 5.298175e+06, c': -5.2209e+04
[facinv] 2024-01-10 15:07:43,842: k:    24 / 1000 f: 5.297484e+06, c': -4.6720e+04
-----Solve MIP_RMP-----
24 MIP_RMP 5298001.708525593 GAP 517.7780575659126
Reset Over
[facinv] 2024-01-10 15:07:49,646: k:    25 / 1000 f: 5.296758e+06, c': -5.2295e+04
[facinv] 2024-01-10 15:07:50,619: k:    26 / 1000 f: 5.296469e+06, c': -6.3394e+04
-----Solve MIP_RMP-----
26 MIP_RMP 5297289.183554949 GAP 819.7838332569227
Reset Over
[facinv] 2024-01-10 15:07:56,012: k:    27 / 1000 f: 5.296130e+06, c': -6.3985e+04
[facinv] 2024-01-10 15:07:56,785: k:    28 / 1000 f: 5.296103e+06, c': -1.3066e+05
-----Solve MIP_RMP-----
28 MIP_RMP 5297289.183555013 GAP 1186.4383972454816
Reset Over
[facinv] 2024-01-10 15:08:04,086: k:    29 / 1000 f: 5.296103e+06, c': -5.8281e+04
[facinv] 2024-01-10 15:08:05,719: k:    30 / 1000 f: 5.295773e+06, c': -6.7183e+04
-----Solve MIP_RMP-----
30 MIP_RMP 5296486.474446868 GAP 713.5401581069455
Reset Over
[facinv] 2024-01-10 15:08:13,463: k:    31 / 1000 f: 5.295416e+06, c': -6.6997e+05
[facinv] 2024-01-10 15:08:14,414: k:    32 / 1000 f: 5.295350e+06, c': -1.3893e+05
-----Solve MIP_RMP-----
32 MIP_RMP 5296486.474446954 GAP 1136.0830381261185
Reset Over
[facinv] 2024-01-10 15:08:19,583: k:    33 / 1000 f: 5.295328e+06, c': -1.1162e+05
[facinv] 2024-01-10 15:08:20,558: k:    34 / 1000 f: 5.295311e+06, c': -5.3089e+04
-----Solve MIP_RMP-----
34 MIP_RMP 5296920.949054831 GAP 1610.0739481709898
Reset Over
[facinv] 2024-01-10 15:08:26,803: k:    35 / 1000 f: 5.294904e+06, c': -4.4434e+04
[facinv] 2024-01-10 15:08:27,953: k:    36 / 1000 f: 5.293755e+06, c': -5.4260e+04
-----Solve MIP_RMP-----
36 MIP_RMP 5296580.56912514 GAP 2825.5345779424533
Reset Over
[facinv] 2024-01-10 15:08:33,926: k:    37 / 1000 f: 5.293755e+06, c': -4.5736e+04
[facinv] 2024-01-10 15:08:34,983: k:    38 / 1000 f: 5.293755e+06, c': -5.0428e+04
-----Solve MIP_RMP-----
38 MIP_RMP 5296486.474446362 GAP 2731.4398991707712
Reset Over
[facinv] 2024-01-10 15:08:42,040: k:    39 / 1000 f: 5.293647e+06, c': -6.3156e+04
[facinv] 2024-01-10 15:08:43,023: k:    40 / 1000 f: 5.293647e+06, c': -3.8184e+04
-----Solve MIP_RMP-----
40 MIP_RMP 5296486.474446394 GAP 2839.274775631726
Reset Over
[facinv] 2024-01-10 15:08:51,039: k:    41 / 1000 f: 5.293647e+06, c': -3.8500e+04
[facinv] 2024-01-10 15:08:52,261: k:    42 / 1000 f: 5.293105e+06, c': -5.7873e+04
-----Solve MIP_RMP-----
42 MIP_RMP 5296502.725026509 GAP 3397.51811038889
Reset Over
[facinv] 2024-01-10 15:08:57,609: k:    43 / 1000 f: 5.293105e+06, c': -5.0751e+04
[facinv] 2024-01-10 15:08:59,326: k:    44 / 1000 f: 5.293105e+06, c': -4.2760e+04
-----Solve MIP_RMP-----
44 MIP_RMP 5296257.142925467 GAP 3151.936009315774
Reset Over
[facinv] 2024-01-10 15:09:07,764: k:    45 / 1000 f: 5.293105e+06, c': -6.2096e+04
[facinv] 2024-01-10 15:09:16,804: k:    46 / 1000 f: 5.293105e+06, c': -4.8889e+04
-----Solve MIP_RMP-----
46 MIP_RMP 5296069.231359367 GAP 2964.0244431728497
Reset Over
[facinv] 2024-01-10 15:09:26,969: k:    47 / 1000 f: 5.293105e+06, c': -6.3637e+04
[facinv] 2024-01-10 15:09:51,951: k:    48 / 1000 f: 5.293105e+06, c': -6.2623e+04
-----Solve MIP_RMP-----
48 MIP_RMP 5296143.610934538 GAP 3038.4040183937177
Reset Over
[facinv] 2024-01-10 15:10:01,289: k:    49 / 1000 f: 5.293105e+06, c': -6.2541e+04
[facinv] 2024-01-10 15:10:47,149: k:    50 / 1000 f: 5.293105e+06, c': -1.2652e+05
-----Solve MIP_RMP-----
50 MIP_RMP 5296502.725026528 GAP 3397.518110392615
Reset Over
[facinv] 2024-01-10 15:10:57,333: k:    51 / 1000 f: 5.293105e+06, c': -1.0808e+05
[facinv] 2024-01-10 15:12:01,295: k:    52 / 1000 f: 5.293105e+06, c': -1.9830e+05
-----Solve MIP_RMP-----
52 MIP_RMP 5296143.610934655 GAP 3038.4040185948834
Reset Over
[facinv] 2024-01-10 15:12:17,712: k:    53 / 1000 f: 5.293105e+06, c': -1.0215e+05
[facinv] 2024-01-10 15:13:57,046: k:    54 / 1000 f: 5.293105e+06, c': -7.1889e+04
-----Solve MIP_RMP-----
54 MIP_RMP 5302643.21436657 GAP 9538.007450481877
Reset Over
[facinv] 2024-01-10 15:14:14,635: k:    55 / 1000 f: 5.293105e+06, c': -6.3120e+04
[facinv] 2024-01-10 15:16:12,820: k:    56 / 1000 f: 5.293105e+06, c': -1.0967e+05
-----Solve MIP_RMP-----
56 MIP_RMP 5296414.5631856425 GAP 3309.3562695272267
Reset Over
[facinv] 2024-01-10 15:16:33,134: k:    57 / 1000 f: 5.293105e+06, c': -1.3555e+05
[facinv] 2024-01-10 15:18:54,812: k:    58 / 1000 f: 5.293105e+06, c': -1.0873e+05
-----Solve MIP_RMP-----
58 MIP_RMP 5296610.5735924 GAP 3505.3666762504727
Reset Over
[facinv] 2024-01-10 15:19:18,092: k:    59 / 1000 f: 5.293105e+06, c': -1.1101e+05
[facinv] 2024-01-10 15:21:48,410: k:    60 / 1000 f: 5.293105e+06, c': -1.0383e+05
-----Solve MIP_RMP-----
60 MIP_RMP 5296502.725026536 GAP 3397.5181103870273
Reset Over
