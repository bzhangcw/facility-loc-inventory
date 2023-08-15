[facinv] 2023-08-13 20:25:23,960: -- The FACINV python package --
[facinv] 2023-08-13 20:25:23,960:   LLLGZ, 2023 (c)   
[facinv] 2023-08-13 20:25:23,960: :solution      to ./out
[facinv] 2023-08-13 20:25:23,961: :data          to ./data
[facinv] 2023-08-13 20:25:23,961: :logs and tmps to ./tmp
[facinv] 2023-08-13 20:25:23,968: generating the signature of this problem
[facinv] 2023-08-13 20:25:23,968: {
  "customer_num": 4,
  "data_dir": "data/data_0401_0inv.xlsx",
  "one_period": false,
  "plant_num": 10,
  "sku_num": 50,
  "warehouse_num": 10
}
[facinv] 2023-08-13 20:25:23,968: current data has not been generated before
[facinv] 2023-08-13 20:25:23,968: creating a temporary cache @./out/4-False-10-50-10.pk
[facinv] 2023-08-13 20:26:00,456: dumping a temporary cache @./out/4-False-10-50-10.pk
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

Setting parameter 'Logging' to 1
Model fingerprint: c6253e9b

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing a MIP problem

The original problem has:
    9335 rows, 13006 columns and 33222 non-zero elements
    5119 binaries

Presolving the problem

The presolved problem has:
    688 rows, 3663 columns and 9791 non-zero elements
    79 binaries

Starting the MIP solver with 8 threads and 32 tasks

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --       0 -1.423820e+07            --    Inf  0.13s
H        0         1      --       0 -1.423820e+07  7.574504e+06 153.2%  0.13s
         0         1      --      18  1.110892e+06  7.574504e+06  85.3%  0.21s
         0         1      --      18  1.110892e+06  7.574504e+06  85.3%  0.21s
         0         1      --      23  1.115550e+06  7.574504e+06  85.3%  0.22s
         0         1      --      23  1.115550e+06  7.574504e+06  85.3%  0.23s
         0         1      --      20  1.125811e+06  7.574504e+06  85.1%  0.23s
         0         1      --      20  1.125811e+06  7.574504e+06  85.1%  0.23s
         0         1      --      19  1.130958e+06  7.574504e+06  85.1%  0.24s
         0         1      --      19  1.130958e+06  7.574504e+06  85.1%  0.24s
         0         1      --      18  1.137700e+06  7.574504e+06  85.0%  0.25s
         0         1      --      18  1.137700e+06  7.574504e+06  85.0%  0.25s
         0         1      --      18  1.492398e+06  7.574504e+06  80.3%  0.25s
         0         1      --      18  1.492398e+06  7.574504e+06  80.3%  0.25s
         0         1      --      17  1.506423e+06  7.574504e+06  80.1%  0.26s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --      17  1.506423e+06  7.574504e+06  80.1%  0.26s
         0         1      --      15  1.508683e+06  7.574504e+06  80.1%  0.27s
         0         1      --      15  1.508683e+06  7.574504e+06  80.1%  0.27s
         0         1      --      15  1.510583e+06  7.574504e+06  80.1%  0.27s
         0         1      --      15  1.510583e+06  7.574504e+06  80.1%  0.27s
         0         1      --      15  1.516348e+06  7.574504e+06  80.0%  0.28s
         0         1      --      15  1.516348e+06  7.574504e+06  80.0%  0.28s
         0         1      --      15  1.516930e+06  7.574504e+06  80.0%  0.29s
         0         1      --      15  1.516930e+06  7.574504e+06  80.0%  0.29s
         0         1      --      15  1.518383e+06  7.574504e+06  80.0%  0.30s
         0         1      --      15  1.518383e+06  7.574504e+06  80.0%  0.30s
         0         1      --      16  1.520268e+06  7.574504e+06  79.9%  0.30s
         0         1      --      16  1.520268e+06  7.574504e+06  79.9%  0.30s
         0         1      --      16  1.520813e+06  7.574504e+06  79.9%  0.31s
         0         1      --      16  1.520813e+06  7.574504e+06  79.9%  0.31s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --      17  1.524839e+06  7.574504e+06  79.9%  0.32s
         0         1      --      17  1.524839e+06  7.574504e+06  79.9%  0.32s
         0         1      --      17  1.526373e+06  7.574504e+06  79.8%  0.33s
         0         1      --      17  1.526373e+06  7.574504e+06  79.8%  0.33s
         0         1      --      17  1.527229e+06  7.574504e+06  79.8%  0.34s
         0         1      --      17  1.527229e+06  7.574504e+06  79.8%  0.34s
         0         1      --      16  1.530299e+06  7.574504e+06  79.8%  0.34s
         0         1      --      16  1.530299e+06  7.574504e+06  79.8%  0.34s
         0         1      --      16  1.534681e+06  7.574504e+06  79.7%  0.35s
         0         1      --      16  1.534681e+06  7.574504e+06  79.7%  0.35s
         0         1      --      18  1.559809e+06  7.574504e+06  79.4%  0.37s
         0         1      --      18  1.559809e+06  7.574504e+06  79.4%  0.37s
         0         1      --      18  1.560841e+06  7.574504e+06  79.4%  0.38s
         0         1      --      18  1.560841e+06  7.574504e+06  79.4%  0.38s
         0         1      --      19  1.566429e+06  7.574504e+06  79.3%  0.39s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --      19  1.566429e+06  7.574504e+06  79.3%  0.39s
         0         1      --      18  1.566770e+06  7.574504e+06  79.3%  0.40s
         0         1      --      18  1.566770e+06  7.574504e+06  79.3%  0.40s
         0         1      --      17  1.566955e+06  7.574504e+06  79.3%  0.40s
         0         1      --      17  1.566955e+06  7.574504e+06  79.3%  0.40s
         0         1      --      17  1.567050e+06  7.574504e+06  79.3%  0.41s
         0         1      --      17  1.567050e+06  7.574504e+06  79.3%  0.41s
         0         1      --      17  1.567174e+06  7.574504e+06  79.3%  0.42s
         0         1      --      17  1.567230e+06  7.574504e+06  79.3%  0.42s
         0         1      --      17  1.567230e+06  7.574504e+06  79.3%  0.42s
         0         1      --      17  1.567680e+06  7.574504e+06  79.3%  0.43s
         0         1      --      17  1.567680e+06  7.574504e+06  79.3%  0.43s
         0         1      --      17  1.567752e+06  7.574504e+06  79.3%  0.44s
         0         1      --      17  1.567752e+06  7.574504e+06  79.3%  0.44s
         0         1      --      17  1.568406e+06  7.574504e+06  79.3%  0.45s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         0         1      --      17  1.568406e+06  7.574504e+06  79.3%  0.45s
         0         1      --      17  1.568457e+06  7.574504e+06  79.3%  0.45s
         0         1      --      18  1.568511e+06  7.574504e+06  79.3%  0.46s
         0         1      --      17  1.568516e+06  7.574504e+06  79.3%  0.47s
         0         1      --      18  1.568528e+06  7.574504e+06  79.3%  0.47s
         0         1      --      18  1.568528e+06  7.574504e+06  79.3%  0.47s
         0         1      --      17  1.568535e+06  7.574504e+06  79.3%  0.48s
         0         1      --      18  1.568602e+06  7.574504e+06  79.3%  0.49s
         0         1      --      18  1.568730e+06  7.574504e+06  79.3%  0.50s
         0         1      --      18  1.568730e+06  7.574504e+06  79.3%  0.50s
         0         1      --      18  1.568745e+06  7.574504e+06  79.3%  0.50s
         0         1      --      18  1.568759e+06  7.574504e+06  79.3%  0.51s
         0         1      --      18  1.568760e+06  7.574504e+06  79.3%  0.52s
         1         2    1498      18  1.778914e+06  7.574504e+06  76.5%  0.56s
         2         2   763.0      18  1.789550e+06  7.574504e+06  76.4%  0.58s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
         3         4   655.3      16  1.789550e+06  7.574504e+06  76.4%  0.58s
         4         2   509.0      16  1.965277e+06  7.574504e+06  74.1%  0.62s
         5         4   556.6      17  1.965277e+06  7.574504e+06  74.1%  0.62s
         6         6   486.2      18  1.965277e+06  7.574504e+06  74.1%  0.62s
         7         8   434.0      16  1.965277e+06  7.574504e+06  74.1%  0.62s
         8         2   391.0      16  2.112700e+06  7.574504e+06  72.1%  0.63s
         9         4   361.8      14  2.112700e+06  7.574504e+06  72.1%  0.63s
        10         6   335.2      14  2.112700e+06  7.574504e+06  72.1%  0.63s
        20         9   217.1      17  2.327905e+06  7.574504e+06  69.3%  0.64s
        30         6   171.2      18  2.463013e+06  7.574504e+06  67.5%  0.68s
H       40        22   152.0      13  2.463013e+06  7.352431e+06  66.5%  0.68s
        40        24   154.6      13  2.463013e+06  7.352431e+06  66.5%  0.68s
H       48         8   136.9      10  2.658459e+06  5.853419e+06  54.6%  0.71s
H       49        10   136.0      11  2.658459e+06  4.283160e+06  37.9%  0.71s
        50        14   135.1      13  2.658459e+06  4.283160e+06  37.9%  0.71s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
        60        14   122.5      14  2.658459e+06  4.283160e+06  37.9%  0.71s
        70        14   116.1      14  2.658459e+06  4.283160e+06  37.9%  0.71s
H       73         2   111.8      10  2.658459e+06  3.349382e+06  20.6%  0.72s
        80         2   104.7      10  2.684113e+06  3.349382e+06  19.9%  0.73s
        90         2    99.2       7  2.826024e+06  3.349382e+06  15.6%  0.75s
       100         8    91.6       5  2.862282e+06  3.349382e+06  14.5%  0.75s
H      106         4    86.7       9  2.911326e+06  3.280422e+06  11.3%  0.76s
H      110         0    84.8       5  2.911327e+06  3.250675e+06  10.4%  0.77s
       110         2    87.5       5  2.911327e+06  3.250675e+06  10.4%  0.77s
       120         8    86.2       5  2.913412e+06  3.250675e+06  10.4%  0.78s
*      125         2    84.3       0  2.966984e+06  3.185583e+06  6.86%  0.79s
       130         6    84.4       3  2.966984e+06  3.185583e+06  6.86%  0.79s
       140        14    81.1       3  2.966984e+06  3.185583e+06  6.86%  0.80s
H      144         4    79.0       4  2.966984e+06  3.048479e+06  2.67%  0.80s
       150        10    76.1       3  2.966984e+06  3.048479e+06  2.67%  0.80s

     Nodes    Active  LPit/n  IntInf     BestBound  BestSolution    Gap   Time
*      159         2    71.9       0  2.966984e+06  2.980348e+06  0.45%  0.81s
       160         4    71.9       3  2.966984e+06  2.980348e+06  0.45%  0.81s
*      161         4    71.5       0  2.966984e+06  2.967007e+06  0.00%  0.81s
       169         6    69.3       4  2.966984e+06  2.967007e+06  0.00%  0.81s

Best solution   : 2967006.540914614
Best bound      : 2966984.154055137
Best gap        : 0.0008%
Solve time      : 0.81
Solve node      : 169
MIP status      : solved
Solution status : integer optimal (relative gap limit 0.0001)

Violations      :     absolute     relative
  bounds        :  9.31323e-10  9.31323e-10
  rows          :  2.55974e-10  2.55974e-10
  integrality   :            0
[facinv] 2023-08-13 20:26:01,467: table warehouse_total_avg_inventory failed
{'customer_fullfill_rate': 0.6890689286625993, 'warehouse_fullfill_rate': 1, 'overall_fullfill_rate': 0.6890689286625993, 'warehouse_overall_avg_inventory': 0}
[facinv] 2023-08-13 20:26:01,486: saving finished
----------DCG Model------------
[facinv] 2023-08-13 20:26:01,486: the CG algorithm chooses verbosity at CG_EXTRA_VERBOSITY: 1
Cardinal Optimizer v6.5.1. Build date Apr 11 2023
Copyright Cardinal Operations 2023. All Rights Reserved

[facinv] 2023-08-13 20:26:01,487: initialization complete, start generating columns...
[facinv] 2023-08-13 20:26:01,487: generating column oracles

  0%|          | 0/4 [00:00<?, ?it/s]
100%|██████████| 4/4 [00:00<00:00, 29.94it/s]
100%|██████████| 4/4 [00:00<00:00, 29.91it/s]
[facinv] 2023-08-13 20:26:01,633: generating column helpers

  0%|          | 0/4 [00:00<?, ?it/s]
100%|██████████| 4/4 [00:00<00:00, 1358.26it/s]
Setting parameter 'Logging' to 1
[facinv] 2023-08-13 20:26:03,084: initialization of restricted master finished
[facinv] 2023-08-13 20:26:03,084: solving the first rmp
Setting parameter 'LpMethod' to 2
Setting parameter 'Crossover' to 0
Model fingerprint: 4467a66a

Using Cardinal Optimizer v6.5.1 on macOS (aarch64)
Hardware has 8 cores and 8 threads. Using instruction set ARMV8 (30)
Minimizing an LP problem

The original problem has:
    126 rows, 4 columns and 38 non-zero elements
The presolved problem is empty

Postsolving

Method   Iteration           Objective  Primal.NInf   Dual.NInf        Time
Dual             0    3.7082547457e+06            0           0       0.00s

Solving finished
Status: Optimal  Objective: 3.7082547457e+06  Iterations: 0  Time: 0.00s
k:      1 / 500  f: 3.708255e+06  c': -1.1005e+01
k:      2 / 500  f: 3.708248e+06  c': -9.1594e+00
k:      3 / 500  f: 3.708244e+06  c': -7.9162e-09
[facinv] 2023-08-13 20:26:07,849: 
=== describing time statistics ===
                 name  count      mean  ...       50%       75%       max
0           get_duals    3.0  0.000021  ...  0.000022  0.000023  0.000024
1  initialize_columns    1.0  0.148932  ...  0.148932  0.148932  0.148932
2      initialize_rmp    1.0  0.002396  ...  0.002396  0.002396  0.002396
3       solve_columns    3.0  1.581482  ...  1.642738  1.714346  1.785953
4           solve_rmp    3.0  0.000524  ...  0.000102  0.000736  0.001369
5          update_rmp    2.0  0.000463  ...  0.000463  0.000473  0.000483

[6 rows x 9 columns]
    
[facinv] 2023-08-13 20:26:07,849: save solutions to ./out

⬆️2T4C的情况

1T4C:
DNP: 1957328.665535010
CG: 2.417077e+06
猜测的原因：CG是对每一个customer的oracle加的lower bound

验证之后发现：
4T1C可以对的上 所以显然是因为上述猜测的原因导致的
DNP：14614.796652500
CG：1.461480e+04

想到的解决方案：
1. 把Lower bound在CG中去掉
2. 然后在RMP中加Lower bound 然后对上index--问题是edge 是否选为0，1这怎么在RMP中体现呢？
3. 但是可能存在的问题是emm初始可行解怎么找

所以Maybe更改之后：
1. 正常的Oracle不加Lower bound约束（干脆全部让np_cg中的‘bool_edge_lb = False’）
2. 然后在init_col的代码中 if lb<np.inf ： 那么额外为初始化的列加lower bound约束（就是>=edge.lb）然后初始化 然后oracle把这些删除掉