import coptpy as cp

env = cp.Envr()
lp1 = env.createModel("lp1")
lp2 = env.createModel("lp2")

# cus_idx = "C00120730"
cus_idx = "669"


lp1_path = (
    f"/Users/liu/Desktop/MyRepositories/facility-loc-inventory/oracle_lp/{cus_idx}.lp"
    # r"/Users/liu/Desktop/MyRepositories/facility-loc-inventory/oracle_lp/init_rmp.lp"
)
lp2_path = (
    f"/Users/liu/Desktop/MyRepositories/facility-loc-inventory_copy/lp/{cus_idx}.lp"
    # r"/Users/liu/Desktop/MyRepositories/facility-loc-inventory_copy/lp/init_rmp.lp"
)

lp1.readLp(lp1_path)
lp2.readLp(lp2_path)

lp1.solve()
# lp1.computeIIS()
# lp1.writeIIS("oracle_lp/init_rmp.iis")
lp2.solve()
