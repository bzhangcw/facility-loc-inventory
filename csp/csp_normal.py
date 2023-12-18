#
# This file is part of the Cardinal Optimizer, all rights reserved.
#
from coptpy import *

# Cutting-stock data
rollwidth  = 110

rollsize   = [25, 40, 50, 55, 70]
rolldemand = [50, 36, 24, 8, 30]
# rollsize   = [25, 40]
# rolldemand = [50, 36]
nkind      = len(rollsize)
ninitpat   = 10000

rollsize_dict = {j: rollsize[j] for j in range(len(rollsize))}

# Maximal number of CG iterations
MAX_CGTIME = 100
# Create COPT environment
def heuristic(rollsize_dict,nkind, ninitpat, rollwidth, rolldemand):
    env = Envr()
    # Create RMP and SUB model
    pricing = env.createModel("pricing")
    # Disable log information
    pricing.setParam(COPT.Param.Logging, 0)
    # Build the RMP model

    nbr = pricing.addVars(nkind, ninitpat, vtype=COPT.INTEGER, nameprefix='vuse')

    constrs = {}
    # add constraints
    constr_types = {
                "capacity": {"index": "(i)"},
                # "slack": {"index": "(j)"},
                "demand": {"index": "(j)"},
            }

    for constr in constr_types.keys():
        constrs[constr] = dict()

    for j in range(ninitpat):
      con = 0
      for i in range(nkind):
        con += nbr[i, j] * rollsize_dict[i]
      constr = pricing.addConstr((con <= rollwidth), name='capacity{}'.format(j))
      constrs["capacity"][j] = constr

    for i in range(nkind):
        con = 0
        for j in range(ninitpat):
            con += nbr[i, j]
        constr = pricing.addConstr((con >= rolldemand[i]), name='demand{}'.format(i))
        constrs["demand"][i] = constr
    for j in range(ninitpat):
        obj = 0
        for i in range(nkind):
            con += nbr[i, j] * rollsize_dict[i]
        obj += rollwidth- con
    pricing.setObjective(0, COPT.MINIMIZE)
    pricing.solve()
    if pricing.status == COPT.OPTIMAL:
        vars = pricing.getVars()
        return vars
        # for var in vars:
        #     if var.x > 1e-6:
        #         print("  {0} = {1:.6f}".format(var.name, var.x))

nbr = heuristic(rollsize_dict, nkind, ninitpat, rollwidth, rolldemand)
# for iter in range(100):
#     print(iter)
#     heuristic(rollsize_dict, nkind, ninitpat, rollwidth)
#     nbr

# Report solution of the RMP model
def reportRMP(rmpmodel):
  if rmpmodel.status == COPT.OPTIMAL:
    print("Using {0} rolls".format(rmpmodel.objval))

    rmpvars = rmpmodel.getVars()
    for var in rmpvars:
      if var.x > 1e-6:
        print("  {0} = {1:.6f}".format(var.name, var.x))
  else:
    print(rmpmodel.LpStatus == COPT.INFEASIBLE)
    print(rmpmodel.LpStatus == COPT.UNBOUNDED)
    # rmpmodel.computeIIS()
    # rmpmodel.writeIIS("exa.iis")
    # rmpvars = rmpmodel.getVars()
    # for var in rmpvars:
    #     print("  {0} = {1:.6f}".format(var.name, var.x))

# Create COPT environment
env = Envr()
# Create RMP and SUB model
mCutOpt = env.createModel("mCutOpt")

# Disable log information
mCutOpt.setParam(COPT.Param.Logging, 0)

# Build the RMP model
vcut = mCutOpt.addVars(ninitpat,lb=0, nameprefix="vcut")
# vcut = mCutOpt.addVars(ninitpat,lb=0,vtype=COPT.INTEGER, nameprefix="vcut")
# vuse = mCutOpt.addVars(nkind, vtype=COPT.INTEGER, nameprefix='vuse')
# nbr = mCutOpt.addVars(nkind, ninitpat, vtype=COPT.INTEGER, nameprefix='vuse')
slack_demand = mCutOpt.addVars(ninitpat, nameprefix="slack_demand")

# For each width, roll cuts should meet demands
# for i in range(nkind):
#   rowdata = [0.0] * ninitpat
#   rowdata[i] = int(rollwidth / rollsize[i])
#   nbr.append(rowdata)

constrs = {}
# add constraints
constr_types = {
            "cfill": {"index": "(i)"},
            # "capacity": {"index": "(i)"},
            "demand": {"index": "(i)"},
        }

for constr in constr_types.keys():
    constrs[constr] = dict()

for i in range(nkind):
  con = 0
  for j in range(ninitpat):
    con += nbr[i*ninitpat+j].x * vcut[j]
  constr = mCutOpt.addConstr(con + slack_demand[i] == rolldemand[i], name = 'cfill{}'.format(i))
  constrs["cfill"][i] = constr
  # constr = mCutOpt.addConstr(quicksum(nbr[i,j] * vcut[j] for j in range(ninitpat)) + slack_demand[i] == rolldemand[i],
  #                         name='cfill{}'.format(i),
  #                     )
  # constrs["cfill"][i] = constr

# for j in range(ninitpat):
#   con = 0
#   for i in range(nkind):
#     con += nbr[i*ninitpat+j].x * rollsize_dict[i]
#   constr = mCutOpt.addConstr((con <= rollwidth ),name = 'capacity{}'.format(j))
#   constrs["capacity"][j] = constr

for i in range(nkind):
  constr = mCutOpt.addConstr(slack_demand[i] <= 0 ,name = 'demand{}'.format(i))
  constrs["demand"][i] = constr

# # # Minimize total rolls cut
obj = vcut.sum('*')-slack_demand.sum('*')
mCutOpt.setObjective(obj, COPT.MINIMIZE)
mCutOpt.solve()
#
mCutOpt.write("./mCutOpt000.lp")
reportRMP(mCutOpt)
# # Build the SUB model
# vuse = mPatGen.addVars(nkind, vtype=COPT.INTEGER, nameprefix='vuse')
# mPatGen.addConstr(vuse.prod(rollsize_dict) <= rollwidth, 'width_limit')
#
# print("               *** Column Generation Loop ***               ")
# for i in range(MAX_CGTIME):
#   print("Iteration {0}: \n".format(i))
#
#   # Solve the RMP model and report solution
#   mCutOpt.solve()
#   reportRMP(mCutOpt)
#
#   # Get the dual values of constraints
#   price = mCutOpt.getInfo(COPT.Info.Dual, constrs["c_temp"])
#
#   # Update objective function of SUB model
#   mPatGen.setObjective(1 - vuse.prod(price), COPT.MINIMIZE)
#
#   # Solve the SUB model and report solution
#   mPatGen.solve()
#   reportSUB(mPatGen)
#
#   # Test if CG iteration has converged
#   if mPatGen.objval >= -1e-6:
#     break
#
#   # Add new variable to RMP model
#   newnbr = mPatGen.getInfo(COPT.Info.Value, vuse)
#   # column的意思是cfill这个约束的变量系数是newnbr
#   cutcol = Column(constrs["c_temp"], newnbr)
#   mCutOpt.addVar(obj=1.0, name="npat({})".format(i), column=cutcol)
# print("                     *** End Loop ***                     \n")
#
# # Set all variables in RMP model to integers
# mCutOpt.setVarType(vcut, COPT.INTEGER)
#
# # Solve the MIP model and report solution
# mCutOpt.solve()
# reportMIP(mCutOpt)
#
