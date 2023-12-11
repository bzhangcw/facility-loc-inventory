#
# This file is part of the Cardinal Optimizer, all rights reserved.
#
from coptpy import *

# Cutting-stock data
rollwidth  = 115

rollsize   = [25, 40, 50, 55, 70]
rolldemand = [50, 36, 24, 8, 30]
nkind      = len(rollsize)
ninitpat   = nkind

rollsize_dict = {j: rollsize[j] for j in range(len(rollsize))}

# Maximal number of CG iterations
MAX_CGTIME = 1000

# Report solution of the RMP model
def reportRMP(rmpmodel):
  if rmpmodel.status == COPT.OPTIMAL:
    print("Using {0} rolls".format(rmpmodel.objval))

    rmpvars = rmpmodel.getVars()
    for var in rmpvars:
      if var.x > 1e-6:
        print("  {0} = {1:.6f}".format(var.name, var.x))
  else:
    print(rmpmodel.status)
    rmpvars = rmpmodel.getVars()
    for var in rmpvars:
      if var.x > 1e-6:
        print("  {0} = {1:.6f}".format(var.name, var.x))

# Report solution of the SUB model
def reportSUB(submodel):
  if submodel.status == COPT.OPTIMAL:
    print("\nPrice: {0:.6f}\n".format(submodel.objval))

# Report solution of the MIP model
def reportMIP(mipmodel):
  if mipmodel.status == COPT.OPTIMAL:
    print("Best MIP objective value: {0:.0f} rolls".format(mipmodel.objval))

    mipvars = mipmodel.getVars()
    for var in mipvars:
      if var.x > 1e-6:
        print("  {0} = {1:.0f}".format(var.name, var.x))

# Create COPT environment
env = Envr()

# Create RMP and SUB model
mCutOpt = env.createModel("mCutOpt")
mPatGen = env.createModel("mPatGen")

# Disable log information
mCutOpt.setParam(COPT.Param.Logging, 0)
mPatGen.setParam(COPT.Param.Logging, 0)

# Build the RMP model
vcut = mCutOpt.addVars(ninitpat,lb=0, nameprefix="vcut")
# vcut = mCutOpt.addVars(ninitpat,lb=0,vtype=COPT.INTEGER, nameprefix="vcut")

tem_var = mCutOpt.addVars(ninitpat, nameprefix="temp")
slack_demand = mCutOpt.addVars(ninitpat, ub=0, nameprefix="slack_demand")

# For each width, roll cuts should meet demands
nbr = []
for i in range(nkind):
  rowdata = [0.0] * ninitpat
  rowdata[i] = int(rollwidth / rollsize[i])
  nbr.append(rowdata)

constrs = {}
# add constraints
constr_types = {
            "cfill": {"index": "(i)"},
            "c_temp": {"index": "(i)"},
            "demand": {"index": "(i)"},
        }

for constr in constr_types.keys():
    constrs[constr] = dict()

for i in range(nkind):
  constr = mCutOpt.addConstr(tem_var[i] + slack_demand[i] == rolldemand[i],
                          name='cfill{}'.format(i),
                      )
  constrs["cfill"][i] = constr

for i in range(nkind):
  constr = mCutOpt.addConstr(quicksum(nbr[i][j] * vcut[j] for j in range(ninitpat)) == tem_var[i],name = 'c_temp{}'.format(i))
  constrs["c_temp"][i] = constr

for i in range(nkind):
  constr = mCutOpt.addConstr(slack_demand[i] <= 0 ,name = 'demand{}'.format(i))
  constrs["demand"][i] = constr


# # # Minimize total rolls cut
obj = vcut.sum('*')-slack_demand.sum('*')
mCutOpt.setObjective(obj, COPT.MINIMIZE)
mCutOpt.solve()
mCutOpt.write("./mCutOpt.lp")
reportRMP(mCutOpt)
# # Build the SUB model
vuse = mPatGen.addVars(nkind, vtype=COPT.INTEGER, nameprefix='vuse')
mPatGen.addConstr(vuse.prod(rollsize_dict) <= rollwidth, 'width_limit')

print("               *** Column Generation Loop ***               ")
for i in range(MAX_CGTIME):
  print("Iteration {0}: \n".format(i))

  # Solve the RMP model and report solution
  mCutOpt.solve()
  reportRMP(mCutOpt)

  # Get the dual values of constraints
  price = mCutOpt.getInfo(COPT.Info.Dual, constrs["c_temp"])

  # Update objective function of SUB model
  mPatGen.setObjective(1 - vuse.prod(price), COPT.MINIMIZE)

  # Solve the SUB model and report solution
  mPatGen.solve()
  reportSUB(mPatGen)

  # Test if CG iteration has converged
  if mPatGen.objval >= -1e-6:
    break

  # Add new variable to RMP model
  newnbr = mPatGen.getInfo(COPT.Info.Value, vuse)
  for i in range(nkind):
    print(vuse[i].x)
  # column的意思是cfill这个约束的变量系数是newnbr
  cutcol = Column(constrs["c_temp"], newnbr)
  mCutOpt.addVar(obj=1.0, name="npat({})".format(i), column=cutcol)
  mCutOpt.write("./mCutOpt{0}.lp".format(i))
print("                     *** End Loop ***                     \n")

# Set all variables in RMP model to integers
mCutOpt.setVarType(vcut, COPT.INTEGER)

# Solve the MIP model and report solution
mCutOpt.solve()
reportMIP(mCutOpt)

