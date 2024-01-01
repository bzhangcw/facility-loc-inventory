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
vcut = mCutOpt.addVars(ninitpat, nameprefix="vcut")

# For each width, roll cuts should meet demands
nbr = []
for i in range(nkind):
  rowdata = [0.0] * ninitpat
  rowdata[i] = int(rollwidth / rollsize[i])
  nbr.append(rowdata)

cfill = mCutOpt.addConstrs((quicksum(nbr[i][j] * vcut[j] for j in range(ninitpat)) \
                           >= rolldemand[i] for i in range(nkind)), \
                           nameprefix='cfill')

# Minimize total rolls cut
mCutOpt.setObjective(vcut.sum('*'), COPT.MINIMIZE)

# Build the SUB model
vuse = mPatGen.addVars(nkind, vtype=COPT.INTEGER, nameprefix='vuse')
mPatGen.addConstr(vuse.prod(rollsize_dict) <= rollwidth, 'width_limit')

# Main CG loop
print("               *** Column Generation Loop ***               ")
for i in range(MAX_CGTIME):
  print("Iteration {0}: \n".format(i))

  # Solve the RMP model and report solution
  mCutOpt.solve()
  reportRMP(mCutOpt)

  # Get the dual values of constraints
  price = mCutOpt.getInfo(COPT.Info.Dual, cfill)

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
  # column的意思是cfill这个约束的变量系数是newnbr
  cutcol = Column(cfill, newnbr)
  mCutOpt.addVar(obj=1.0, name="npat({})".format(i), column=cutcol)
print("                     *** End Loop ***                     \n")

# Set all variables in RMP model to integers
mCutOpt.setVarType(vcut, COPT.INTEGER)

# Solve the MIP model and report solution
mCutOpt.solve()
reportMIP(mCutOpt)
