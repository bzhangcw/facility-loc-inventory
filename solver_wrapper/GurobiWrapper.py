import gurobipy
from gurobipy import GRB

from .SolverWrapper import SolverWrapper


class GurobiWrapper(SolverWrapper):
    def __init__(self, model_name="model"):
        super().__init__(model_name, "GUROBI")
        self.model = self.create_solver_model()
        self.ENVR = None

    def create_solver_model(self):
        return gurobipy.Model(self._model_name)

    def setParam(self, param_name, param_value):
        self.model.setParam(param_name, param_value)

    def setObjective(self, expression, sense=None):
        self.model.setObjective(expression)
        if sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
            self.model.ModelSense = sense
        else:
            raise ValueError("sense is not set")

    def solve(self):
        self.model.optimize()

    def addVars(
        self, *indices, lb=None, ub=None, obj=None, vtype=None, nameprefix=None
    ):
        vars_arg_dict = {
            "lb": lb,
            "ub": ub,
            "obj": obj,
            "vtype": vtype,
            "name": nameprefix,
        }
        return self.model.addVars(*indices, **self.get_exist_args(vars_arg_dict))

    def addColumn(self, constrs=None, coeffs=None):
        return gurobipy.Column(coeffs, constrs)

    def setVarType(self, var, vtype):
        var.setAttr("VType", vtype)

    def getExpr(self, expr):
        return expr

    def getExprSize(self, expr):
        return expr.size()

    def getDuals(self):
        return self.model.getAttr("Pi", self.model.getConstrs())

    def getVarValue(self, var):
        return var.getAttr("x")

    def setEqualConstr(self, constr, value):
        constr.setAttr("RHS", value)
        constr.setAttr("Sense", GRB.EQUAL)
