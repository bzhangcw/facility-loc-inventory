import coptpy

from coptpy import COPT

from .SolverWrapper import SolverWrapper


class CoptWrapper(SolverWrapper):
    model = None
    ENVR = None

    def __init__(self, model_name="model"):
        print("use COPT to build and solve model")
        super().__init__(model_name, "COPT")
        self.model = self.create_solver_model()

    def create_solver_model(self):
        if CoptWrapper.ENVR is None:
            CoptWrapper.ENVR = coptpy.Envr()
        return CoptWrapper.ENVR.createModel(self._model_name)

    def setParam(self, param_name, param_value):
        self.model.setParam(param_name, param_value)

    def setObjective(self, expression, sense=None):
        self.model.setObjective(expression)
        if sense in (COPT.MAXIMIZE, COPT.MINIMIZE):
            self.model.ObjSense = sense
        else:
            raise ValueError("sense is not set")

    def solve(self):
        self.model.solve()

    def addVars(
        self, *indices, lb=None, ub=None, obj=None, vtype=None, nameprefix=None
    ):
        vars_arg_dict = {
            "lb": lb,
            "ub": ub,
            "obj": obj,
            "vtype": vtype,
            "nameprefix": nameprefix,
        }
        return self.model.addVars(*indices, **self.get_exist_args(vars_arg_dict))

    def addColumn(self, constrs=None, coeffs=None):
        return coptpy.Column(constrs, coeffs)

    def setVarType(self, var, vtype):
        var.setType(vtype)

    def getExpr(self, expr):
        return expr.getExpr()

    def getExprSize(self, expr):
        return expr.getSize()

    def getDuals(self):
        return self.model.getDuals()

    def getVarValue(self, var):
        return var.value
