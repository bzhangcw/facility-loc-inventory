from abc import abstractmethod


class SolverWrapper:
    def __init__(self, model_name="model", solver_name="COPT"):
        self._model_name = model_name
        self._solver_name = solver_name.upper()
        self.model = self.create_solver_model()

    @abstractmethod
    def create_solver_model(self):
        pass

    @abstractmethod
    def setParam(self, param_name, param_value):
        pass

    @abstractmethod
    def setObjective(self, expression, sense=None):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def addVars(
        self, *indices, lb=None, ub=None, obj=None, vtype=None, nameprefix=None
    ):
        pass

    def getObjective(self):
        return self.model.getObjective()

    def addVar(self, lb=None, ub=None, obj=None, vtype=None, name=None):
        fixed_name = name if isinstance(name, str) else None
        var_arg_dict = {
            "lb": lb,
            "ub": ub,
            "obj": obj,
            "vtype": vtype,
            "name": fixed_name,
        }
        return self.model.addVar(**self.get_exist_args(var_arg_dict))

    def addConstr(self, lhs, sense=None, rhs=None, name=None):
        if isinstance(name, str):
            return self.model.addConstr(lhs, sense=sense, rhs=rhs, name=name)
        return self.model.addConstr(lhs, sense=sense, rhs=rhs)

    def remove(self, items):
        self.model.remove(items)

    def computeIIS(self):
        self.model.computeIIS()

    def write(self, file_name=""):
        self.model.write(file_name)

    def addColumn(self, constrs=None, coeffs=None):
        pass

    def setVarType(self, var, vtype):
        pass

    def getExpr(self, expr):
        pass

    def getExprSize(self, expr):
        pass

    def getDuals(self):
        pass

    def getVarValue(self, var):
        pass

    def setEqualConstr(self, constr, value):
        pass

    @property
    def status(self):
        return self.model.status

    @staticmethod
    def get_exist_args(arg_dict):
        """
        get rid of all the arguments which is None.
        :param arg_dict: use dict to store all the arguments' name and value
        :return:
        """
        kargs = dict()
        for arg_name, arg_value in arg_dict.items():
            if arg_value is not None:
                kargs[arg_name] = arg_value
        return kargs
