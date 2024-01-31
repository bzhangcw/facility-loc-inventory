from enum import IntEnum
from slim_cg.slim_rmp_model import *

CG_RMP_USE_WS = int(os.environ.get("CG_RMP_USE_WS", 1))
CG_RMP_WS_OPTION = int(os.environ.get("CG_RMP_WS_OPTION", 1))
CG_RMP_METHOD = int(os.environ.get("CG_RMP_METHOD", 0))


class RMPAlg(IntEnum):
    Direct = 0
    ALM = 1


class ALMConf(object):
    max_iter = 100
    max_inner_iter = 10


_alm_default_conf = ALMConf()


def solve(self):
    if CG_RMP_METHOD == RMPAlg.Direct:
        solve_direct(self)
    elif CG_RMP_METHOD == RMPAlg.ALM:
        solve_alm(self)


def update(self):
    if CG_RMP_METHOD == RMPAlg.Direct:
        update_direct(self)
    elif CG_RMP_METHOD == RMPAlg.ALM:
        update_alm(self)


def solve_direct(self):
    if CG_RMP_USE_WS and self.iter > 0:
        if CG_RMP_WS_OPTION == 0:
            _ws_v = self.rmp_vbasis_init
            _ws_c = self.rmp_cbasis_init
        elif CG_RMP_WS_OPTION == 1:
            _ws_v = self.rmp_vbasis
            _ws_c = self.rmp_cbasis
        else:
            raise ValueError("unrecognized option")

        n_new_rmp_vars = len(self.rmp_model.getVars()) - len(_ws_v)
        n_new_rmp_cons = len(self.rmp_model.getConstrs()) - len(_ws_c)
        self._logger.info(
            "using option {}: add {} new vars and {} new cons".format(
                CG_RMP_WS_OPTION, n_new_rmp_vars, n_new_rmp_cons
            )
        )
        _ws_v = [*_ws_v, *[GRB.NONBASIC_LOWER] * n_new_rmp_vars]
        _ws_c = [*_ws_c, *[GRB.NONBASIC_LOWER] * n_new_rmp_cons]

        # set the modified basis status back to the model
        self.rmp_model.setAttr("VBasis", self.rmp_model.getVars(), _ws_v)
        self.rmp_model.setAttr("CBasis", self.rmp_model.getConstrs(), _ws_c)
        self.rmp_model.write(f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.bas")

    self.rmp_model.setParam(self.solver_constant.Param.LpMethod, 4)

    self.solver.solve()

    if CG_RMP_USE_WS:
        self.rmp_model.setParam("LPWarmStart", 2)
        # Get the variable basis status
        self.rmp_vbasis = self.rmp_model.getAttr("VBasis", self.rmp_model.getVars())

        # Get the constraint basis status
        self.rmp_cbasis = self.rmp_model.getAttr("CBasis", self.rmp_model.getConstrs())
        if self.iter == 0:
            # save the initial basis.
            self.rmp_vbasis_init = self.rmp_model.getAttr(
                "VBasis", self.rmp_model.getVars()
            )
            self.rmp_cbasis_init = self.rmp_model.getAttr(
                "CBasis", self.rmp_model.getConstrs()
            )


def update_coefficients(self):

    self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
    self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
    self.ws_cons_idx = {customer: 0 for customer in self.customer_list}
    for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
        edges = self.rmp_oracle.cg_downstream[node]
        for ee in edges:
            c = ee.end
            self.delievery_cons_idx[c].append(v)

    for c, v in self.rmp_oracle.cg_binding_constrs_ws.items():
        self.ws_cons_idx[c] = v
        self.solver.setEqualConstr(v, 1.0)

    for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
        edges = self.rmp_oracle.cg_downstream[node]
        for ee in edges:
            c = ee.end
            this_col = self.columns[c][-1]
            _val = -this_col["sku_flow"].get((t, ee, k), 0)
            self.delievery_cons_coef[c].append(_val if abs(_val) > 1e-4 else 0)


def update_direct(self):
    """
    update the RMP with new columns
    """
    update_coefficients(self)

    for c in self.customer_list:
        _cons_idx = [*self.delievery_cons_idx[c], self.ws_cons_idx[c]]
        _cons_coef = [*self.delievery_cons_coef[c], 1]
        col_idxs = list(zip(_cons_idx, _cons_coef))
        try:
            new_col = self.solver.addColumn(_cons_idx, _cons_coef)
            self.rmp_oracle.variables["column_weights"][
                c, self.columns[c].__len__() - 1
            ] = self.rmp_model.addVar(
                obj=self.columns[c][-1]["beta"],
                name=f"lambda_{c.idx}_{len(self.columns[c])}",
                lb=0.0,
                ub=1.0,
                column=new_col,
            )
        except Exception as e:
            print(f"failed at {c}\n\t{col_idxs}")
            raise e


def update_alm(self):
    """
    update the RMP with new columns
    """
    update_coefficients(self)


def solve_alm(self):
    # in this case, you never add columns to RMP,
    #   the RMP LP block always allows the same size
    # we run a block coordinate descent method here,
    # see: WZPGW: A Greedy Augmented Lagrangian Method for Block-Structured Integer Programming
    # 1 + |C| blocks,
    # -  (y, z) \in RMP oracle (By LP to the exact sense)
    # -  z + Σ Xc λc
    #       each λ_c in a unit simplex + positive orthant
    # L+ = c'y + f'z + β'z - Σ β'Xc λc + ρ/2 |z - Xc λc|^2

    pass
