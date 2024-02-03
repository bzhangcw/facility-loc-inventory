from enum import IntEnum
from gurobipy import *
import numpy as np
import scipy.sparse as scisp
from slim_cg.slim_rmp_model import *

CG_RMP_USE_WS = int(os.environ.get("CG_RMP_USE_WS", 1))
CG_RMP_WS_OPTION = int(os.environ.get("CG_RMP_WS_OPTION", 1))
CG_RMP_METHOD = int(os.environ.get("CG_RMP_METHOD", 1))


class RMPAlg(IntEnum):
    Direct = 0
    ALM = 1


class ALMConf(object):
    max_iter = 100
    max_inner_iter = 10


_alm_default_conf = ALMConf()


def mute(*args, bool_unmute=False):
    for v in args:
        v.Params.OutputFlag = bool_unmute


def cleanup(self):
    vv = self.rmp_oracle.variables.get("cg_temporary")
    if vv is not None:
        self.rmp_model.remove(vv)
        self.rmp_oracle.variables["cg_temporary"] = None
        self._logger.info(f"removed initial skeleton")

    if CG_RMP_METHOD == RMPAlg.ALM:
        # remove cg bindings constraints
        self._logger.info(f"remove the binding blocks in RMP LP blocks")
        for k, v in self.rmp_oracle.cg_binding_constrs.items():
            self.rmp_model.remove(v)

    self.bool_rmp_update_initialized = True
    return None


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


def update_coefficients(self):
    self.delievery_cons_coef = {customer: [] for customer in self.customer_list}
    self.delievery_cons_rows = {customer: [] for customer in self.customer_list}

    for row, (node, k, t) in enumerate(self.rmp_oracle.cg_binding_constrs_keys):
        edges = self.rmp_oracle.cg_downstream[node]
        for ee in edges:
            c = ee.end
            this_col = self.columns[c][-1]
            _val = -this_col["sku_flow"].get((t, ee, k), 0)
            self.delievery_cons_coef[c].append(_val if abs(_val) > 1e-4 else 0)
            self.delievery_cons_rows[c].append(row)


def fetch_dual_info(self):
    if CG_RMP_METHOD == RMPAlg.Direct:
        return fetch_dual_info_direct(self.rmp_oracle)
    elif CG_RMP_METHOD == RMPAlg.ALM:
        return fetch_dual_info_alm(self)


#############################################################################################
# direct method
#############################################################################################
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
    self.rmp_objval = self.rmp_model.objval
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


def update_direct(self):
    """
    update the RMP with new columns
    """
    update_coefficients(self)

    self.delievery_cons_idx = {customer: [] for customer in self.customer_list}
    self.ws_cons_idx = {customer: 0 for customer in self.customer_list}

    for (node, k, t), v in self.rmp_oracle.cg_binding_constrs.items():
        edges = self.rmp_oracle.cg_downstream[node]
        for ee in edges:
            c = ee.end
            self.delievery_cons_idx[c].append(v)

    for c, v in self.rmp_oracle.cg_binding_constrs_ws.items():
        self.ws_cons_idx[c] = v
        self.solver.setEqualConstr(v, 1.0)

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


def fetch_dual_info_direct(lp_oracle):
    if lp_oracle.arg.backend.upper() == "COPT":
        node_dual = {
            k: v.pi if v is not None else 0
            for k, v in lp_oracle.cg_binding_constrs.items()
        }
        # the dual of weight sum = 1
        ws_dual = {
            k: v.pi if v is not None else 0
            for k, v in lp_oracle.cg_binding_constrs_ws.items()
        }
    else:
        node_dual = {
            k: v.getAttr(GRB.Attr.Pi) if v is not None else 0
            for k, v in lp_oracle.cg_binding_constrs.items()
        }
        # the dual of weight sum = 1
        ws_dual = {
            k: v.getAttr(GRB.Attr.Pi) if v is not None else 0
            for k, v in lp_oracle.cg_binding_constrs_ws.items()
        }

    node_vals = np.array(list(node_dual.values()))

    for cc, ccols in lp_oracle.dual_cols.items():
        lp_oracle.dual_vals[cc] = lp_oracle.broadcast_matrix[ccols, :] @ node_vals
    return (
        node_vals,
        ws_dual,
        set(lp_oracle.dual_keys.keys()),
    )


#############################################################################################
# alm
#############################################################################################
def update_alm(self):
    """
    update the RMP with new columns
    """
    update_coefficients(self)
    if self.iter == 0:
        self.X = []
        self.beta = []
        self.nu = {c: 0 for c in self.customer_list}
        self.m = self.rmp_oracle.cg_binding_constrs_keys.__len__()
        self.eta = np.zeros((self.m, 1))
        self.z = [
            self.rmp_oracle.variables["sku_delivery"][t, node, k]
            for node, k, t in
            self.rmp_oracle.cg_binding_constrs
        ]
        self.f = [v.obj for v in self.z]
        # keep a k-1 th iterate.
        self.etaz = np.zeros((self.m, 1))
        self.lmbd = None
        self.lmbd = None

    for idb, c in enumerate(self.customer_list):
        _newcol = np.zeros(self.m)
        _newcol[self.delievery_cons_rows[c]] = self.delievery_cons_coef[c]
        if self.iter == 0:
            self.X.append(scisp.csr_matrix(-_newcol))
            self.beta.append([self.columns[c][-1]["beta"]])
        else:
            self.X[idb] = scisp.vstack([self.X[idb], -_newcol])
            self.beta[idb].append(self.columns[c][-1]["beta"])


def solve_alm(self):
    # in this case, you never add columns to RMP,
    #   the RMP LP block always allows the same size
    # we run a block coordinate descent method here,
    # see: WZPGW: A Greedy Augmented Lagrangian Method for Block-Structured Integer Programming
    # 1 + |C| blocks,
    # -  (y, z) \in RMP oracle (By LP to the exact sense)
    # -  z + Σ Xc λc
    #       each λ_c in a unit simplex + positive orthant
    # take:
    # Xl = Σ Xc λc
    # L+ = c'y + f'z + β'(z - Σ Xc λc) + ρ/2 |z - Σ Xc λc|^2
    #    = c'y + (f + β - ρXl)'z + ρ/2 |z|^2 +
    # basic
    _card = self.customer_list.__len__()
    lp_model = self.rmp_model
    lp_oracle = self.rmp_oracle

    # alias

    X = self.X
    z = self.z
    f = np.array(self.f)
    eta = self.eta.copy().flatten()
    beta = [np.array(bb) for bb in self.beta]

    #
    k = 0
    p = 2e1
    clst = list(range(_card))
    if self.lmbd is not None:
        lmbd = [np.resize(self.lmbd, X[ibd].shape[0]) for ibd in clst]
    else:
        lmbd = [np.ones(X[ibd].shape[0]) / X[ibd].shape[0] for ibd in clst]

    self.qpmodel = Model("qp")
    self.var_lmbda = {}
    self.con_lmbda = {}
    for idb in clst:
        Xc = X[idb]
        self.var_lmbda[idb] = self.qpmodel.addMVar(Xc.shape[0], ub=1.0)
        self.con_lmbda[idb] = self.qpmodel.addConstr(self.var_lmbda[idb].sum() == 1)

    _var_Xl = [self.var_lmbda[idb] @ X[idb] for idb in clst]
    _var_Xls = sum(_var_Xl)
    _Xl = [lmbd[idb] @ X[idb] for idb in clst]
    _Xls = sum(_Xl)
    _sumquadz = quicksum(v * v for v in z)

    mute(lp_model, self.qpmodel)
    # mute(self.qpmodel)
    while k <= _alm_default_conf.max_iter:

        # linear optimization oracle
        var_feas_z = z - _Xls

        zobj = eta @ var_feas_z + p / 2 * (_sumquadz + _Xls @ _Xls) - p * (z @ _Xls)
        # zlobj = LinExpr(eta, z)
        # zqobj =
        lp_oracle.solver.setObjective(
            lp_oracle.original_obj + zobj, sense=self.solver_constant.MINIMIZE
        )
        lp_model.optimize()
        lp_model.setParam("LPWarmStart", 2)

        zk = np.array(lp_model.getAttr("x", z))
        _feas = zk - _Xls

        var_feas = zk - _var_Xls

        _obj_quad = p / 2 * (zk @ zk + _var_Xls @ _var_Xls) - p * (_var_Xls @ zk)
        _obj_lin = sum(
            (beta[idb] - X[idb] @ eta) @ self.var_lmbda[idb] for idb in clst
        )
        self.qpmodel.setObjective(_obj_quad + _obj_lin)
        self.qpmodel.optimize()

        lmbd = [self.var_lmbda[ibd].x for ibd in clst]

        # update
        _Xl = [lmbd[idb] @ X[idb] for idb in clst]
        _Xls = sum(_Xl)
        _feas = zk - _Xls
        _eps_feas = np.linalg.norm(_feas)
        _eps_feas_rel = _eps_feas / (sum(_Xls) + 1e-1)

        fk = (
                lp_oracle.original_obj.getValue()
                + eta @ _feas
                + _feas @ _feas * p / 2
                + sum(beta[idb] @ lmbd[idb] for idb in clst)
        )
        print(f"-- k: {k}, |df|/ε: {_eps_feas: .1e}/{_eps_feas_rel: .1e}, f: {fk: .6e}")
        if _eps_feas_rel < 1e-6:
            break

        eta += p * _feas
        p *= 1.02
        k += 1

    self.rmp_objval = fk
    self.eta_ws = {c: self.con_lmbda[ibd].pi.tolist() for ibd, c in enumerate(self.customer_list)}
    self.etaz = self.eta
    self.eta = eta
    self.zk = zk
    self.lmbd = lmbd


def fetch_dual_info_alm(self):
    lp_oracle = self.rmp_oracle
    for cc, ccols in lp_oracle.dual_cols.items():
        lp_oracle.dual_vals[cc] = lp_oracle.broadcast_matrix[ccols, :] @ self.eta
    return (
        self.eta,
        self.eta_ws,
        set(lp_oracle.dual_keys.keys()),
    )
