from enum import IntEnum
import pandas as pd
from coptpy import COPT
from tqdm import tqdm

import utils

CG_SEQUENTIAL_BSIZE = 1000
CG_FIX_TOL = 1e-1


class PrimalMethod(IntEnum):
    Null = -1
    Direct = 0
    Sequential = 1

    @classmethod
    def select(cls, choice):
        if choice == cls.Null:
            return None
        if choice == cls.Direct:
            return milp_direct
        if choice == cls.Sequential:
            return milp_sequential


def milp_direct(self):
    model = self.rmp_model
    self.rmp_oracle.switch_to_milp()
    self.rmp_model.write(f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.mip.mps")
    model.optimize()
    mip_objective = model.objval
    # reset back to LP
    self.rmp_oracle.switch_to_lp()
    self._logger.info("rmp reset over")

    return mip_objective


def milp_sequential(self):
    model = self.rmp_model
    binaries = self.rmp_oracle.binaries

    self.rmp_model.setParam("LPWarmStart", 2)
    self.rmp_model.setAttr("VBasis", self.rmp_model.getVars(), self.rmp_vbasis)
    self.rmp_model.setAttr("CBasis", self.rmp_model.getConstrs(), self.rmp_cbasis)

    def _set_tob(v):
        v.setType(COPT.BINARY) if self.rmp_oracle.backend == "COPT" else v.setAttr(
            "VType", COPT.BINARY
        )

    def _set_toc(v):
        v.setType(COPT.CONTINUOUS) if self.rmp_oracle.backend == "COPT" else v.setAttr(
            "VType", COPT.CONTINUOUS
        )

    def _fix_by_v(v):
        if self.rmp_oracle.backend == "COPT":
            v.setInfo(COPT.Info.LB, sol[idx])
            v.setInfo(COPT.Info.UB, sol[idx])
        else:
            v.setAttr(COPT.Info.LB, sol[idx])
            v.setAttr(COPT.Info.UB, sol[idx])

    def _fix_by_attr(attr, v, val):
        if self.rmp_oracle.backend == "COPT":
            v.setInfo(attr, val)
        else:
            v.setAttr(attr, val)

    def _reset_binary_bound(v):
        if self.rmp_oracle.backend == "COPT":
            v.setInfo(COPT.Info.LB, 0)
            v.setInfo(COPT.Info.UB, 1)
        else:
            v.setAttr(COPT.Info.LB, 0)
            v.setAttr(COPT.Info.UB, 1)

    with utils.TimerContext(self.iter, "sequential-major-binaries"):
        for v in binaries:
            _set_tob(v)
        self.rmp_model.write(f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.mip.mps")

        self.rmp_oracle.solver.solve()
        # fix the binaries
        sol = [v.x for v in binaries]
        for idx, v in enumerate(binaries):
            _fix_by_v(v)

        # fix if sku_flow < 0
        _reset_vals = []
        for k, v in self.rmp_oracle.variables["sku_flow"].items():
            if v.x < CG_FIX_TOL:
                _reset_vals.append((v, "UB", v.UB))
                _fix_by_attr("UB", v, 0.0)

    # with utils.TimerContext(self.iter, "sequential-column-weights"):
    #     for _, v in self.rmp_oracle.variables["column_weights"].items():
    #         _set_tob(v)
    #     self.rmp_oracle.solver.solve()

    # summarizing the columns
    df = pd.DataFrame.from_records(
        [
            {"user_str": c.idx, "user": c, **col}
            for c, cols in self.columns.items()
            for col in cols
        ]
    )
    dfn = df.set_index("user_str")["user"].to_dict()
    _sorted_c = (
        df.groupby("user_str")["beta"]
        .max()
        .sort_values(ascending=False)
        .index.to_list()
    )
    _weight_c = {dfn[c]: idx for idx, c in enumerate(_sorted_c)}
    _keys_col = sorted(
        self.rmp_oracle.variables["column_weights"],
        key=lambda x: (_weight_c[x[0]], -x[1]),
    )

    def chunks(lst, n):
        """yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    ckid = 0
    for ck in tqdm(list(chunks(_keys_col, CG_SEQUENTIAL_BSIZE)), ncols=90, leave=True):
        with utils.TimerContext(f"{self.iter}@{ckid}", "sequential-column-weights"):
            cols = (self.rmp_oracle.variables["column_weights"][cc] for cc in ck)
            for v in cols:
                _set_tob(v)
            self.rmp_oracle.solver.solve()
            for v in cols:
                _fix_by_v(v)
        ckid += 1

    # reset back to LP
    self.rmp_oracle.switch_to_lp()
    for v in binaries:
        _reset_binary_bound(v)
    for _, v in self.rmp_oracle.variables["column_weights"].items():
        _reset_binary_bound(v)
    for v, attr, val in _reset_vals:
        _fix_by_attr(attr, v, val)

    mip_objective = model.objval

    return mip_objective
