from enum import IntEnum

from coptpy import COPT

import utils


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
    model.solve()
    mip_objective = model.objval
    # reset back to LP
    self.rmp_oracle.switch_to_lp()
    self._logger.info("rmp reset over")

    return mip_objective


def milp_sequential(self):
    model = self.rmp_model
    binaries = self.rmp_oracle.binaries

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

        with utils.TimerContext(self.iter, "sequential-column-weights"):
            for _, v in self.rmp_oracle.variables["column_weights"].items():
                _set_tob(v)

            self.rmp_oracle.solver.solve()
    # reset back to LP
    self.rmp_oracle.switch_to_lp()
    for v in binaries:
        _reset_binary_bound(v)
    for _, v in self.rmp_oracle.variables["column_weights"].items():
        _reset_binary_bound(v)

    mip_objective = model.objval

    return mip_objective
