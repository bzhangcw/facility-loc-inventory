import utils
from enum import IntEnum
from coptpy import COPT


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

    with utils.TimerContext(self.iter, "sequential-column-weights"):
        for _, v in self.rmp_oracle.variables["column_weights"].items():
            v.setType(COPT.BINARY)

        model.solve()

    with utils.TimerContext(self.iter, "sequential-major-binaries"):
        for v in binaries:
            v.setType(COPT.BINARY)
        # self.rmp_model.write(f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.mip.mps")

        model.solve()
        # fix the binaries
        sol = [v.x for v in binaries]
        for idx, v in enumerate(binaries):
            v.setInfo(COPT.Info.LB, sol[idx])
            v.setInfo(COPT.Info.UB, sol[idx])

    # reset back to LP
    self.rmp_oracle.switch_to_lp()
    for v in binaries:
        v.setInfo(COPT.Info.LB, 0)
        v.setInfo(COPT.Info.UB, 1)
    for _, v in self.rmp_oracle.variables["column_weights"].items():
        v.setInfo(COPT.Info.LB, 0)
        v.setInfo(COPT.Info.UB, 1)

    mip_objective = model.objval

    return mip_objective
