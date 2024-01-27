import utils
from enum import IntEnum


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
    self.rmp_oracle.switch_to_milp()
    self.rmp_model.write(f"{utils.CONF.DEFAULT_SOL_PATH}/rmp@{self.iter}.mip.mps")
    model.solve()
    mip_objective = model.objval
    # reset back to LP
    self.rmp_oracle.switch_to_lp()
    self._logger.info("rmp reset over")

    return mip_objective
