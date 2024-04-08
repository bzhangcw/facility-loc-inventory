######################################################################
# @date: 04/05/2023
# @author: cz
# @note: module for generating columns
#   beyond the pricing problems.
#
######################################################################
from enum import IntEnum

import numpy as np
import scipy.sparse as scisp
from gurobipy import *

import utils
from slim_cg.slim_rmp_model import *

CG_COL_INIT_METHOD = 0


def add_extra_columns(self):
    if self.iter == 0:
        # only works in the beginning
        add_initial_columns(self)
    else:
        # along-the-way methods
        add_along_columns(self)
    # after all methods invoking this
    self.update_rmp_by_cols()
    return 1


def add_initial_columns(self):
    if CG_COL_INIT_METHOD == 1:
        pass
    pass


def add_along_columns(self):
    pass
