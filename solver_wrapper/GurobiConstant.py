# -*- coding: utf-8 -*-
# @author: xiangjunhao
# @email: xiangjunhao@shanshu.ai
# @date: 2022/07/22

from gurobipy import GRB

from .SolverConstant import SolverConstant


class GurobiConstant(SolverConstant):
    MINIMIZE = GRB.MINIMIZE
    MAXIMIZE = GRB.MAXIMIZE

    INFINITY = GRB.INFINITY

    EQUAL = GRB.EQUAL
    LESS_EQUAL = GRB.LESS_EQUAL
    GREATER_EQUAL = GRB.GREATER_EQUAL

    BINARY = GRB.BINARY
    INTEGER = GRB.INTEGER
    CONTINUOUS = GRB.CONTINUOUS

    # SOS type constraint
    SOS_TYPE1 = GRB.SOS_TYPE1
    SOS_TYPE2 = GRB.SOS_TYPE2

    UNSTARTED = GRB.LOADED
    OPTIMAL = GRB.OPTIMAL
    INFEASIBLE = GRB.INFEASIBLE
    UNBOUNDED = GRB.UNBOUNDED
    INF_OR_UNB = GRB.INF_OR_UNBD
    NUMERICAL = GRB.NUMERIC
    NODELIMIT = GRB.NODE_LIMIT
    TIMEOUT = GRB.TIME_LIMIT
    UNFINISHED = GRB.INPROGRESS
    INTERRUPTED = GRB.INTERRUPTED
    SUBOPTIMAL = GRB.SUBOPTIMAL

    class Param(SolverConstant.Param):
        # limit and gap
        TimeLimit = "TimeLimit"
        NodeLimit = "NodeLimit"
        BarIterLimit = "BarIterLimit"
        FeasTol = "FeasibilityTol"
        DualTol = "OptimalityTol"
        IntTol = "IntFeasTol"
        RelGap = "MIPGap"

        # presolve
        Presolve = "Presolve"
        Scaling = "ScaleFlag"
        Dualize = "PreDual"

        # LP
        LpMethod = "Method"
        DualPrice = "SimplexPricing"
        BarHomogeneous = "BarHomogeneous"
        BarOrder = "BarOrder"
        Crossover = "Crossover"

        # integer
        CutLevel = "Cuts"
        RootCutRounds = "CutPasses"
        HeurLevel = "Heuristics"
        StrongBranching = "VarBranch"
        MipStartNodeLimit = "StartNodeLimit"
        SubMipNodeLimit = "SubMIPNodes"

        # parallel
        Threads = "Threads"

        # IIS
        IISMethod = "IISMethod"

        # Logging
        Logging = "LoToConsole"
        LogToConsole = "LogToConsole"

        # not listed in copt doc
        IntegralityFocus = "IntegralityFocus"

        # not supported in copt
        OutputFlag = "OutputFlag"
        LogFile = "LogFile"
        Seed = "Seed"
        MIPFocus = "MIPFocus"
        NumericFocus = "NumericFocus"
        Symmetry = "Symmetry"
        NoRelHeurTime = "NoRelHeurTime"
        ConcurrentMIP = "ConcurrentMIP"
        LazyConstraints = "LazyConstraints"
        PreCrush = "PreCrush"

    class Attr(SolverConstant.Attr):
        # model related
        Cols = "NumVars"
        Rows = "NumConstrs"
        Elems = "NumNZs"
        QElems = "NumQNZs"

        Bins = "NumBinVars"
        Ints = "NumIntVars"

        Soss = "NumSOS"
        QConstrs = "NumQConstrs"
        Indicators = "NumGenConstrs"

        ObjSense = "ModelSense"
        ObjConst = "ObjCon"
        IsMIP = "IsMIP"

        # solution related
        LpStatus = "Status"
        MipStatus = "Status"
        SimplexIter = "IterCount"
        BarrierIter = "BarIterCount"
        NodeCnt = "NodeCount"
        PoolSols = "SolCount"

        IsMinIIS = "IISMinimal"

        LpObjVal = "ObjVal"
        BestObj = "ObjVal"
        BestBnd = "ObjBound"
        BestGap = "MIPGap"

        SolvingTime = "Runtime"

        # not supported in copt
        Start = "Start"
        VarHintVal = "VarHintVal"

    class Info(SolverConstant.Info):
        # model related
        Obj = "Obj"
        UB = "UB"
        LB = "LB"

        # solution related
        Value = "X"
        Slack = "Slack"
        RedCost = "RC"

        # dual farkas and primal ray
        DualFarkas = "FarkasDual"
        PrimalRay = "FarkasProof"

        # not supported in copt
        VType = "VType"
        VarName = "VarName"
        Start = "Start"
        ConstrName = "ConstrName"
        RHS = "RHS"
        Sense = "Sense"
        Lazy = "Lazy"

    class Callback(SolverConstant.Callback):
        MIPNODE = GRB.Callback.MIPNODE
        MIPNODE_STATUS = GRB.Callback.MIPNODE_STATUS
        MIPSOL = GRB.Callback.MIPSOL
