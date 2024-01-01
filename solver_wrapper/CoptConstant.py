# -*- coding: utf-8 -*-
# @author: xiangjunhao
# @email: xiangjunhao@shanshu.ai
# @date: 2022/07/22

from coptpy import COPT

from .SolverConstant import SolverConstant


class CoptConstant(SolverConstant):
    # optimization direction
    MINIMIZE = COPT.MINIMIZE
    MAXIMIZE = COPT.MAXIMIZE
    INFINITY = COPT.INFINITY

    # constraint sense
    EQUAL = COPT.EQUAL
    LESS_EQUAL = COPT.LESS_EQUAL
    GREATER_EQUAL = COPT.GREATER_EQUAL
    FREE = COPT.FREE
    RANGE = COPT.RANGE

    # SOS type constraint
    SOS_TYPE1 = COPT.SOS_TYPE1
    SOS_TYPE2 = COPT.SOS_TYPE2

    # variable type
    BINARY = COPT.BINARY
    INTEGER = COPT.INTEGER
    CONTINUOUS = COPT.CONTINUOUS

    # solution status
    UNSTARTED = COPT.UNSTARTED
    OPTIMAL = COPT.OPTIMAL
    INFEASIBLE = COPT.INFEASIBLE
    UNBOUNDED = COPT.UNBOUNDED
    INF_OR_UNB = COPT.INF_OR_UNB
    NUMERICAL = COPT.NUMERICAL
    NODELIMIT = COPT.NODELIMIT
    TIMEOUT = COPT.TIMEOUT
    UNFINISHED = COPT.UNFINISHED
    INTERRUPTED = COPT.INTERRUPTED

    class Param(SolverConstant.Param):
        # limit and gap
        TimeLimit = 'TimeLimit'
        NodeLimit = 'NodeLimit'
        BarIterLimit = 'BarIterLimit'
        MatrixTol = 'MatrixTol'
        FeasTol = 'FeasTol'
        DualTol = 'DualTol'
        IntTol = 'IntTol'
        RelGap = 'RelGap'

        # presolve
        Presolve = 'Presolve'
        Scaling = 'Scaling'
        Dualize = 'Dualize'

        # LP
        LpMethod = 'LpMethod'
        DualPrice = 'DualPrice'
        DualPerturb = 'DualPerturb'
        BarHomogeneous = 'BarHomogeneous'
        BarOrder = 'BarOrder'
        Crossover = 'Crossover'
        ReqFarkasRay = 'ReqFarkasRay'

        # semidefinite
        SDPMethod = 'SDPMethod'

        # integer
        CutLevel = 'CutLevel'
        RootCutLevel = 'RootCutLevel'
        TreeCutLevel = 'TreeCutLevel'
        RootCutRounds = 'RootCutRounds'
        NodeCutRounds = 'NodeCutRounds'
        HeurLevel = 'HeurLevel'
        RoundingHeurLevel = 'RoundingHeurLevel'
        DivingHeurLevel = 'DivingHeurLevel'
        SubMipHeurLevel = 'SubMipHeurLevel'
        StrongBranching = 'StrongBranching'
        ConflictAnalysis = 'ConflictAnalysis'
        MipStartMode = 'MipStartMode'
        MipStartNodeLimit = 'MipStartNodeLimit'

        # parallel
        Threads = 'Threads'
        BarThreads = 'BarThreads'
        SimplexThreads = 'SimplexThreads'
        CrossoverThreads = 'CrossoverThreads'
        MipTasks = 'MipTasks'

        # IIS
        IISMethod = 'IISMethod'

        # relax feasibility
        FeasRelaxMode = 'FeasRelaxMode'

        # Logging
        Logging = 'Logging'
        LogToConsole = 'LogToConsole'

        # not listed in copt doc
        MipPreMethod = 'MipPreMethod'
        MipLogLevel = "MipLogLevel"
        IntegralityFocus = "NumericFocus"
        MipPreAnalyticCenter = "MipPreAnalyticCenter"
        SubMipNodeLimit = 'SubMipNodeLimit'

        # copt heuristics switch
        HeurLocks = 'HeurLocks'
        HeurSAP = 'HeurSAP'
        HeurClique = 'HeurClique'
        HeurVBound = 'HeurVBound'
        HeurFP = 'HeurFP'
        HeurRAP = 'HeurRAP'
        HeurLineSearch = 'HeurLineSearch'
        HeurConfDive = 'HeurConfDive'
        HeurCoefDive = 'HeurCoefDive'
        HeurFarkas = 'HeurFarkas'
        HeurFracDive = 'HeurFracDive'
        HeurVecLen = 'HeurVecLen'
        HeurActiveConstr = 'HeurActiveConstr'
        HeurPsCost = 'HeurPsCost'
        HeurGuidedDive = 'HeurGuidedDive'
        HeurRENS = 'HeurRENS'
        HeurRINS = 'HeurRINS'
        HeurDINS = 'HeurDINS'
        HeurCrossover = 'HeurCrossover'
        HeurMutation = 'HeurMutation'
        HeurLpFace = 'HeurLpFace'
        HeurLB = 'HeurLB'
        HeurNoObj = 'HeurNoObj'
        HeurNoObjTasks = 'HeurNoObjTasks'
        HeurNewObj = 'HeurNewObj'
        HeurOneOpt = 'HeurOneOpt'
        HeurLpImprv = 'HeurLpImprv'
        HeurPacking = 'HeurPacking'
        HeurRENSNewObj = 'HeurRENSNewObj'
        SubMipHeur26 = 'SubMipHeur26'
        ParallelHeurTasks = "ParallelHeurTasks"
        MipObjScale = "MipObjScale"
        MipPreComponents = "MipPreComponents"
        Heur26 = "Heur26"
        Heur43 = "Heur43"
        SubMipHeur15 = "SubMipHeur15"
        SubMipHeur43 = "SubMipHeur43"

    class Attr(SolverConstant.Attr):
        # model related
        Cols = 'Cols'
        PSDCols = 'PsdCols'
        Rows = 'Rows'
        Elems = 'Elems'
        QElems = 'QElems'
        PSDElems = 'PsdElems'
        SymMats = 'SymMats'

        Bins = 'Bins'
        Ints = 'Ints'

        Soss = 'Soss'
        Cones = 'Cones'
        QConstrs = 'QConstrs'
        PSDConstrs = 'PsdConstrs'
        Indicators = 'Indicators'

        ObjSense = 'ObjSense'
        ObjConst = 'ObjConst'
        HasQObj = 'HasQObj'
        HasPSDObj = 'HasPsdObj'
        IsMIP = 'IsMIP'

        # solution related''
        LpStatus = 'LpStatus'
        MipStatus = 'MipStatus'
        SimplexIter = 'SimplexIter'
        BarrierIter = 'BarrierIter'
        NodeCnt = 'NodeCnt'
        PoolSols = 'PoolSols'

        HasLpSol = 'HasLpSol'
        HasBasis = 'HasBasis'
        HasDualFarkas = 'HasDualFarkas'
        HasPrimalRay = 'HasPrimalRay'
        HasMipSol = 'HasMipSol'

        IISCols = 'IISCols'
        IISRows = 'IISRows'
        IISSOSs = 'IISSOSs'
        IISIndicators = 'IISIndicators'
        HasIIS = 'HasIIS'
        HasFeasRelaxSol = 'HasFeasRelaxSol'
        IsMinIIS = 'IsMinIIS'

        LpObjVal = 'LpObjVal'
        BestObj = 'BestObj'
        BestBnd = 'BestBnd'
        BestGap = 'BestGap'
        FeasRelaxObj = 'FeasRelaxObj'

        SolvingTime = 'SolvingTime'

    class Info(SolverConstant.Info):
        # model related
        Obj = 'Obj'
        UB = 'UB'
        LB = 'LB'

        # solution related
        Value = 'Value'
        Slack = 'Slack'
        Dual = 'Dual'
        RedCost = 'RedCost'

        # dual farkas and primal ray
        DualFarkas = 'DualFarkas'
        PrimalRay = 'PrimalRay'

        # relax feasibility
        RelaxLB = 'RelaxLB'
        RelaxUB = 'RelaxUB'
