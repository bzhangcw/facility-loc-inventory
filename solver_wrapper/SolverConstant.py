# -*- coding: utf-8 -*-
# @author: xiangjunhao
# @email: xiangjunhao@shanshu.ai
# @date: 2022/07/22


class SolverConstant:
    # optimization direction
    MINIMIZE = None  # used to define the sense of model objective to be minimize;
    MAXIMIZE = None  # used to define the sense of model objective to be maximize;

    INFINITY = None  # used to define what is infinity;

    # constraint sense
    EQUAL = None
    LESS_EQUAL = None
    GREATER_EQUAL = None
    FREE = None
    RANGE = None

    # variable type
    BINARY = None  # used to define variable as a binary var;
    INTEGER = None  # used to define variable as a INTEGER var;
    CONTINUOUS = None  # used to define variable as a CONTINUOUS var;

    # SOS type constraint
    SOS_TYPE1 = None
    SOS_TYPE2 = None

    # solution status
    UNSTARTED = None
    OPTIMAL = None
    INFEASIBLE = None
    UNBOUNDED = None
    INF_OR_UNB = None
    NUMERICAL = None
    NODELIMIT = None
    TIMEOUT = None
    UNFINISHED = None
    INTERRUPTED = None
    SUBOPTIMAL = None

    class Param:
        # limit and gap
        TimeLimit = None
        NodeLimit = None
        BarIterLimit = None
        MatrixTol = None
        FeasTol = None
        DualTol = None
        IntTol = None
        RelGap = None

        # presolve
        Presolve = None
        Scaling = None
        Dualize = None

        # LP
        LpMethod = None
        DualPrice = None
        DualPerturb = None
        BarHomogeneous = None
        BarOrder = None
        Crossover = None
        ReqFarkasRay = None

        # semidefinite
        SDPMethod = None

        # integer
        CutLevel = None
        RootCutLevel = None
        TreeCutLevel = None
        RootCutRounds = None
        NodeCutRounds = None
        HeurLevel = None
        RoundingHeurLevel = None
        DivingHeurLevel = None
        SubMipHeurLevel = None
        StrongBranching = None
        ConflictAnalysis = None
        MipStartMode = None
        MipStartNodeLimit = None

        # parallel
        Threads = None
        BarThreads = None
        SimplexThreads = None
        CrossoverThreads = None
        MipTasks = None

        # IIS
        IISMethod = None

        # relax feasibility
        FeasRelaxMode = None

        # Logging
        Logging = None
        LogToConsole = None

        # not listed in copt doc
        MipPreMethod = None
        MipLogLevel = None
        IntegralityFocus = None
        MipPreAnalyticCenter = None
        SubMipNodeLimit = None

        # copt heuristics switch
        HeurLocks = None
        HeurSAP = None
        HeurClique = None
        HeurVBound = None
        HeurVBound = None
        HeurVBound = None
        HeurVBound = None
        HeurVBound = None
        HeurVBound = None
        HeurFP = None
        HeurRAP = None
        HeurLineSearch = None
        HeurConfDive = None
        HeurCoefDive = None
        HeurFarkas = None
        HeurFracDive = None
        HeurVecLen = None
        HeurActiveConstr = None
        HeurPsCost = None
        HeurGuidedDive = None
        HeurRENS = None
        HeurRINS = None
        HeurDINS = None
        HeurCrossover = None
        HeurMutation = None
        HeurLpFace = None
        HeurLB = None
        HeurNoObj = None
        HeurNoObjTasks = None
        HeurNewObj = None
        HeurOneOpt = None
        HeurLpImprv = None
        HeurPacking = None
        HeurRENSNewObj = None
        SubMipHeur26 = None
        ParallelHeurTasks = None
        MipObjScale = None
        MipPreComponents = None
        Heur26 = None
        Heur43 = None
        SubMipHeur15 = None
        SubMipHeur43 = None

        # not supported in copt
        OutputFlag = None
        LogFile = None
        Seed = None
        MIPFocus = None
        NumericFocus = None
        Symmetry = None
        NoRelHeurTime = None
        ConcurrentMIP = None
        LazyConstraints = None
        PreCrush = None

    class Attr:
        # model related
        Cols = None
        PSDCols = None
        Rows = None
        Elems = None
        QElems = None
        PSDElems = None
        SymMats = None

        Bins = None
        Ints = None

        Soss = None
        Cones = None
        QConstrs = None
        PSDConstrs = None
        Indicators = None

        ObjSense = None
        ObjConst = None
        HasQObj = None
        HasPSDObj = None
        IsMIP = None

        # solution related
        LpStatus = None
        MipStatus = None
        SimplexIter = None
        BarrierIter = None
        NodeCnt = None
        PoolSols = None

        HasLpSol = None
        HasBasis = None
        HasDualFarkas = None
        HasPrimalRay = None
        HasMipSol = None

        IISCols = None
        IISRows = None
        IISSOSs = None
        IISIndicators = None
        HasIIS = None
        HasFeasRelaxSol = None
        IsMinIIS = None

        LpObjVal = None
        BestObj = None
        BestBnd = None
        BestGap = None
        FeasRelaxObj = None

        SolvingTime = None

        # not supported in copt
        Start = None
        VarHintVal = None

    class Info:
        # model related
        Obj = None
        UB = None
        LB = None

        # solution related
        Value = None
        Slack = None
        Dual = None
        RedCost = None

        # dual farkas and primal ray
        DualFarkas = None
        PrimalRay = None

        # relax feasibility
        RelaxLB = None
        RelaxUB = None

        # not supported in copt
        VType = None
        VarName = None
        Start = None
        ConstrName = None
        RHS = None
        Sense = None
        Lazy = None

    class Callback:
        MIPNODE = None
        MIPNODE_STATUS = None
        MIPSOL = None
