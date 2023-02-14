import argparse
import os
import time
import functools
import collections
import pandas as pd


class AlgParams(object):
    parser = argparse.ArgumentParser(
        "SINO Facility location", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--phase1_relax", type=int, default=0)
    parser.add_argument(
        "--phase1_resolve",
        type=int,
        default=0,
        help="""
        1 then build and solve phase I.
        0 (o.w.) read from local.
        """,
    )
    parser.add_argument(
        "--phase2_use_full_model",
        type=int,
        default=2,
        help="""
        0 if we use the reduced model to preserve sparsity in phase-II.
        1 (o.w.), we use X_W, X_C to build the model.
        2 (o.w.), we use a greedy algorithm to build the routes
        """,
    )
    parser.add_argument(
        "--phase2_use_qty_heur",
        type=int,
        default=1,
        help="""
        1 if we apply a greedy heuristic to fix production qty x s.t.
            x >= min_prod.
        0 we apply the disjunctive constraint (exact approach using an indicator variable)
        """,
    )
    parser.add_argument(
        "--phase2_qty_heur_reset",
        type=int,
        default=0,
        help="""
            once we apply a greedy heuristic to fix production qty x s.t.
                x >= min_prod.
            In theory dual simplex should work well, so we have two strategy:
            0 we do not reset model and hot start
            1 reset and presolve and solve from scratch.
        """,
    )
    parser.add_argument(
        "--timelimit",
        type=int,
        default=1200,
        help="""
            linear programming time limit
        """,
    )
    parser.add_argument(
        "--phase1_lpmethod",
        type=int,
        default=1,
        help="""
            phase I linear programming method, default dual simplex
        """,
    )
    parser.add_argument(
        "--phase2_lpmethod",
        type=int,
        default=2,
        help="""
            phase II linear programming method, default ipm
        """,
    )
    parser.add_argument(
        "--phase2_greedy_range",
        type=int,
        default=5,
        help="""
            see covering.py
        """,
    )

    __slots__ = (
        "phase1_relax",
        "phase1_resolve",
        "phase2_use_full_model",
        "phase2_use_qty_heur",
        "phase2_qty_heur_reset",
        "phase2_greedy_range",
        "phase1_lpmethod",
        "phase2_lpmethod",
        "timelimit",
    )

    def __init__(self):
        args = self.parser.parse_args()
        self.phase2_greedy_range = 5  # todo, unparameterized
        for i in self.__slots__:
            self.__setattr__(i, args.__getattribute__(i))

    def show(self):
        print("--- ALG PARAMS ---")
        for i in self.__slots__:
            print(f"- {i}:", self.__getattribute__(i))


# singleton
DEFAULT_ALG_PARAMS = AlgParams()

GLOBAL_PROFILE = {
    "count": collections.defaultdict(int),
    "total": collections.defaultdict(float),
}


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        cc = func(*args, **kwargs)
        et = time.time()
        GLOBAL_PROFILE["total"][func.__qualname__] += et - st
        GLOBAL_PROFILE["count"][func.__qualname__] += 1
        print(f"func finished eps: {et - st:5.2f} @{func.__qualname__} ")
        return cc

    return wrapper


def show_profiling():
    print("|--- PROFILING STATS ---")
    stats = pd.DataFrame.from_dict(GLOBAL_PROFILE)
    stats["avg"] = stats["total"] / stats["count"]
    stats = stats.sort_values(by="total", ascending=False)
    print(stats.to_markdown())
