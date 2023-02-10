import argparse
import os
import time
import functools
import collections
import pandas as pd


class AlgParams(object):
    parser = argparse.ArgumentParser(
        "SINO Facility location",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--phase1_relax", type=int, default=0)
    parser.add_argument("--phase1_resolve", type=int, default=1, help="""
        1 then build and solve phase I.
        0 (o.w.) read from local.
    """)
    parser.add_argument("--phase2_use_full_model", type=int, default=0, help="""
        0 if we use the reduced model to preserve sparsity in phase-II.
        1 (o.w.), we use X_W, X_C to build the model.
        2 (o.w.), we use a greedy algorithm to build the routes
    """)
    parser.add_argument("--phase2_use_qty_heur", type=int, default=1, help="""
        1 if we apply a greedy heuristic to fix production qty x s.t.
            x >= min_prod.
        0 we apply the disjunctive constraint (exact approach using an indicator variable)
    """)
    parser.add_argument("--phase2_qty_heur_reset", type=int, default=0, help="""
            once we apply a greedy heuristic to fix production qty x s.t.
                x >= min_prod.
            In theory dual simplex should work well, so we have two strategy:
            0 we do not reset model and hot start
            1 reset and presolve and solve from scratch.
        """)

    def __init__(self):
        args = self.parser.parse_args()
        self.phase1_relax = args.phase1_relax
        self.phase1_resolve = args.phase1_resolve
        self.phase2_use_full_model = args.phase2_use_full_model
        self.phase2_use_qty_heur = args.phase2_use_qty_heur
        self.phase2_qty_heur_reset = args.phase2_qty_heur_reset
        self.phase2_greedy_range = 5

    def show(self):
        print("--- ALG PARAMS ---")
        print("- phase1_relax:", self.phase1_relax)
        print("- phase1_resolve:", self.phase1_resolve)
        print("- phase2_use_full_model:", self.phase2_use_full_model)
        print("- phase2_use_qty_heur:", self.phase2_use_qty_heur)
        print("- phase2_qty_heur_reset:", self.phase2_qty_heur_reset)
        print("------------------")


# singleton
DEFAULT_ALG_PARAMS = AlgParams()

GLOBAL_PROFILE = {
    'count': collections.defaultdict(int),
    'total': collections.defaultdict(float)
}


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time.time()
        cc = func(*args, **kwargs)
        et = time.time()
        GLOBAL_PROFILE['total'][func.__qualname__] += et - st
        GLOBAL_PROFILE['count'][func.__qualname__] += 1
        print(f"func @{func.__qualname__} finished")
        return cc

    return wrapper


def show_profiling():
    print("|--- PROFILING STATS ---")
    stats = pd.DataFrame.from_dict(GLOBAL_PROFILE)
    stats['avg'] = stats['total'] / stats['count']
    stats = stats.sort_values(by='total', ascending=False)
    print(stats.to_markdown())
