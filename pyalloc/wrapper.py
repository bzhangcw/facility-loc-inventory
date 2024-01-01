import numpy as np

from .pydp import *


def convert_to_c_arr(size, lambda_k):
    c_arr = double_array_py(size)
    for i in range(size):
        c_arr[i] = lambda_k[i]

    return c_arr


def convert_to_c_arr_int(size, lambda_k):
    c_arr = int_array_py(size)
    for i in range(size):
        c_arr[i] = lambda_k[i]

    return c_arr


def solve_by_dp_cc(data, select=None, verbose=True, inexact=False):
    verbose and print("libroute start")
    if select is None:
        sol = run_dp(
            data['n'],
            data['m'],
            convert_to_c_arr(data['m'], data['f']),
            convert_to_c_arr(data['m'], data['D']), # not really used
            convert_to_c_arr_int(data['m'], data['I'].tolist()),
            convert_to_c_arr_int(data['m'], data['J'].tolist()),
            convert_to_c_arr_int(data['n'], data['V'].tolist()),
            convert_to_c_arr(data['n'], data['c']),
            convert_to_c_arr(data['m'], data['T']),
            convert_to_c_arr(data['n'], data['S']),
            convert_to_c_arr(data['n'], data['a']),
            convert_to_c_arr(data['n'], data['b']),
            data['C'],
            verbose,
            inexact,
            20.0
        )
        return [*sol, 0]

    _idx_v, _idx_e = select
    _n, _m = int(len(_idx_v)), int(len(_idx_e))
    nmax = int(max(data['I']) + 1)
    try:
        sol = run_dp(
            _n,
            _m,
            convert_to_c_arr(_m, data['f']),
            convert_to_c_arr(_m, data['D']),  # not really used
            convert_to_c_arr_int(_m, data['I'].tolist()),
            convert_to_c_arr_int(_m, data['J'].tolist()),
            convert_to_c_arr_int(_n, data['V'][_idx_v].tolist()),
            convert_to_c_arr(nmax, data['c']),
            convert_to_c_arr(_m, data['T']),
            convert_to_c_arr(nmax, data['S']),
            convert_to_c_arr(nmax, data['a']),
            convert_to_c_arr(nmax, data['b']),
            data['C'],
            verbose,
            inexact,
            20000.0
        )
    except:
        raise ValueError("libroute failed")
    return [*sol, 0]


