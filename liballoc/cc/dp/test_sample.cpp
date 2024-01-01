//
// Created by C. Zhang on 2021/2/4.
//

#include <iostream>
#include <queue>
#include <unordered_map>

#include "Eigen/Dense"
#include "nlohmann/json.hpp"

#include "problem_queue.h"
#include "dp.h"

using Eigen::MatrixXf;
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) {

    /*
     * TEST DATA WITH BENCHMARK RESULTS
     * @date: 2021/02/05
     *
     * */
    nlohmann::json test = parse_json(argv[1]); // benchmark stored at "src/test/test.json"
    problem_data p = parse_data(argv[1]);
    bool inexact = argv[2][0] == 'T';
    void_run_dp_single_sol(
            p.n,
            p.m,
            p.f.data(),
            p.D.data(),
            p.I.data(),
            p.J.data(),
            p.V.data(),
            p.c.data(),
            p.T.data(),
            p.S.data(),
            p.a.data(),
            p.b.data(),
            p.C,
            true,
            false,
            2000
    );
    return 0;
}