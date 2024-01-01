// @license: %MIT License%:~ http://www.opensource.org/licenses/MIT
// @project: dp
// @file: /sol.cpp
// @created: Wednesday, 17th February 2021
// @author: C. Zhang (chuwzhang@gmail.com)
// @modified: C. Zhang (chuwzhang@gmail.com>)
//    Friday, 19th February 2021 4:25:55 pm
// @description:
//  solution container for DP algorithm
#include <iostream>
#include <Eigen/Dense>
#include "sol.h"

Solution::Solution(Eigen::ArrayXXd &output, double v)
{
    x = output.col(0).data();
    m = output.col(1).data();
    s = output.col(2).data();
    value = v;

    compactSol = Eigen::ArrayXd(output.col(0).size() * 3);
    compactSol << output.col(2), output.col(1), output.col(0);
}

double *Solution::get_x() const
{
    return this->x;
}

double *Solution::get_m() const
{
    return this->m;
}

std::vector<double> get_solutions(Solution &sol, int size, bool verbose = false)
{

    auto rtl = std::vector<double>(size * 3);
    for (int i = 0; i < size * 3; ++i)
    {
        rtl[i] = sol.compactSol[i];
    }
    rtl.push_back(sol.value);
    if (verbose)
    {
        std::cout << sol.compactSol << std::endl;
        for (auto &x : rtl)
            std::cout << x << ",";
    }
    return rtl;
}

std::vector<double> get_solutions(std::vector<Solution> &sols, int size)
{

    auto length = sols.size();
    auto rtl = std::vector<double>((size * 3) * length);
    int current_sol = 0; // slides
    for (auto &sol : sols)
    {
        for (int i = 0; i < size * 3; ++i)
        {
            rtl[current_sol] = sol.compactSol[i];
            current_sol++;
        }
        rtl.push_back(sol.value);
    }
    return rtl;
}