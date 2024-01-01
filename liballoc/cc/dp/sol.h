//
// Created by C. Zhang on 2021/2/7.
//

#ifndef DUALNEWSVENDOR_SOL_H
#define DUALNEWSVENDOR_SOL_H
#include <vector>
class Solution {
public:

    double *x;
    double *s;
    double *m;
    double value;
    Eigen::ArrayXd compactSol;

    Solution(){
        x = nullptr;
        s = nullptr;
        m = nullptr;
        value = 0.0;
    }

    Solution(Eigen::ArrayXXd &array, double v);

    ~Solution() = default;
    double * get_x() const;
    double * get_s() const;
    double * get_m() const;

};

std::vector<double> get_solutions(Solution &s, int size, bool print);
std::vector<double> get_solutions(std::vector<Solution> &s, int size);
#endif //DUALNEWSVENDOR_SOL_H
