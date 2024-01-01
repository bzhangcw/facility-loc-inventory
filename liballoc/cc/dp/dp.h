//
// Created by chuwen on 2021/2/6.
//

#ifndef DUALNEWSVENDOR_DP_H
#define DUALNEWSVENDOR_DP_H

#include <iostream>
#include <fstream>
#include <array>
#include <random>
#include <thread>
#include <future>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "nlohmann/json.hpp"
#include "problem_queue.h"
#include "sol.h"

using json = nlohmann::json;
using arrayd = Eigen::ArrayXd;
using arrayf = Eigen::ArrayXf;
using int_array = Eigen::ArrayXi;

bool bool_valid_route(state, double);

bool bool_valid_route(state, double, double, double);

struct problem_data {
    int n;     // node size
    int m;     // edge size
    std::vector<double> f; // cost array of E
    std::vector<double> D; // distance array of E
    std::vector<int> I;    // i~ of (i,j) := e in E
    std::vector<int> J;    // j~ of (i,j) := e in E
    std::vector<int> V;    // nodes
    std::vector<double> c; // capacity usage
    std::vector<double> T; // time needed to travel
    std::vector<double> S; // time needed to serve
    std::vector<double> a; // lb of time-window
    std::vector<double> b; // ub of time-window
    double C;  // capacity
};


//Solution run_dp_single_sol(
void void_run_dp_single_sol(
        int n,     // node size
        int m,     // edge size
        double *f, // cost array of E
        double *D, // distance array of E
        int *I,    // i~ of (i,j) := e in E
        int *J,    // j~ of (i,j) := e in E
        int *V,    // nodes
        double *c, // capacity usage
        double *T, // time needed to travel
        double *S, // time needed to serve
        double *a, // lb of time-window
        double *b, // ub of time-window
        double C,  // capacity
        bool verbose,
        bool inexact,
        double timelimit
);

std::vector<action>
run_dp_single(
        int n,     // node size
        int m,     // edge size
        double *f, // cost array of E
        double *D, // distance array of E
        int *I,    // i~ of (i,j) := e in E
        int *J,    // j~ of (i,j) := e in E
        int *V,    // nodes
        double *c, // capacity usage
        double *T, // time needed to travel
        double *S, // time needed to serve
        double *a, // lb of time-window
        double *b, // ub of time-window
        double C,  // capacity
        bool verbose,
        bool inexact,
        double timelimit
);

std::vector<int>
run_dp(
        int n,     // node size
        int m,     // edge size
        double *f, // cost array of E
        double *D, // distance array of E
        int *I,    // i~ of (i,j) := e in E
        int *J,    // j~ of (i,j) := e in E
        int *V,    // nodes
        double *c, // capacity usage
        double *T, // time needed to travel
        double *S, // time needed to serve
        double *a, // lb of time-window
        double *b, // ub of time-window
        double C,  // capacity
        bool verbose,
        bool inexact,
        double timelimit
);


problem_data parse_data(const std::string &fp);

problem_data parse_data(char *fp);

json parse_json(char *fp);

json parse_json(const std::string &fp);


#endif //DUALNEWSVENDOR_DP_H
