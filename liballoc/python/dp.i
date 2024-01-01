/* File: dp.i */
%module dp
%include "carrays.i"
%include "std_vector.i"
%include "cpointer.i"

// arrays and vectors
%array_class(double, double_array_py);
%array_class(int, int_array_py);

%pointer_functions(double, doubleP);
%pointer_functions(int, intP);

namespace std
        {
                %template(DoubleVector) vector<double>;
                %template(IntVector) vector<int>;
        }


%{
#define SWIG_FILE_WITH_INIT

#include "dp.h"

%}


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