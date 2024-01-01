//
// Created by C. Zhang on 2021/2/4.
//

#include "state.h"
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

state::state(int s, double v, double t, double dt, double c, const int *vs, int n) {
    this->v = v; // v(s)
    this->s = s; // s
    this->ts = t; // vs
    this->te = t + dt; // vs
    this->c = c; // vs
    std::vector<int> vec(vs, vs + n);
    this->unv = vec;
}

state::state(int s, double v, double t, double dt, double c, std::vector<int> unc) {
    this->v = v; // v(s)
    this->s = s; // s
    this->ts = t; // vs
    this->te = t + dt; // vs
    this->c = c; // vs
    this->unv = unc;
}

state::state(const state &s) {
    *this = s;
}

std::string state::to_string() const {

    std::stringstream result;
    std::copy(
            this->unv.begin(),
            this->unv.end(),
            std::ostream_iterator<int>(result, "-"));
    return std::to_string(this->s)         // current
           + ":" + std::to_string(this->c) // used capacity
           + "/" + std::to_string(this->ts) // time
           + "@" + result.str();           // left nodes

}

bool operator==(const state &lhs, const state &rhs) {
    return lhs.s == rhs.s && lhs.v == rhs.v;
}


state state::apply(const action &ac, double node_time) {
    auto cc = std::vector<int>(this->unv);
    auto position = std::find(cc.begin(), cc.end(), ac.j);
    if (position != cc.end()) // == cc.end() means the element was not found
        cc.erase(position);
    state s = state(
            ac.j,
            this->v + ac.f,
            this->te + ac.t,
            node_time,
            this->c + ac.c,
            cc
    );

    return s;
}


double state::apply() {
    return 0.0;
}

void state::adjust(double lb, double ub, double *b) {
    // if arrival earlier, move to lb;
    if (this->ts < lb) {
        double offset = lb - this->ts;
        this->ts = lb;
        this->te += offset;
    }
    // filter out nodes if this.t > b[n]
    this->unv.erase(std::remove_if(
            this->unv.begin(), this->unv.end(),
            [&b, this](const int &x) {
                return b[x] <= this->te;
            }), this->unv.end());

}



