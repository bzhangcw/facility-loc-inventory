//
// Created by C. Zhang on 2023/4/16.
//

#ifndef DPROUTING_ACTION_H
#define DPROUTING_ACTION_H

#include <utility>
#include <string>
#include <vector>
class action {
public:
    int i{};
    int j{};
    double f{}; // cost of this approach
    double t{}; // time spent of this approach
    double c{}; // capacity usage of this approach
    action() {
        this->i = 0;
        this->j = 0;
        this->f = 0.0;
        this->t = 0.0;
        this->c = 0.0;
    }

    ~action();

    action(const action &action);

    action(int, int, double, double, double);

    std::string to_string();
};


#endif //DPROUTING_ACTION_H
