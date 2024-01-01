//
// Created by C. Zhang on 2021/2/4.
//

#include "action.h"
#include <iostream>

action::~action() = default;


action::action(const action &action) {
    *this = action;
}

action::action(int i, int j, double f, double t, double c) {
    this->i = i;
    this->j = j;
    this->f = f;
    this->t = t;
    this->c = c;
}

std::string action::to_string() {
    return std::to_string(this->i)
           + "-" + std::to_string(this->j)
           + "@" + std::to_string(this->c)
           + "/" + std::to_string(this->t)
           + ":" + std::to_string(this->f);
}

//std::string action::to_string() const {
//
//}