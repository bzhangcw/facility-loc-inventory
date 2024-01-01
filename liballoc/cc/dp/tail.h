//
// Created by C. Zhang on 2021/2/3.
//

#ifndef REPAIRCPP_TAIL_H
#define REPAIRCPP_TAIL_H

#include <iostream>
#include "state.h"
#include "action.h"

class tail_counter {
public:
    int number = 0;
};


class tail {
public:
    int id;
    state st;
    action ac;
    std::string key;

    tail() {
        this->st = state();
        this->ac = action(); // upstream ac
        this->id = -1;
        this->key = "";
    }

    tail(state &st, action &ac);

    std::string to_string();

    ~tail();
};


#endif //REPAIRCPP_TAIL_H
