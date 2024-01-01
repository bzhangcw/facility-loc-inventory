//
// Created by C. Zhang on 2021/2/4.
//

#ifndef REPAIRCPP_PROBLEM_QUEUE_H
#define REPAIRCPP_PROBLEM_QUEUE_H

#include<stack>
#include<unordered_set>
#include<optional>
#include<functional>

#include "tail.h"
#include <unordered_set>
#include <unordered_map>

using problem_kv_pair = std::pair<std::string, state>;
using state_map = std::unordered_map<std::string, state>;
using value_map = std::unordered_map<std::string, double>;
using action_vec = std::vector<action>;
using tail_vec = std::vector<tail>;
using action_map = std::unordered_map<std::string, action_vec>;
using tail_map = std::unordered_map<std::string, tail_vec>;
using best_tail_map = std::unordered_map<std::string, tail>;


class problem_queue {
public:
    std::stack<state> _queue;
    std::unordered_set<std::string> _set;

    problem_queue() {
        this->_queue = std::stack<state>();
        this->_set = std::unordered_set<std::string>();
    }

    int insert(state &state);

    bool is_empty() const;

    problem_kv_pair get_last();

    problem_kv_pair pop();
};


#endif //REPAIRCPP_PROBLEM_QUEUE_H
