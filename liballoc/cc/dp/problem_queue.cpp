//
// Created by C. Zhang on 2021/2/4.
//

#include "problem_queue.h"

int problem_queue::insert(state &tail) {
    std::string tail_key = tail.to_string();

    auto got = this->_set.find(tail_key);
    if (got == this->_set.end()) {
        // insert
        this->_queue.push(tail);
        this->_set.insert(tail_key);
        return 1;
    }
    return 0;
}

bool problem_queue::is_empty() const {
    return this->_queue.empty();
}

problem_kv_pair problem_queue::get_last() {
    state tl = this->_queue.top();
    return problem_kv_pair(tl.to_string(), tl);
}

problem_kv_pair problem_queue::pop() {
    state tl = this->_queue.top();
    this->_queue.pop();
    this->_set.erase(tl.to_string());
    return problem_kv_pair(tl.to_string(), tl);
}

