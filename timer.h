#pragma once

#include <chrono>

class Timer {
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;

public:
    Timer() : beg_(clock_::now()) {
    }

    double measure() {
        auto now = clock_::now();
        auto count = std::chrono::duration_cast<second_>(now - beg_).count();
        beg_ = now;
        return count;
    }
};
