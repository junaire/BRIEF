#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
 public:
  Timer(std::string name)
      : start_(std::chrono::high_resolution_clock::now()),
        name_(std::move(name)){};

  ~Timer() {
    auto end = std::chrono::high_resolution_clock::now();

    auto s = std::chrono::time_point_cast<std::chrono::microseconds>(start_)
                 .time_since_epoch()
                 .count();
    auto e = std::chrono::time_point_cast<std::chrono::microseconds>(end)
                 .time_since_epoch()
                 .count();

    auto dura = e - s;
    std::cout << "[" << name_ << "] Duration: " << dura << "us(" << dura * 0.001
              << " ms)\n";
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::string name_;
};
