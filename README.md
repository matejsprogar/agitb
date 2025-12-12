# AGITB – Artificial General Intelligence Testbed

This repository contains the official C++ reference implementation of the **Artificial General Intelligence Testbed (AGITB)**, as described in the latest [paper](doc/AGITB.pdf). A corresponding [arXiv](https://arxiv.org/abs/2504.04430) preprint is also available, though it may not reflect the most recent updates.

---

## Thesis

> **The capacity to pass the AGITB constitutes a necessary condition for moving beyond narrow, task-specific AI.**

While current AI systems often give the impression of intelligence, they lack a grounded understanding and therefore cannot be regarded as genuine instances of AGI. To distinguish between surface-level imitation and measurable progress toward true general intelligence, we need a rigorous, transparent, and actionable benchmark.

---

## AGITB Goal

AGITB serves as a benchmark for evaluating artificial general intelligence. It consists of fourteen core requirements, thirteen of which are implemented as fully automated tests that assess essential characteristics of AGI candidates. Unlike traditional benchmarks centred on symbolic reasoning, language performance, or domain-specific tasks, AGITB evaluates systems at the level of binary signal processing. This low-level design compels models to demonstrate adaptation, prediction, and generalisation in ways that cannot be reduced to memorisation or large-scale pretraining.

The goal is to advance the **development**, **evaluation**, and **validation** of AGI by offering a biologically inspired low-level testing benchmark.

---

## The C++ Reference Implementation

AGITB is distributed as a header-only library. Its central abstraction is the templated class `TestBed<MyModel>`, where `MyModel` denotes the AGI type under evaluation. Each instance of the `MyModel` represents a candidate model that, given an input object, is expected to generate a prediction for the subsequent input.

An `InputType` encodes a binary input sample from simulated sensors or actuators, consisting of ten parallel one-bit channels captured at a single time step. By default, AGITB defines `InputType` as `std::bitset<10>`.

---

## API Requirements for `SystemUnderEvaluation`
The MyModel class must:
- Satisfy the `std::regular` concept.
- Provide methods to accept inputs and retrieve predictions using the following interface:
  ```cpp
  MyModel& MyModel::operator << (const InputType& p);   // Process input p
  InputType MyModel::get_prediction() const;                // Returns the prediction for the next input
  ```

### Stub Implementation of the MyModel Class for AGI Testbed

```cpp
using Input = std::bitset<10>;
class MyModel
{
    Input _prediction;

public:
    bool operator==(const MyModel& rhs) const {
      // TODO
      return false;
    }

    MyModel& operator << (const Input& p) {
        _prediction = AGI(p);
        return *this;
    }
    Input get_prediction() const { return _prediction; }

private:
    Input AGI(const Input& current) {
      // AGI magic TODO here!
      return Input{};
    }
};
```
---


## Usage

To use the AGITB testbed, include the main header file `agitb.h` and call the static `run()` method of the `TestBed<MyModel>` class, providing your `MyModel` type as template parameter.

### Example

```cpp
#include "path/to/agitb.h"

int main() {
    using AGITB = sprogar::AGI::TestBed<MyModel>;
    
    AGITB::run();
    return 0;
}
```
---

## Requirements

To build and run this project, you will need a **C++20-compatible compiler** 

> 💡 Make sure your build environment is configured to enable C++20 support  
> (e.g., use `-std=c++20` with `g++` or `clang++`).

---

## License

This implementation is released as **free software** under the **GNU General Public License v3.0 (GPL-3.0)**. You are free to run, study, modify, and share this software under the terms of the license.

🔗 [https://github.com/matejsprogar/agitb](https://github.com/matejsprogar/agitb)
