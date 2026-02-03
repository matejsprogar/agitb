# AGITB – Artificial General Intelligence Testbed

This repository contains the official C++ reference implementation of the **Artificial General Intelligence Testbed (AGITB)**, as described in the latest [paper](doc/AGITB.pdf). A corresponding [arXiv](https://arxiv.org/abs/2504.04430) preprint is also available, though it may not reflect the most recent updates.

---

## Thesis

> **The capacity to pass the AGITB constitutes a necessary condition for moving beyond narrow, task-specific AI.**

While current AI systems often give the impression of intelligence, they lack a grounded understanding and therefore cannot be regarded as genuine instances of AGI. To distinguish between surface-level imitation and measurable progress toward true general intelligence, we need a rigorous, transparent, and actionable benchmark.

---

## AGITB Goal

AGITB is a benchmark designed to evaluate artificial general intelligence at its most fundamental level. It consists of a collection of short, intuitive tests that assess whether a system satisfies a set of axioms intended to capture core properties of general intelligence. All tests are fully automated.

Unlike conventional benchmarks that focus on symbolic reasoning, natural language performance, or domain-specific tasks, AGITB operates at the level of binary signal processing. Models interact only with low-level binary inputs and outputs, without access to semantic structure, task descriptions, or pretrained knowledge. This design forces systems to demonstrate genuine adaptation, prediction, and generalisation, rather than relying on memorisation, heuristics, or large-scale pretraining.

By stripping away high-level abstractions, AGITB provides a biologically inspired testing environment that mirrors how intelligence can emerge from raw sensory data. The benchmark is intended to support the **development**, **evaluation**, and **validation** of AGI systems in an architecture-agnostic and implementation-independent manner.

---

## The C++ Reference Implementation

AGITB is distributed as a header-only library. Its central abstraction is the class template `TestBed<MyModel>`, where the template parameter `MyModel` denotes 
the AGI type under evaluation. Each instance of `MyModel` represents a candidate model that, given an input object, is expected to produce a prediction for the 
next input.

An input represents a binary input sample originating from (simulated) sensors or actuators. It consists of multiple parallel one-bit channels captured at a single 
time step. Although AGITB's internal input type is `std::bitset<10>`, a model may instead operate on a custom input type (i.e. `MyInput`), as long as `MyInput` is both 
constructible from and convertible to `std::bitset<>`.

---

## API Requirements for the `MyModel` class

The `MyModel` class must:
- Satisfy the `std::regular` concept.
- Provide a callable interface (functor) that accepts a single input and returns a prediction, using the following signature:
  ```cpp
  MyInput MyModel::operator ()(const MyInput& p);
  ```


### Stub Implementation of the `MyModel` Class

```cpp
class MyModel
{
	using MyInput = std::bitset<10>;  // or a custom input type satisfying the requirements below

public:
    bool operator==(const MyModel& rhs) const {
      // TODO
      return false;
    }

    MyInput operator ()(const MyInput& p) { 
      // TODO AGI magic here!
      return MyInput{};
    }
};
```
#### Support for a custom `MyInput` class
If `MyModel` was originally designed to operate on input types other than `std::bitset`, it can still be used, as long as `MyInput` supports construction from and conversion to `std::bitset`:
```cpp
struct MyInput
{
    MyInput() = default;
    friend bool operator==(const MyInput&, const MyInput&) = default;
    
    template <size_t L> MyInput(const std::bitset<L>&) { ... }			// std::constructible_from<MyInput, std::bitset<L>>
    template <size_t L> operator std::bitset<L> () const { ... }		// std::convertible_to<MyInput, std::bitset<L>
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

## Reproducibility

When a benchmark run fails, AGITB stops immediately at the first failing test and reports both the **test number** and the **random generator seed** used for that execution. This makes every failure fully reproducible.

To reproduce the exact scenario, rerun the benchmark with the reported test ID and seed.

For example, if test `#12` fails with seed `4026412173`, you can reproduce it with:

```cpp
AGITB::run(12, 4026412173);
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
