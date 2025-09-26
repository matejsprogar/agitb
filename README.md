# AGITB – Artificial General Intelligence Testbed

This repository contains the official C++ reference implementation of the **Artificial General Intelligence Testbed (AGITB)**, as described in the latest [paper](doc/AGITB.pdf). A corresponding [arXiv](https://arxiv.org/abs/2504.04430) preprint is also available, though it may not reflect the most recent updates.

---

## Thesis

> **The capacity to pass the AGITB constitutes a necessary condition for the existence of Artificial General Intelligence.**

While current AI systems often give the impression of intelligence, they lack a grounded understanding and therefore cannot be regarded as genuine instances of AGI. To distinguish between surface-level imitation and true general intelligence—or measurable progress toward it—we require a rigorous, transparent, and actionable benchmark.

---

## AGITB Goal

AGITB serves as a benchmark for evaluating artificial general intelligence. It consists of thirteen core requirements, twelve of which are implemented as fully automated tests that assess essential characteristics of AGI candidates. Unlike traditional benchmarks centered on symbolic reasoning, language performance, or domain-specific tasks, AGITB evaluates systems at the level of binary signal processing. This low-level design compels models to demonstrate adaptation, prediction, and generalization in ways that cannot be reduced to memorization or large-scale pretraining.

The goal is to advance the **development**, **evaluation**, and **validation** of AGI by offering a biologically inspired low-level testing benchmark.

---

## C++ Implementation

AGITB is distributed as a header-only library. Its central abstraction is the templated class `TestBed<CortexUnderTest>`, where `CortexUnderTest` denotes the model type under evaluation. Each instance of `CortexUnderTest` represents a candidate model that, given an input object, is expected to generate a prediction for the subsequent input.

An `InputType` encodes a binary input sample from simulated sensors or actuators, consisting of ten parallel one-bit channels captured at a single time step. By default, AGITB defines `InputType` as `std::bitset<10>`. If this default is unsuitable, users may declare `TestBed<MyCortex, MyInput>` with a custom type. The custom `MyInput` must satisfy the `Indexable` concept defined in `utils.h`.

---

## API Requirements

### `CortexUnderTest`
Your Cortex class must:
- Satisfy the `std::regular` concept.
- Provide methods to accept inputs and retrieve predictions using the following interface:
  ```cpp
  Cortex& Cortex::operator << (const InputType& p); // Process input p
  InputType Cortex::prediction() const;                 // Returns the prediction for the next input
  ```

where `InputType` defaults to `std::bitset<10>`.

### Stub Implementation of the Cortex Class for AGI Testbed

```cpp
using Input = std::bitset<10>;
class Cortex
{
    Input _prediction;

public:
    bool operator==(const Cortex& rhs) const { return false; }  // TODO

    Cortex& operator << (const Input& p) {
        _prediction = AGI(p);                                   // TODO: Magic occurs here!
        return *this;
    }
    Input prediction() const { return _prediction; }
};
```
---


## Usage

To use the AGITB testbed, include the main header file `agitb.h` and call the static `run()` method of the `TestBed<Cortex>` class, providing your `Cortex` type as template parameter.

### Example

```cpp
#include "path/to/agitb.h"

int main() {
    using AGITB = sprogar::AGI::TestBed<Cortex>;
    
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
