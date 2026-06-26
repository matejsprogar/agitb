# AGITB – Artificial General Intelligence Testbed

A small, self-contained C++ benchmark that evaluates predictive models on raw binary streams, 
intended as a practical step toward artificial general intelligence. Provide your model and run AGITB.
By design, most systems will not pass.

AGITB includes 12 short, intuitive, fully automated tests.

- header-only implementation
- no dependencies
- deterministic and reproducible
- builds in seconds

Concept paper: [doc/FR4AGI.pdf](doc/FR4AGI.pdf)

### Quick start

Compile and run the provided stub model:

```bash
g++ -std=c++23 stub.cpp -o stub
$ ./stub
```
Example output (the stub fails on test #3):
```text
Artificial General Intelligence Testbed

Running 12 tests...
0 Semantic integrity
#1 Uninformed start
#2 Determinism
#3 Trace
1/5000

Assertion failed in C:\Users\matej\Projects\library\sprogar\include\agitb.h:139
A != B

rng_seed: 2140661623
```

---

## Motivation

> **The ability to pass AGITB is a necessary condition for moving beyond narrow, task-specific AI.**

While many current AI systems create the impression of intelligence, they lack grounded understanding and therefore cannot be considered genuine instances of AGI.

To distinguish surface-level imitation from measurable progress toward true general intelligence, we need a benchmark that is rigorous, transparent, and actionable.

---


## Implementation

AGITB is distributed as a header-only library. Its core component is the class template `TestBed<MyModel>`, where `MyModel` is the AGI type under evaluation. 
Each instance represents a candidate model that receives an input object and predicts the next one.

An input is a binary sample originating from (simulated) sensors or actuators. It consists of multiple parallel one-bit channels captured at a single time step. 
Internally, AGITB uses `std::bitset<10>`, but models may define a custom input type (e.g. `MyInput`) as long as it is constructible from and convertible 
to `std::bitset<>` (see below).


### API Requirements for the `MyModel` class

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
      return true;
    }

    MyInput operator ()(const MyInput& p) { 
      // TODO AGI magic here!
      return MyInput{};
    }
};
```
#### Support for a custom `MyInput` class
If `MyModel` was originally designed to operate on input types other than `std::bitset`, it can still be used, as long as `MyInput` 
supports construction from and conversion to `std::bitset`:

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

To use the AGITB testbed, include the main header file `agitb.h` and call the static `run()` method of the `TestBed<MyModel>` class, 
providing your `MyModel` type as template parameter:

```cpp
#include "path/to/agitb.h"

int main() {
	using AGITB = sprogar::AGI::TestBed<MyModel>;
    
    AGITB::run();
    return 0;
}
```

For quicker feedback during development, you can adjust the evaluation thoroughness by specifying how many times each test is repeated:

```cpp
    AGITB::run(10);	// repeats each test 10 times
```
---

## Reproducibility

When a benchmark run fails, AGITB stops immediately at the first failing test and reports the **random generator seed** used for that run, 
allowing the exact scenario to be reproduced.

Rerun the benchmark with the reported values to recreate the failure.

For example, in case of the Trace `#3` test failure above, you can reproduce it with:

```cpp
AGITB::run(3, 830706803);
```
---
##Cheating the Benchmark

Because AGITB's individual tests are intentionally simple and transparent, it is theoretically possible to tailor a model specifically to the benchmark rather than to the underlying capabilities it is intended to measure. For example, one could modify the model comparison function or otherwise exploit knowledge of the testbed's implementation. Such strategies do not demonstrate general learning ability — they merely exploit the benchmark itself.

While AGITB could be made more resistant to this type of manipulation, doing so would inevitably reduce its transparency, interpretability, and ease of inspection. Since one of the benchmark's primary goals is to help researchers understand and improve their models, the reference implementation deliberately prioritizes readability over adversarial robustness. Some safeguards against trivial exploitation are already in place, but they are not intended to obscure the benchmark's operation.

A more subtle form of gaming the benchmark would be to construct a model that succeeds only because AGITB approximates conditions that are, in principle, unbounded. Although this value is necessarily finite, it has proven sufficient in combination with the current benchmark configuration (temporal patterns consisting of seven 10-bit input signals) to distinguish promising approaches from those that fail to exhibit the required behaviour. At the same time, it keeps execution times low enough to support rapid experimentation and iterative model development.

For these reasons, the AGITB reference implementation is intentionally kept simple, readable, and efficient. Unless a practical method of exploiting the benchmark is demonstrated, introducing additional complexity solely to make the implementation harder to game would offer little benefit while diminishing its value as a research tool.

---

## Feedback and contributions welcome

AGITB is intended to be small, transparent, and easy to experiment with.

Feedback, bug reports, and improvement ideas are very welcome. In particular:

- models that pass the full test suite
- unclear or ambiguous tests
- correctness issues or edge cases
- performance improvements
- additional tests or tasks
- alternative model adapters or integrations
- results from interesting systems

Issues and pull requests are encouraged.

---

## Requirements

To build and run this project, you will need a **C++23-compatible compiler** 

> 💡 Make sure your build environment is configured to enable C++23 support  
> (e.g., use `-std=c++23` with `g++` or `clang++`).

---

## License

This implementation is released as **free software** under the **GNU General Public License v3.0 (GPL-3.0)**. You are free to run, study, modify, 
and share this software under the terms of the license.

🔗 [https://github.com/matejsprogar/agitb](https://github.com/matejsprogar/agitb)
