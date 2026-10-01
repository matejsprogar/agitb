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

Running the tests...
#1 Uninformed start
#2 Determinism
#3 Trace
1/5000

Assertion failed in agitb.h:138
ASSERT(A != Model());

rng_seed: 2140661623
```

---

## Motivation

> **The ability to pass AGITB is a necessary condition for moving beyond narrow, task-specific AI.**

While many current AI systems create the impression of intelligence, they lack grounded understanding and therefore cannot be considered genuine instances of AGI.

To distinguish surface-level imitation from measurable progress toward true general intelligence, we need a benchmark that is rigorous, transparent, and actionable.

---


## Implementation

AGITB is distributed as a header-only library. Its core component is the class template `TestBed<MyModel>`, where `MyModel` is the AGI type under evaluation. Each instance represents a candidate model that receives an input object and predicts the next one.

An input is a binary sample originating from (simulated) sensors or actuators. It consists of multiple parallel one-bit channels captured at a single time step. Internally, AGITB uses `std::bitset<10>`, but models may define a custom input type (e.g. `MyInput`) as long as it is constructible from and convertible 
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

For faster feedback during development, you can control the thoroughness of the evaluation by specifying the number of repetitions for each test:

```cpp
AGITB::run(10);  // Repeat each test 10 times
```

Increasing the number of repetitions improves the thoroughness of the evaluation. However, no finite number of tests can prove that a requirement is satisfied, as doing so would require testing an infinite number of cases.


---

## Reproducibility

When a benchmark run fails, AGITB stops immediately at the first failing test and reports the **random generator seed** used for that run, allowing the exact scenario to be reproduced.

Rerun the benchmark with the reported values to recreate the failure.

For example, in case of the Trace `#3` test failure above, you can reproduce it with:

```cpp
AGITB::run(3, 830706803);
```
---
## Cheating the Benchmark

Because AGITB consists of intentionally simple and transparent tests, it is conceivable that a model could be engineered to optimise for the benchmark rather than for the underlying learning capabilities it is intended to measure. 

For example, a model with a built-in mechanism that shifts its input around a ring of channels passes the test of Req. 11 without learning anything from the world it is shown, although such built-in knowledge is forbidden by Req. 1. One might argue that a rotation mechanism is merely an architectural choice. It is, however, the very rule the test is built on, and any other operation the test might use could be built in just as easily. Such a model is useless in general as it does not learn rules that govern the world. Unfortunately, no test of Req. 1 can detect pre-installed knowledge, because it can only compare fresh instances of a model with each other, and all of them are built alike.

For example, a model with a built-in mechanism that shifts its input around a ring of channels can pass the test of Req. 11 without learning anything from the world it is presented with, even though such pre-installed knowledge is prohibited by Req. 1. One might argue that the rotation mechanism is merely an architectural choice. However, in this case, it is precisely the rule on which the test is based, and any other operation used by the test could be built into the model in the same way. Such a model is of little use in general, as it does not learn the rules governing the world. Unfortunately, no test of Req. 1 can detect pre-installed knowledge, because it can only compare fresh instances of a model with one another, and all instances share the same built-in knowledge.


Other potential avenues include exploiting weaknesses in the benchmark's randomisation or exploiting properties of the comparison procedure between model instances. Such strategies optimise performance with respect to the benchmark implementation rather than the underlying capabilities being assessed.

In principle, the benchmark could be hardened against these forms of exploitation by introducing additional randomisation, obfuscation, or verification mechanisms. However, these measures would inevitably increase implementation complexity while reducing transparency, interpretability, and reproducibility. Since AGITB is intended not only as an evaluation protocol but also as a practical development tool, the reference implementation deliberately favours simplicity and readability over resistance to benchmark-specific optimisations.
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
