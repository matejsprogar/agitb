# AGITB – Artificial General Intelligence Test Bed

This repository contains a C++ implementation of the **Artificial General Intelligence Test Bed (AGITB)**, as described in [this paper](doc/AGITB.pdf).

---

## Thesis

> **AGI needs a metric.**

While Large Language Models (LLMs) may appear capable of passing the Turing test, they lack grounded reasoning and cannot be considered genuine Artificial General Intelligence (AGI) in the sense originally envisioned by Alan Turing. To determine whether a machine is truly intelligent—or at least progressing in that direction—we need a clear, rigorous, and actionable metric.

---

## AGITB Goal

AGITB provides a suite of 12 intuitive and rigorous tests that evaluate essential characteristics required of an AGI system. Unlike benchmarks focused on symbolic reasoning or language performance, AGITB operates at the level of binary signal processing, where the model must demonstrate adaptation, prediction, and generalization without relying on pretraining or memorization.

The goal is to support the **development**, **evaluation**, and **recognition** of AGI by offering a low-level yet biologically inspired testing framework.

---

## C++ Implementation

AGITB is implemented as a **header-only** library. It defines a templated `AGI::TestBed` class, which requires the user to provide two interacting component types:

- **`Cortex`** – The core model under test. It accumulates internal state from past inputs and generates predictions of future inputs.
- **`Input`** – A binary-encoded input sample representing signals from virtual sensors or actuators. Each input consists of multiple parallel 1-bit signals (channels) at a single point in time.

---

## API Requirements

### `Input`
Your Input class must:
- Satisfy the `std::regular` concept.
- Provide methods to access the input size and enable bit-level access through:
  ```cpp
  static size_t Input::size();                  // Returns number of input bits
  bool Input::operator[](size_t i) const;       // Read-only access to the i-th bit
  Input::reference Input::operator[](size_t i); // Write access to the i-th bit
  ```
### `Cortex`
Your Cortex class must:
- Satisfy the `std::regular` concept.
- Provide methods to accept inputs and generate predictions using the following interface:
  ```cpp
  Cortex& Cortex::operator << (const Input& p); // Process input p
  Input Cortex::predict() const;                // Returns predicted next input
  ```

### Stub Implementation of Input and Cortex Classes for AGI TestBed

```cpp

class Input
{
public:
    bool operator==(const Input& rhs) const { return true; }    // TODO: Full member-wise comparison
    
    struct reference{
        reference& operator = (bool x) { return *this; }   // TODO: Assigns a value to the referenced bit
        reference& operator = (const reference& x) { return *this; } // TODO: Assigns a value
    };
    static size_t size() { return 0; }                          // TODO: Returns number of input bits
    bool operator[](size_t i) const { return false; }           // TODO: Read-only access to the i-th bit
    reference operator[](size_t i) { return reference{}; }      // TODO: Write access to the i-th bit    
};

class Cortex
{
public:
    bool operator==(const Cortex& rhs) const { return true; }   // TODO: Full member-wise comparison

    Cortex& operator << (const Input& p) { return *this; }      // TODO: Process input p
    Input predict() const { return Input{}; }                   // TODO: Returns predicted next input
};

```
---

## Configuration Parameters

AGITB requires one solution-specific and two system-level template parameters:

- **`input_period`** – A required parameter passed to the TestBed::run() method. It defines the number of time steps in the repeating input sequence that the cortex must learn to recognize and adapt to. A longer input period increases the temporal complexity of the task, making it more difficult for the model to capture and generalize the pattern. Since excessively long input periods may exceed the cortex’s learning capacity—especially in combination with high-dimensional inputs—the user should choose a value that balances temporal complexity with the spatial size of each input sample.

- **`SimulatedInfinity`** – An optional template parameter of the TestBed class. It defines a practical upper bound on the number of timesteps available for learning, simulating an "infinite" time window within a finite setting. Default: 1000.

- **`Repetitions`** – An optional template parameter of the TestBed class. It specifies how many times each of the 12 tests is repeated to improve statistical robustness. Default: 100.

---

## Usage

To use the AGITB testbed, include the main header file and call the static `run()` method of the `TestBed<Cortex, Input>` class, providing your `Cortex` and `Input` types as template parameters, and specifying the required `input_period` as a runtime argument.

### Example

```cpp
#include "agitb.h"

int main() {
    using AGITB = sprogar::AGI::TestBed<Cortex, Input>;
    
    AGITB::run(5 /* input_period */);
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
