# AGITB – Artificial General Intelligence Testbed

This repository contains the official C++ reference implementation of the **Artificial General Intelligence Testbed (AGITB)**, as described in the latest [paper](doc/AGITB.pdf). A corresponding [arXiv](https://arxiv.org/abs/2504.04430) preprint is also available, though it may not reflect the most recent updates.

---

## Thesis

> **AGI needs a metric.**
<p>While Large Language Models (LLMs) may be able to pass certain versions of the Turing Test, they do so without grounded understanding and cannot be considered genuine Artificial General Intelligence (AGI). If we are to determine whether a system is truly intelligent—or meaningfully progressing toward general intelligence—we need a clear, rigorous, and actionable metric that goes beyond surface-level imitation.</p>

---

## AGITB Goal

AGITB provides a suite of thirteen core requirements, twelve of which are implemented as fully automated tests that evaluate essential characteristics required of an AGI system. Unlike benchmarks focused on symbolic reasoning or language performance, AGITB operates at the level of binary signal processing, where the model must demonstrate adaptation, prediction, and generalization without relying on pretraining or memorization.

The goal is to advance the **development**, **evaluation**, and **verification** of AGI by offering a biologically inspired low-level testing framework.
---

## C++ Implementation

AGITB is implemented as a **header-only** library. It defines a templated `TestBed<Cortex, Input=std::bitset<10>>` class, which requires the user to provide the Cortex component type. The instances of this type represent models under test. Upon receiving an input, a cortex object is supposed to generate a prediction for the subsequent input.

The AGITB assumes that the Cortex objects can interact with std::bitset<> type inputs. If this is not the case, the user has the option to define a custom `Input` type that meets the specified interface requirements. An Input object is nothing but a binary-encoded sample representing signals from virtual sensors or actuators. Each input consists of multiple parallel 1-bit signals (channels) at a single point in time.

---

## API Requirements

### `Cortex`
Your Cortex class must:
- Satisfy the `std::regular` concept.
- Provide methods to accept inputs and retrieve predictions using the following interface:
  ```cpp
  Cortex& Cortex::operator << (const Input& p);    // Process input p
  Input Cortex::prediction() const;                // Returns the (cached) prediction for the next input
  ```

- where `Input` defaults to std::bitset<10>, but can also be custom implemented:
- 
- ### `Input`
If you need to define a custom `Input`, your type must meet the following interface requirements:
- Satisfy the `std::regular` concept.
- Provide methods to access the input size and enable bit-level access through:
  ```cpp
  static size_t Input::size();                  // Returns number of input bits
  bool Input::operator[](size_t i) const;       // Read-only access to the i-th bit
  Input::reference Input::operator[](size_t i); // Write access to the i-th bit
  ```

- 
### Stub Implementation of the Cortex Class for AGI TestBed

```cpp
class Cortex
{
public:
    bool operator==(const Cortex& rhs) const { return true; }   // TODO: Full member-wise comparison

    Cortex& operator << (const Input& p) { return *this; }      // TODO: Process input p
    Input prediction() const { return Input{}; }                // TODO: Returns the (cached) prediction for the next input
};

/* If your Cortex cannot work with std::bitset<>, you can define a custom Input type */
//class CustomInput
//{
//public:
//    CustomInput() {}
//    bool operator==(const CustomInput& rhs) const { }     // TODO: Full member-wise comparison
//
//    using reference = bool&;
//    static size_t size() { }                              // TODO: Returns number of input bits
//    bool operator[](size_t i) const { }                   // TODO: Read-only access to the i-th bit
//    reference operator[](size_t i) { }                    // TODO: Write access to the i-th bit    
//};

```
---


## Usage

To use the AGITB testbed, include the main header file and call the static `run()` method of the `TestBed<Cortex>` class, providing your `Cortex` type as template parameter.

### Example

```cpp
#include "agitb.h"

int main() {
    using AGITB = sprogar::AGI::TestBed<Cortex>;
    // using AGITB = sprogar::AGI::TestBed<Cortex, CustomInput>;	// only if using custom Input type
    
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
