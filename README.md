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

This implementation defines a templated `AGI::TestBed` class, which requires the user to provide two interacting component types:

- **`CortexType`** – The core model under test. It accumulates internal state from past inputs and generates predictions of future inputs.
- **`InputSample`** – A binary-encoded input sample representing signals from virtual sensors or actuators. Each input consists of multiple parallel 1-bit signals (channels) at a single point in time.

---

## API Requirements

### `CortexType`
Your cortex class must:
- Satisfy the `std::regular` concept.
- Provide methods to accept inputs and generate predictions using the following interface:
  ```cpp
  CortexType& CortexType::operator << (const InputSample&); // Process input p
  InputSample CortexType::predict() const;                  // Returns predicted next input
  ```

### `InputSample`
Your input class must:
- Satisfy the `std::regular` concept.
- Provide methods to access the input size and enable bit-level access through:
  ```cpp
  static size_t InputSample::size();                  // Returns number of input bits
  bool InputSample::operator[](size_t i) const;       // Read-only access to the i-th bit
  InputSample::reference InputSample::operator[](size_t i); // Write access to the i-th bit
  ```

---

## Configuration Parameters

AGITB requires one problem-specific and two system-level parameters:

- **`temporal_pattern_length`** (*runtime parameter*) – This is passed to `Testbed::run(...)` and defines the length of repeating input sequences. Longer patterns increase test complexity. Choose a value that balances the cortex’s learning ability and the spatial size of the input.

- **`SimulatedInfinity`** (*compile-time parameter*) – Defined as a template argument to the `Testbed` class. It sets a practical upper bound on the number of timesteps available for learning. This simulates an "infinite" time window in a finite setting. Default: `1000`.

- **`Repetitions`** (*compile-time parameter*) – Also defined as a template argument to the `Testbed` class. It sets the number of times each of the 12 tests is repeated to increase statistical robustness. Default: `100`.

---

## Example: Main Program

```cpp
#include "agitb.h"

class MyInput { ... };
class MyCortex { ... };

int main() {
    using AGITB = sprogar::AGI::TestBed<MyCortex, MyInput, 1000 /* SimulatedInfinity */, 100 /* Repetitions */>;
    
    AGITB::run(5 /* temporal_pattern_length */);
    return 0;
}
```

---

## License

This implementation is released as **free software** under the **GNU General Public License v3.0 (GPL-3.0)**. You are free to run, study, modify, and share this software under the terms of the license.

🔗 [https://github.com/matejsprogar/agitb](https://github.com/matejsprogar/agitb)
