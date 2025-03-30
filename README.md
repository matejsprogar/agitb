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

- **`Cortex`** – The core model under test. It accumulates internal state from past inputs and generates predictions of future inputs.
- **`Input`** – A binary-encoded input sample representing signals from virtual sensors or actuators. Each input consists of multiple parallel 1-bit signals (channels) at a single point in time.

---

## API Requirements

### `Input`
Your input class must:
- Satisfy the `std::regular` concept.
- Provide methods to access the input size and enable bit-level access through:
  ```cpp
  static size_t Input::size();                  // Returns number of input bits
  bool Input::operator[](size_t i) const;       // Read-only access to the i-th bit
  Input::reference Input::operator[](size_t i); // Write access to the i-th bit
  ```

### `Cortex`
Your cortex class must:
- Satisfy the `std::regular` concept.
- Provide methods to accept inputs and generate predictions using the following interface:
  ```cpp
  Cortex& Cortex::operator << (const Input& p); // Process input p
  Input Cortex::predict() const;                // Returns predicted next input
  ```

---

## Configuration Parameters

AGITB requires one solution-specific and two system-level template parameters:

- **`temporal_pattern_length`** – A required parameter passed to the TestBed::run() method. It defines the length of the repeating input sequences. Longer patterns increase test difficulty, so this value should be chosen to balance the Cortex’s learning ability with the spatial size of the input.

- **`SimulatedInfinity`** – An optional template parameter of the TestBed class. It defines a practical upper bound on the number of timesteps available for learning, simulating an "infinite" time window within a finite setting. Default: 1000.

- **`Repetitions`** – An optional template parameter of the TestBed class. It specifies how many times each of the 12 tests is repeated to improve statistical robustness. Default: 100.

---

## Example: Main Program

```cpp
#include "agitb.h"

class Input { ... };
class Cortex { ... };

int main() {
    using AGITB = sprogar::AGI::TestBed<Cortex, Input>;
    
    AGITB::run(5 /* temporal_pattern_length */);
    return 0;
}
```

---

## License

This implementation is released as **free software** under the **GNU General Public License v3.0 (GPL-3.0)**. You are free to run, study, modify, and share this software under the terms of the license.

🔗 [https://github.com/matejsprogar/agitb](https://github.com/matejsprogar/agitb)
