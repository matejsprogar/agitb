/*
* Copyright 2024 Matej Sprogar <matej.sprogar@gmail.com>
*
* This file is part of AGITB - Artificial General Intelligence TestBed.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
* */
#pragma once

#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <chrono>

#include "utils.h"

namespace sprogar {

namespace AGI {
// AGITB environment settings
const size_t SimulatedInfinity = 5000;
const size_t BitsPerInput = 10;
const size_t SequenceLength = 7;

// Artificial General Intelligence TestBed
template <typename SystemUnderEvaluation>
    requires utils::InputPredictor<SystemUnderEvaluation, std::bitset<BitsPerInput>>
class TestBed
{
public:
    using Input = std::bitset<BitsPerInput>;
    using InputSequence = utils::InputSequence<Input>;
    using Model = utils::Model<SystemUnderEvaluation, Input, SimulatedInfinity>;

    enum test_repetitions { RepeatOnce = 1, RepeatForever = SimulatedInfinity };

    static_assert(BitsPerInput > 1);
    static_assert(SequenceLength > 1);

public:
    // Runs all tests from the testbed using the specified test mode.
    static bool run(size_t repetitions_override = 0)
    {
        std::clog << yellow("Artificial General Intelligence Testbed");

        std::clog << "\n\nRunning the tests...\n";
        const std::string go_back(25, '\b'), clear(5, ' ');
        for (const auto& [info, repetitions, test] : testbed) {
            std::clog << info << clear << std::endl;

            const size_t test_repetitions = repetitions_override == 0 ? (size_t)repetitions : std::min((size_t)repetitions, repetitions_override);
            for (size_t r = 1; r <= test_repetitions; ++r) {
                std::clog << r << '/' << test_repetitions << clear << go_back;

                utils::rng.seed(utils::rng_seed = utils::rng());

                test();
            }
        }

        std::clog << green("\n\nPASS\n");
        return true;
    }
    // Runs a specified test from the testbed using the given RNG seed.
    static bool run(unsigned test_number, unsigned seed)
    {
        utils::rng.seed(utils::rng_seed = seed);
        ASSERT(test_number > 0 and test_number <= testbed.size());

        const auto& [info, repetitions, test] = testbed[test_number-1];

        std::clog << yellow("Artificial General Intelligence Testbed");
        std::clog << "\nRunning a single test";
        std::clog << "\nRandom seed: " << rng_seed;
        std::clog << "\n\n" << info << std::endl;

        // Run once
        test();

        std::clog << green("\nPASS\n");
        return true;
    }

private:
    static const Model& adult() { static const Model A = Model::adult(SequenceLength); return A; }

    static inline const std::vector<std::tuple<std::string, test_repetitions, void(*)()>> testbed =
    {
        {
            // All instances of a given model type begin transitioning from an identical initial configuration.
            "#1 Uninformed start", 
            RepeatOnce,
            []() {
                Model A, B;

                ASSERT(A == B);				                            // A_0 == B_0
            }
        },
        {
            // Model evolution is deterministic with respect to input.
            "#2 Determinism", 
            RepeatForever,
            []() {
                Model A, B;
                for (size_t i = 0; i < SimulatedInfinity; ++i) {
                    const Input x = random<Input>();

                    A << x;
                    B << x;

                    ASSERT(A == B);
                }
            }
        },
        {
            // Each input leaves a permanent internal trace.
            "#3 Trace",
            RepeatForever,
            []() {
                const Input x = random<Input>();
                const InputSequence start_A = { x, x };                         // violation to ARP(#5)
                const InputSequence start_B = { x, Input() };                   // one input apart
                const InputSequence pattern(InputSequence::trivial, SequenceLength);

                Model A, B;
                A << start_A;
                B << start_B;

                // the same long life for both, in which both beginnings recur
                for (size_t i = 0; i < SimulatedInfinity; ++i) {
                    A << pattern << start_A << pattern << start_B;
                    B << pattern << start_A << pattern << start_B;
                }

                ASSERT(not A.behaves_identically(B));                           // the one input still shows
            }
        },
        {
            // Model evolution depends on input order.
            "#4 Time",
            RepeatForever,
            []() {
                const Input x = random<Input>();
                Model Axy(Model::random), Ayx = Axy;
                Axy << x << ~x;
                Ayx << ~x << x;

                ASSERT(Axy != Ayx);
            }
        },
        {
            // A model can learn a cyclic sequence only if the sequence satisfies the absolute refractory-period constraint.
            "#5 Absolute refractory period",
            RepeatForever,
            []() {
                const Input x = random<Input>();
                if (x.any()) {
                    InputSequence no_consecutive_spikes = { x, ~x };
                    InputSequence consecutive_spikes = { x, x };

                    Model A, B;

                    ASSERT(A.learn(no_consecutive_spikes));
                    ASSERT(not B.learn(consecutive_spikes));
                }
            }
        },
        {
            // No model can learn everything there is to learn, except for length-2 sequences.
            "#6 Inevitable saturation",
            RepeatForever,
            []() {
                auto inevitable_saturation = [](Model& A) -> bool {
                    for (time_t attempt = 0; attempt < SimulatedInfinity; ++attempt) {
                        const auto [time, learnable_sequence] = A.learn_anything(SequenceLength);

                        if (time == Infinity)
                            return true;
                    }
                    return false;
                };
                auto universal_learnability_of_length_2_sequences = [](Model& A) -> bool {
                    InputSequence admissible_length_2_sequence(InputSequence::circular_random, 2);

                    if (not A.learn(admissible_length_2_sequence))
                        return false;
                    return true;
                };

                Model A(Model::random);

                ASSERT(inevitable_saturation(A));                                       // Requirement 6.a
                ASSERT(universal_learnability_of_length_2_sequences(A));                // Requirement 6.b
            }
        },
        {
            // The model must be able to learn sequences with varying cycle lengths.
            "#7 Temporal adaptability",
            RepeatOnce,
            []() {
                Model A;

                ASSERT(A.learn(InputSequence(InputSequence::trivial, SequenceLength)));
                ASSERT(A.learn(InputSequence(InputSequence::trivial, SequenceLength + 1)));
            }
        },
        {
            // Adaptation time is input dependent.
            "#8 Content sensitivity",
            RepeatForever,
            []() {
                // Null Hypothesis: Adaptation time is independent of the input sequence content
                auto adaptation_time_is_input_dependent = []() -> bool {
                    Model A;
                    const auto [A_time, base_seq] = A.learn_anything(SequenceLength);
                    for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                        InputSequence seq(InputSequence::circular_random, SequenceLength);

                        Model B;
                        time_t B_time = B.time_to_learn(seq);
                        if (B_time < Infinity and A_time != B_time)
                            return true;
                    }
                    return false;
                };

                ASSERT(adaptation_time_is_input_dependent());
            }
        },
        {
            // Adaptation time is model dependent.
            "#9 Context sensitivity",
            RepeatForever,
            []() {
                // Null Hypothesis: Adaptation time is independent of the model
                auto adaptation_time_is_model_dependent = []() -> bool {
                    Model A;
                    const auto [A_time, seq] = A.learn_anything(SequenceLength);
                    for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                        Model B(Model::random); 

                        time_t B_time = B.time_to_learn(seq);
                        if (B_time < Infinity and A_time != B_time)
                            return true;
                    }
                    return false;
                };

                ASSERT(adaptation_time_is_model_dependent());
            }
        },
        {
            // An informed model consistently outperforms its uninformed self at predicting corrupted inputs.
            "#10 Denoising",
            RepeatForever,
            []() {
                // a familiar sequence, and the same sequence with one admissible bit of its last input flipped
                auto familiar_and_noisy = []() {
                    InputSequence reality;
                    Input flippable;                                        // bits whose flip keeps the sequence admissible
                    do {
                        reality = InputSequence(InputSequence::circular_random, SequenceLength);
                        flippable = reality.back() | ~(reality[SequenceLength - 2] | reality.front());
                    } while (flippable.none());

                    size_t bit;
                    do bit = utils::random(0uz, BitsPerInput - 1); while (not flippable[bit]);

                    InputSequence noisy = reality;
                    noisy.back().flip(bit);
                    return std::make_pair(reality, noisy);
                };

                size_t informed_score = 0, uninformed_score = 0;
                for (int i = 0; i < 20; ++i) {
                    InputSequence reality, noisy;
                    Model informed;
                    do {                                                    // a sequence the adult can learn
                        informed = adult();
                        std::tie(reality, noisy) = familiar_and_noisy();
                    } while (not informed.learn(reality));

                    Model uninformed = adult();
                    informed << noisy;
                    uninformed << noisy;

                    informed_score += informed() == reality[0];
                    uninformed_score += uninformed() == reality[0];
                }
                ASSERT(informed_score > uninformed_score);
            }
        },
        {
            "#11 Generalisation",
            RepeatOnce,
            []() {
                const auto [world, prefix, continuation] = world_generator<Input>::generate();

                Model informed, uninformed;

                ASSERT(informed.master(world));                     // learn the world's rules, for good
                
                informed << prefix;
                uninformed << prefix;

                ASSERT(informed() == continuation);
                ASSERT(uninformed() != continuation);
            }
        },
        {
            // Each model update completes within a fixed wall-clock time bound, independent of the input history.
            // 
            // Here, a model is considered to exhibit real-time liveness if its measured update time does not exhibit 
            // significant scaling with input-history length.
            "#12 Real-time liveness",
            RepeatForever,
            []() {
                // Measure a batch of updates instead of a single update to reduce timing noise.
                auto autotune_batch_size = [=]() -> size_t {
                    const time_t min_batch_duration_us = 100;

                    InputSequence batch(InputSequence::circular_random, 2uz);
                    while (true) {
                        Model M;
                        time_t time = utils::time_it([&]() { M << batch; });
                        if (time > min_batch_duration_us)
                            return batch.size();
                        batch = InputSequence(InputSequence::circular_random, 2 * batch.size());
                    }
                };

                const size_t batch_size = autotune_batch_size();
                const InputSequence timed_batch(InputSequence::circular_random, batch_size);

                // the fastest of several timings of the same batch, each on a copy, so that a single interruption
                // by the operating system cannot decide the test
                auto batch_time = [&](const Model& M) {
                    time_t fastest = Infinity;
                    for (int i = 0; i < 5; ++i) {
                        Model C = M;
                        fastest = std::min(fastest, utils::time_it([&]() { C << timed_batch; }));
                    }
                    return fastest;
                };

                Model M;
                time_t batch_time_pre = batch_time(M);

                // a long random history (can violate #5 ARP)
                for (size_t i = 0; i < SimulatedInfinity; ++i) 
                    M << InputSequence(InputSequence::circular_random, batch_size);

                time_t batch_time_post = batch_time(M);

                double R = static_cast<double>(batch_time_post) / batch_time_pre;
                double R_tolerated = 2;   // tolerate 100% noise, caching, allocation, etc. (R should be close to 1)

                ASSERT(R < R_tolerated);
            } 
        }
    };
};
}
}
