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

// Artificial General Intelligence TestBed
template <typename SystemUnderEvaluation, size_t BitsPerInput=20, size_t SequenceLength=7>
    requires utils::InputPredictor<SystemUnderEvaluation, std::bitset<BitsPerInput>>
class TestBed
{
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
        const std::string go_back(20, '\b');
        for (const auto& [info, repetitions, test] : testbed) {
            std::clog << info << "  " << std::endl;

            const size_t test_repetitions = repetitions_override == 0 ? (size_t)repetitions : std::min((size_t)repetitions, repetitions_override);
            for (size_t r = 1; r <= test_repetitions; ++r) {
                std::clog << r << '/' << test_repetitions << "   " << go_back;

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
        std::clog << "\nRunning the test #" << test_number;
        std::clog << "\nRandom seed: " << rng_seed;
        std::clog << "\n\n" << info << std::endl;

        // Run once
        test();

        std::clog << green("\nPASS\n");
        return true;
    }
            
private:
    static inline const auto all_distinct_inputs = std::views::iota(0, 1 << BitsPerInput)
        | std::views::transform([](int i) { return Input(i); });
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
                Model A;
                std::vector<Model> trajectory;
                trajectory.reserve(SimulatedInfinity);

                // simplest edge case
                A << Input{};
                trajectory.push_back(A);
                A << Input{};
                ASSERT(A != trajectory.back());

                // general behaviour
                while (trajectory.size() < SimulatedInfinity) {
                    trajectory.push_back(A);
                    A << random<Input>();

                    ASSERT(std::find(trajectory.begin(), trajectory.end(), A) == trajectory.end());
                }
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
            // A model cannot learn everything there is to learn, except for length-2 sequences.
            "#6 Inevitable saturation",
            RepeatForever,
            []() {
                auto inevitable_saturation = [](Model& A) -> bool {
                    for (time_t time = 0; time < SimulatedInfinity; ++time) {
                        InputSequence learnable_sequence = Model::learnable_random_sequence(SequenceLength);

                        if (not A.learn(learnable_sequence))
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

                Model A;

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
                    const InputSequence base_seq = Model::learnable_random_sequence(SequenceLength);
                    const time_t A_time = A.time_to_learn(base_seq);
                    for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                        InputSequence seq(InputSequence::circular_random, SequenceLength);          // admissible by construction

                        if (seq != base_seq) {
                            Model B;
                            time_t B_time = B.time_to_learn(seq);
                            if (B_time < Infinity and A_time != B_time)
                                return true;
                        }
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
                    const InputSequence seq = Model::learnable_random_sequence(SequenceLength);     // A_time < Infinity
                    Model A;
                    const time_t A_time = A.time_to_learn(seq);
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
                // flips one random bit of x, keeping the sequence admissible
                auto corrupt = [](Input x, const Input& x_prev, const Input& x_next) -> std::optional<Input> {
                    const Input flippable = x | ~(x_prev | x_next);
                    if (flippable.none())
                        return std::nullopt;
                    size_t bit;
                    do bit = utils::random(0uz, BitsPerInput - 1); while (not flippable[bit]);
                    return x.flip(bit);
                };
                // an informed model: an adult that has reached its capacity (#6a) and then lived as long again,
                // but never shorter than a minimal life, so that a model cannot shorten its own test by failing early
                static const Model adult = []() {
                    const auto rng_state = utils::rng;
                    utils::rng.seed();                                      // the same life in every run keeps failures reproducible
                    Model M;
                    time_t youth = 0;                                       // sequences learned before the first failure
                    while (youth < SimulatedInfinity and M.learn(Model::learnable_random_sequence(SequenceLength)))
                        ++youth;
                    const time_t lifetime = std::max(50uz, 2 * youth);
                    for (time_t time = youth + 1; time < lifetime; ++time)  // successful or not
                        M.learn(Model::learnable_random_sequence(SequenceLength));
                    utils::rng = rng_state;
                    return M;
                }();
                size_t informed_score = 0, uninformed_score = 0;
                const int num_of_runs = 20;                                 // within each of 5,000 trials
                for (int i = 0; i < num_of_runs; ++i) {
                    const InputSequence reality(InputSequence::circular_random, SequenceLength);
                    const Input true_elt = reality[0];
                    if (const auto corrupted_elt = corrupt(reality.back(), reality[SequenceLength - 2], reality[0])) {
                        Model informed = adult, uninformed = adult;
                        informed << reality << reality;                     // the minimal stream that reveals the cycle

                        // the noisy input is the most recent context, so the familiar pattern must be recognised despite the noise
                        const auto noisy_pass = [&](Model& M) { M << (reality | std::views::take(SequenceLength - 1)) << *corrupted_elt; };
                        noisy_pass(informed);
                        noisy_pass(uninformed);

                        informed_score += utils::match_score(informed.get_prediction(), true_elt);
                        uninformed_score += utils::match_score(uninformed.get_prediction(), true_elt);
                    }
                    else
                        i -= 1;
                }
                ASSERT(informed_score > uninformed_score);
            }
        },
        {
            // Each model update completes within a fixed wall-clock time bound, independent of the input history.
            // 
            // Here, a model is considered to exhibit real-time liveness if its measured update time does not exhibit 
            // significant scaling with input-history length.
            "#11 Real-time liveness",
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

                Model M;
                time_t batch_time_pre = utils::time_it([&]() { M << timed_batch; });
                    
                // a long random history (can violate #5 ARP)
                for (size_t i = 0; i < SimulatedInfinity; ++i) 
                    M << InputSequence(InputSequence::circular_random, batch_size);

                time_t batch_time_post = utils::time_it([&]() { M << timed_batch; });

                double R = static_cast<double>(batch_time_post) / batch_time_pre;
                double R_tolerated = 2;   // tolerate 100% noise, caching, allocation, etc. (R should be close to 1)  
                    
                ASSERT(R < R_tolerated);
            } 
        }
    };
};
}
}
