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

#define ASSERT(expression) (void)((!!(expression)) || \
                            (std::cerr << std::format("\n\n{} in {}:{}\n{}\n\nrng_seed: {}\n", \
                                red("Assertion failed"), __FILE__, __LINE__, #expression, utils::rng_seed), \
                            exit(-1), 0))

    namespace AGI {
        // AGITB environment settings
        const size_t SimulatedInfinity = 5000;

        // AGITB settings : temporal patterns with seven inputs of ten bits each
        const size_t BitsPerInput = 10;         // L
        const time_t SequenceLength = 7;        // N
        static_assert(SequenceLength > 1);
        static_assert(BitsPerInput > 1);



        // Artificial General Intelligence TestBed
        template <typename SystemUnderEvaluation>
            requires utils::InputPredictor<SystemUnderEvaluation, std::bitset<BitsPerInput>>
        class TestBed
        {
            using Input = std::bitset<BitsPerInput>;
            using InputSequence = utils::InputSequence<Input>;
            using Model = utils::Model<SystemUnderEvaluation, Input>;

            enum test_repetitions { RepeatOnce = 1, Repeat100x = 100, RepeatForever = SimulatedInfinity };

        public:
            // Runs all tests from the testbed using the specified test mode.
            static bool run(size_t repetitions_override = 0)
            {
                std::clog << "Artificial General Intelligence Testbed\n\nRunning 12 tests...\n";

                const std::string go_back(10, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;

                    const size_t test_repetitions = repetitions_override == 0 ? repetitions : std::min((size_t)repetitions, (size_t)repetitions_override);
                    for (size_t r = 1; r <= test_repetitions; ++r) {
                        std::clog << r << '/' << test_repetitions << go_back;

                        utils::rng_seed = utils::rng();
                        test();
                    }
                }

                std::clog << green("\nPASS\n");
                return true;
            }
            // Runs a specified test from the testbed using the given RNG seed.
            static bool run(unsigned test_number, unsigned seed)
            {
                utils::rng.seed(utils::rng_seed = seed);
                ASSERT(test_number > 0 and test_number <= testbed.size());
                
                const auto& [info, repetitions, test] = testbed[test_number -1];

                std::clog << "Artificial General Intelligence Testbed\nRunning 1 test:\n";
                std::clog << "Random seed: " << rng_seed << std::endl << std::endl;
                std::clog << info << std::endl;

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
                    Repeat100x,
                    []() {
                        Model A, B;

                        ASSERT(A == B);				                    // A_0 == B_0
                    }
                },
                {
                    // Model evolution is deterministic with respect to input.
                    "#2 Determinism", 
                    Repeat100x,
                    []() {
                        const InputSequence warm_up(InputSequence::random, utils::random_warm_up_time(SimulatedInfinity));
                        Model A, B;

                        A << warm_up;
                        B << warm_up;                                   // B = A would allow RNG duplication.

                        for (const Input& x : all_distinct_inputs) {
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
                        Model A(Model::random, utils::random_warm_up_time(SimulatedInfinity));

                        std::vector<Model> trajectory;
                        trajectory.reserve(SimulatedInfinity);

                        trajectory.push_back(A);
                        while (trajectory.size() < SimulatedInfinity) {
                            A << random<Input>();

                            ASSERT(std::find(trajectory.begin(), trajectory.end(), A) == trajectory.end());
                            trajectory.push_back(A);
                        }
                    }
                },
                {
                    // Model evolution depends on input order.
                    "#4 Time",
                    Repeat100x,
                    []() {
                        for (const Input& x : all_distinct_inputs) {
                            Model A(Model::random, utils::random_warm_up_time(SimulatedInfinity)), B = A;
                            A << x << ~x;
                            B << ~x << x;

                            ASSERT(A != B);
                        }
                    }
                },
                {
                    // A model can learn a cyclic sequence only if the sequence satisfies the absolute refractory-period constraint.
                    "#5 Absolute refractory period",
                    RepeatOnce,
                    []() {
                        for (const Input x : all_distinct_inputs) {
                            const InputSequence no_consecutive_spikes = { x, ~x };
                            const InputSequence consecutive_spikes = { x, x };
                            const bool spikes = x.any();

                            Model A, B;

                            ASSERT(A.learn(no_consecutive_spikes, SimulatedInfinity));
                            ASSERT(not B.learn(consecutive_spikes, SimulatedInfinity) || !spikes);
                        }
                    }
                },
                {
                    // A model cannot learn everything there is to learn, except for length-2 sequences.
                    "#6 Saturation",
                    RepeatForever,
                    []() {
                        auto inevitable_saturation = [](Model& A) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                const InputSequence learnable_sequence = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);

                                if (not A.learn(learnable_sequence, SimulatedInfinity))
                                    return true;
                            }
                            return false;
                        };
                        auto universal_learnability_of_admissible_length_2_sequences = [](const Model& A) -> bool {
                            auto admissible = [](const Input& x1, const Input& x2) -> bool { return not (x1 & x2).any(); };

                            for (const Input& x1 : all_distinct_inputs) {
                                for (const Input& x2 : all_distinct_inputs) {
                                    if (!admissible(x1, x2))
                                        continue;

                                    const InputSequence admissible_length_2_sequence = { x1, x2 };
                                    Model B = A;
                                    if (!B.learn(admissible_length_2_sequence, SimulatedInfinity))
                                        return false;
                                }
                            }
                            return true;
                        };

                        Model A;

                        ASSERT(inevitable_saturation(A));                                       // Requirement 6.a
                        ASSERT(universal_learnability_of_admissible_length_2_sequences(A));     // Requirement 6.b
                    }
                },
                {
                    // The model must be able to learn sequences with varying cycle lengths.
                    "#7 Temporal adaptability",
                    RepeatOnce,
                    []() {
                        const InputSequence seq1(InputSequence::trivial, SequenceLength);       // 00....01
                        const InputSequence seq2(InputSequence::trivial, SequenceLength + 1);   // 00....001    
                        Model A;

                        ASSERT(A.learn(seq1, SimulatedInfinity));
                        ASSERT(A.learn(seq2, SimulatedInfinity));
                    }
                },
                {
                    // Adaptation time is input dependent.
                    "#8 Content sensitivity",
                    RepeatForever,
                    []() {
                        // Null Hypothesis: Adaptation time is independent of the input sequence content
                        auto adaptation_time_is_input_dependent = []() -> bool {
                            Model B;
                            const InputSequence base_seq = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
                            const time_t seq_time = B.time_to_learn(base_seq, SimulatedInfinity);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                const InputSequence seq(InputSequence::circular_random, SequenceLength); // admissible by construction

                                if (seq != base_seq) {
                                    Model A;
                                    const time_t seq_time = A.time_to_learn(seq, SimulatedInfinity);
                                    const bool seq_learnable = seq_time != SimulatedInfinity;
                                    if (seq_learnable and seq_time != seq_time)                          // rejects the null hypothesis
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
                            const InputSequence seq = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
                            Model A;
                            const time_t A_time = A.time_to_learn(seq, SimulatedInfinity);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                Model B(Model::random, 1 + utils::random_warm_up_time(SimulatedInfinity));  // B != A by construction

                                time_t B_time = B.time_to_learn(seq, SimulatedInfinity);
                                if (A_time != B_time)                                                       // rejects the null hypothesis
                                    return true;
                            }
                            return false;
                        };

                        ASSERT(adaptation_time_is_model_dependent());
                    }
                },
                {
                    // An informed model consistently outperforms any constant baseline at predicting corrupted inputs.
                    "#10 Denoising",
                    RepeatForever,
                    []() {
                        auto corrupt = [](const Input& x, const Input& x_next, const Input& x_prev) -> Input {
                            Input x_corrupted;
                            do {
                                x_corrupted = random<Input>(x_next, x_prev);  // respect Requirement 6
                            } while (x_corrupted == x);                       // ensure corruption
                            return x_corrupted;
                        };
                        const Input all_zeros = Input{}, all_ones = ~all_zeros;
                        size_t model_score = 0, baseline_0_score = 0, baseline_1_score = 0;
                        const int num_of_runs = 20;                     // within each of 5,000 trials
                        const int n = 5 * SequenceLength;               // informing context length
                        for (int i = 0; i < num_of_runs; ++i) {
                            const InputSequence seq(InputSequence::circular_random, SequenceLength);
                            const Input x_corrupted = corrupt(seq[0], seq[1], seq.back());

                            Model A;
                            for (int j = 0; j < n; ++j)
                                A << seq;                                                   // A << seq^n

                            A << x_corrupted << (seq | std::views::drop(1));                // A << seq'

                            const Input& x_true = seq.front();
                            model_score += utils::match_score(A.get_prediction(), x_true);
                            baseline_0_score += utils::match_score(all_zeros, x_true);
                            baseline_1_score += utils::match_score(all_ones, x_true);
                        }

                        ASSERT(model_score > std::max(baseline_0_score, baseline_1_score));
                    }
                },
                {
                    // After training on a prefix of a structured sequence, a model predicts previously unseen continuations better than chance.
                    "#11 Generalisation",
                    RepeatForever,
                    []() {
                        size_t score = 0;
                        const int num_of_runs = 20, ratio = 10;                                     // |prefix| = ratio * |continuation|
                        for (int i = 0; i < num_of_runs; ++i) {
                            Model G(Model::random, SimulatedInfinity);                              // unknown random rule
                            const auto prefix = G.generate(ratio * SequenceLength);
                            const auto continuation = G.generate(1 * SequenceLength);

                            Model A;
                            A << prefix;

                            const auto continuation_star = A.generate(continuation.size());
                            score += utils::match_score(continuation_star, continuation);
                        }
                        // total_bits = num_of_runs * continuation.size() * L
                        const size_t total_bits = num_of_runs * SequenceLength * BitsPerInput;
                        const size_t random_chance = total_bits / 2;

                        ASSERT(score > random_chance);
                    }
                },
                {
                    // Each model update completes within a fixed wall-clock time bound, independent of the input history.
                    "#12 Real-time liveness",
                    RepeatForever,
                    []() {
                        auto batch_update_time = [](Model& model, const InputSequence& batch) -> time_t {
                            const auto start = std::chrono::steady_clock::now();

                            model << batch;

                            const auto stop = std::chrono::steady_clock::now();
                            return (time_t)std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
                        };
                        auto autotune_batch_size = [&](const Model& model) -> size_t {
                            const time_t target_batch_duration_us = 100;
                            InputSequence batch(InputSequence::random, 1);
                            while (batch.size() < 1'000'000) {
                                Model _model = model;
                                if (batch_update_time(_model, batch) >= target_batch_duration_us)
                                    break;
                                batch = InputSequence(InputSequence::random, 2 * batch.size());
                            }
                            return batch.size();
                        };
                        auto measure_times = [&](const size_t num_batches, const size_t batch_size) {
                            std::vector<time_t> blank_times, complex_times;
                            blank_times.reserve(num_batches);
                            complex_times.reserve(num_batches);

                            const Model blank, complex(Model::random, SimulatedInfinity);

                            const size_t structured_batches = num_batches / 4;      // structured:random = 1:4
                            for (size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
                                const InputSequence batch = blank_times.size() < structured_batches ?
                                    InputSequence(InputSequence::structured, batch_size, batch_id) :
                                    InputSequence(InputSequence::random, batch_size);

                                Model _blank = blank;
                                blank_times.emplace_back(batch_update_time(_blank, batch));

                                Model _complex = complex;
                                complex_times.emplace_back(batch_update_time(_complex, batch));
                            }

                            return std::make_pair(blank_times, complex_times);
                        };

                        const size_t num_of_batches = 100;
                        const size_t batch_size = std::max(autotune_batch_size(Model()), autotune_batch_size(Model(Model::random, SimulatedInfinity)));
                        const auto [blank_times, complex_times] = measure_times(num_of_batches, batch_size);

                        const time_t absolute_non_liveness_guard = 10 * utils::median(blank_times);
                        ASSERT(*std::ranges::max_element(blank_times) <= absolute_non_liveness_guard);
                        ASSERT(*std::ranges::max_element(complex_times) <= absolute_non_liveness_guard);
                        ASSERT(not utils::consistently_greater_second_value(blank_times, complex_times));
                    }
                }
            };
        };
    }
}
