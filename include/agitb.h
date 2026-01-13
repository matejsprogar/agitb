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
                            (std::cerr << std::format("\n{} in {}:{}\n{}\n\n", red("\nAssertion failed"), __FILE__, __LINE__, #expression), \
                            exit(-1), 0))

    namespace AGI {
        // AGITB environment settings
        const size_t SimulatedInfinity = 5000;

        // AGITB settings : temporal patterns with seven inputs of ten bits each
        const time_t SequenceLength = 7;        // N
        const size_t BitsPerInput = 10;         // L
        enum number_of_competent_trials { RepeatOnce = 1, Repeat100x = 100, RepeatForever = SimulatedInfinity };
        enum test_mode { competent = 0, simple = 1, fast = Repeat100x };

        static_assert(SequenceLength > 1);
        static_assert(BitsPerInput > 1);

        template <typename SystemUnderEvaluation>
        class TestBed
        {
            using Input = std::bitset<BitsPerInput>;
            using InputSequence = utils::InputSequence<Input>;
            using Model = utils::Model<SystemUnderEvaluation, Input>;

        public:
            static bool run(test_mode _mode = competent)
            {
                std::clog << "Artificial General Intelligence Testbed\n";
                std::clog << "Random seed: " << random_seed << std::endl << std::endl;

                const std::string go_back(10, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;

                    const size_t T = trials(_mode, repetitions);
                    for (size_t t = 1; t <= T; ++t) {
                        std::clog << t << '/' << T << go_back;

                        test();
                    }
                }

                std::clog << green("\nPASS\n");
                return true;
            }
            static bool run(int test_id, int trials = 1)
            {
                assert(test_id > 0 and test_id <= testbed.size());
                
                std::clog << "Artificial General Intelligence Testbed\n";
                std::clog << "Random seed: " << random_seed << std::endl << std::endl;

                const std::string go_back(10, '\b');
                const auto& [info, repetitions, test] = testbed[test_id-1];
                
                std::clog << info << std::endl;
                for (int t = 1; t <= trials; ++t) {
                    std::clog << t << '/' << trials << go_back;

                    test();
                }

                std::clog << green("\nPASS\n");
                return true;
            }
            
        private:
            static inline const auto all_distinct_inputs = std::views::iota(0, 1 << BitsPerInput)
                | std::views::transform([](int i) { return Input(i); });
            static inline size_t trials(test_mode _mode, size_t repetitions) { return _mode == competent ? repetitions : std::min(repetitions, (size_t)_mode); }
            static inline const std::vector<std::tuple<std::string, number_of_competent_trials, void(*)()>> testbed =
            {
                {
                    "#1 Uninformed start (All instances of a given model type begin transitioning from an identical initial configuration.)",
                    Repeat100x,
                    []() {
                        Model A, B;

                        ASSERT(A == B);				                    // A_0 == B_0
                    }
                },
                {
                    "#2 Determinism (Model evolution is deterministic with respect to input.)",
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
                    "#3 Trace (Each input leaves a permanent internal trace.)",
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
                    "#4 Time (Model evolution depends on input order.)",
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
                    "#5 Absolute refractory period (A model can learn a cyclic sequence only if the sequence satisfies the absolute refractory-period constraint.)",
                    RepeatOnce,
                    []() {
                        for (const Input x : all_distinct_inputs) {
                            const InputSequence no_consecutive_spikes = { x, ~x };
                            const InputSequence consecutive_spikes = { x, x };
                            const bool has_spike = x.any();

                            Model A, B;

                            ASSERT(A.learn(no_consecutive_spikes, SimulatedInfinity));
                            ASSERT(not B.learn(consecutive_spikes, SimulatedInfinity) || !has_spike);
                        }
                    }
                },
                {
                    "#6 Limited learnability (No model can learn everything there is to learn, except for length-2 cases.)",
                    RepeatForever,
                    []() {
                        auto limited_learnability = [](Model& A) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                InputSequence admissible_sequence(InputSequence::circular_random, SequenceLength);

                                if (not A.learn(admissible_sequence, SimulatedInfinity))
                                    return true;
                            }
                            return false;
                        };
                        auto unlimited_learnability_on_length_2_cases = [](const Model& A) -> bool {
                            auto admissible = [](const Input& x1, const Input& x2) -> bool { return not (x1 & x2).any(); };

                            for (const Input& x1 : all_distinct_inputs) {
                                for (const Input& x2 : all_distinct_inputs) {
                                    if (!admissible(x1, x2))
                                        continue;

                                    const InputSequence admissible_length_2_case = { x1, x2 };
                                    Model B = A;
                                    if (!B.learn(admissible_length_2_case, SimulatedInfinity))
                                        return false;
                                }
                            }
                            return true;
                        };

                        Model A;

                        ASSERT(limited_learnability(A));                            // Axiom 6.a
                        ASSERT(unlimited_learnability_on_length_2_cases(A));        // Axiom 6.b
                    }
                },
                {
                    "#7 Temporal adaptability (The model must be able to learn sequences with varying cycle lengths.)",
                    RepeatOnce,
                    []() {
                        const InputSequence phi1(InputSequence::trivial, SequenceLength);
                        const InputSequence phi2(InputSequence::trivial, SequenceLength + 1);
                        Model A;

                        ASSERT(A.learn(phi1, SimulatedInfinity));
                        ASSERT(A.learn(phi2, SimulatedInfinity));
                    }
                },
                {
                    "#8 Content sensitivity (Adaptation time is input dependent.)",
                    RepeatForever,
                    []() {
                        // Null Hypothesis: Adaptation time is independent of the input sequence content
                        auto adaptation_time_is_input_dependent = []() -> bool {
                            Model B;
                            const InputSequence phi1 = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
                            const time_t phi1_time = B.time_to_learn(phi1, SimulatedInfinity);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                const InputSequence phi2(InputSequence::circular_random, SequenceLength);   // admissible by construction

                                if (phi2 != phi1) {
                                    Model A;
                                    time_t phi2_time = A.time_to_learn(phi2, SimulatedInfinity);
                                    const bool phi2_learnable = phi2_time < SimulatedInfinity;
                                    if (phi2_learnable and phi1_time != phi2_time)                          // rejects the null hypothesis
                                        return true;
                                }
                            }
                            return false;
                        };

                        ASSERT(adaptation_time_is_input_dependent());
                    }
                },
                {
                    "#9 Context sensitivity (Adaptation time is model dependent.)",
                    RepeatForever,
                    []() {
                        // Null Hypothesis: Adaptation time is independent of the model
                        auto adaptation_time_is_model_dependent = []() -> bool {
                            const InputSequence phi = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
                            Model A;
                            const time_t A_time = A.time_to_learn(phi, SimulatedInfinity);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                Model B(Model::random, 1 + utils::random_warm_up_time(SimulatedInfinity));  // B != A by construction

                                time_t B_time = B.time_to_learn(phi, SimulatedInfinity);
                                if (A_time != B_time)                                                       // rejects the null hypothesis
                                    return true;
                            }
                            return false;
                        };

                        ASSERT(adaptation_time_is_model_dependent());
                    }
                },
                {
                    "#10 Denoising (An informed model outperforms the best constant baseline at denoising the corrupted input.)",
                    RepeatForever,
                    []() {
                        auto corruption = [](const Input& x0, const Input& x1, const Input& xk) -> Input {
                            Input x;
                            do {
                                x = random<Input>(x1, xk);              // respect Axiom 6
                            } while (x == x0);                          // ensure corruption
                            return x;
                        };
                        const Input all_zeros = Input{}, all_ones = ~all_zeros;
                        size_t model_score = 0, baseline_0_score = 0, baseline_1_score = 0;
                        const int num_of_experiments = 20;              // within each of 5,000 trials
                        const int n = 5 * SequenceLength;               // informing context length
                        for (int i = 0; i < num_of_experiments; ++i) {
                            const InputSequence phi(InputSequence::circular_random, SequenceLength);
                            const Input x1_corrupted = corruption(phi[0], phi[1], phi.back());

                            Model A;
                            for (int j = 0; j < n; ++j)
                                A << phi;                                                   // A << phi^n

                            A << x1_corrupted << (phi | std::views::drop(1));               // A << phi'

                            const Input& x1 = phi.front();
                            model_score += utils::match_score(A(), x1);
                            baseline_0_score += utils::match_score(all_zeros, x1);
                            baseline_1_score += utils::match_score(all_ones, x1);
                        }

                        ASSERT(model_score > std::max(baseline_0_score, baseline_1_score));
                    }
                },
                {
                    "#11 Generalisation (The model performs above chance on previously unseen inputs.)",
                    RepeatForever,
                    []() {
                        size_t score = 0;
                        const int num_of_experiments = 20, k = 10;
                        for (int i = 0; i < num_of_experiments; ++i) {
                            Model phi_generator(Model::random, SimulatedInfinity);          // unknown random rule
                            const auto phi1 = phi_generator.generate(k * SequenceLength);   // prefix
                            const auto phi2 = phi_generator.generate(1 * SequenceLength);   // continuation

                            Model A;
                            A << phi1;

                            const auto phi2_star = A.generate(phi2.size());
                            score += utils::match_score(phi2_star, phi2);
                        }
                        // phi2.size() == SequenceLength, L = BitsPerInput
                        const size_t total_bits = num_of_experiments * SequenceLength * BitsPerInput;
                        const size_t random_guess = total_bits / 2;

                        ASSERT(score > random_guess);
                    }
                },
                {
                    "#12 Real-time liveness (Each model update completes within a uniform time bound.)",
                    RepeatForever,
                    []() {
                        auto batch_update_time = [](Model& model, const InputSequence& batch) -> time_t {
                            const auto start = std::chrono::steady_clock::now();

                            model << batch;

                            const auto end = std::chrono::steady_clock::now();
                            return (time_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
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
