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
        const size_t Repeat100x = 100;
        const size_t RepeatForever = SimulatedInfinity;
        const size_t RepeatOnce = 1;
        enum mode { exhaustive = 0, fast = 1 };

        template <typename SystemUnderEvaluation>
        class TestBed
        {
            using Input = std::bitset<BitsPerInput>;
            using InputSequence = utils::InputSequence<Input>;
            using Model = utils::Model<SystemUnderEvaluation, Input>;

        public:
            static bool run(mode _mode = exhaustive)
            {
                std::clog << "Artificial General Intelligence Testbed\n";
                std::clog << "Random seed: " << random_seed << std::endl << std::endl;

                const std::string go_back(10, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;

                    const size_t R = repeats(_mode, repetitions);
                    for (size_t i = 1; i <= R; ++i) {
                        std::clog << i << '/' << R << go_back;

                        test();
                    }
                }

                std::clog << green("\nPASS\n");
                return true;
            }

        private:
            static inline const auto all_distinct_inputs = std::views::iota(0, 1 << BitsPerInput)
                | std::views::transform([](int i) { return Input(i); });
            static inline size_t repeats(mode _mode, size_t repetitions) { return _mode == exhaustive ? repetitions : std::min(repetitions, Repeat100x); }
            static inline const std::vector<std::tuple<std::string, size_t, void(*)()>> testbed =
            {
                {
                    "#1 Uninformed start (All instances of a given model type begin transitioning from an identical initial configuration.)",
                    Repeat100x,
                    []() {
                        Model A;

                        ASSERT(A == Model{});				        // A_0 == B_0
                    }
                },
                {
                    "#2 Perpetual change (Every input modifies the model configuration.)",
                    Repeat100x,
                    []() {
                        const time_t warm_up = utils::random(SimulatedInfinity);
                        Model A(Model::random, warm_up);

                        for (const Input& x : all_distinct_inputs) {
                            Model A_prev = A;
                            A << x;

                            ASSERT(A != A_prev);                       // A_1 != A_0
                        }
                    }
                },
                {
                    "#3 Determinism (Model evolution is deterministic with respect to input.)",
                    Repeat100x,
                    []() {
                        const InputSequence warm_up(InputSequence::random, utils::random(SimulatedInfinity));
                        Model A, B;

                        A << warm_up;   
                        B << warm_up;       // B = A would allow RNG duplication.

                        for (const Input& x : all_distinct_inputs) {
                            A << x;
                            B << x;

                            ASSERT(A == B);
                        }
                    }
                },
                {
                    "#4 Trace (Each input leaves a permanent internal trace.)",
                    RepeatForever,
                    []() {
                        Model A(Model::random, SequenceLength);

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
                    "#5 Time (Model evolution depends on input order.)",
                    Repeat100x,
                    []() {
                        for (const Input& x : all_distinct_inputs) {
                            Model A(Model::random, SequenceLength), B = A;
                            A << x << ~x;
                            B << ~x << x;

                            ASSERT(A != B);
                        }
                    }
                },
                {
                    "#6 Absolute refractory period (A model can learn a cyclic sequence only if the sequence satisfies the absolute refractory-period constraint.)",
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
                    "#7a Limited learnability (No model can learn everything there is to learn.)",
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

                        Model A;
                        ASSERT(limited_learnability(A));                            // The model has limited learnability.
                    }
                },
                {
                    "#7b Limited learnability (All admissible length-2 sequences are universally learnable.)",
                    RepeatForever,
                    []() {
                        auto admissible = [](const Input & x1, const Input & x2) -> bool {
                            return not (x1 & x2).any();                             // absolute refractory-period constraint
                        };

                        const time_t warm_up = utils::random(SimulatedInfinity);
                        const Model base(Model::random, warm_up);                   // a reachable configuration in Model space
                        for (const Input& x1 : all_distinct_inputs) {
                            for (const Input& x2 : all_distinct_inputs) {
                                if (!admissible(x1, x2))
                                    continue;

                                InputSequence length_2_sequence = { x1, x2 };
                                Model A = base;
                                ASSERT(A.learn(length_2_sequence, SimulatedInfinity));
                            }
                        }
                    }
                },
                {
                    "#8 Temporal adaptability (The model must be able to learn sequences with varying cycle lengths.)",
                    RepeatOnce,
                    []() {
                        const InputSequence trivial_problem(InputSequence::trivial, SequenceLength);
                        const InputSequence longer_trivial_problem(InputSequence::trivial, SequenceLength + 1);
                        Model A;

                        ASSERT(A.learn(trivial_problem, SimulatedInfinity));
                        ASSERT(A.learn(longer_trivial_problem, SimulatedInfinity));
                    }
                },
                {
                    "#9 Content sensitivity (Adaptation time is input dependent.)",
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
                    "#10 Context sensitivity (Adaptation time is model dependent.)",
                    RepeatForever,
                    []() {
                    // Null Hypothesis: Adaptation time is independent of the model
                    auto adaptation_time_is_model_dependent = []() -> bool {
                        const InputSequence phi = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
                        Model A;
                        const time_t A_time = A.time_to_learn(phi, SimulatedInfinity);
                        for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                            Model B(Model::random, 1 + utils::random(SequenceLength));                  // B != A by construction

                            time_t B_time = A.time_to_learn(phi, SimulatedInfinity);
                            if (A_time != B_time)                                                       // rejects the null hypothesis
                                return true;
                        }
                        return false;
                    };

                    ASSERT(adaptation_time_is_model_dependent());
                }
            },
            {
                "#11 Unobservability (Distinct models may be observationally indistinguishable.)",
                RepeatForever,
                []() {
                    // Null Hypothesis: "Different models cannot produce identical behavior."
                    auto different_model_instances_can_produce_identical_behaviour = []() -> bool {
                        const InputSequence simplest_behaviour = { Input{}, Input{} };
                        for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                            Model A, B(Model::random, SequenceLength);
                            A.learn(simplest_behaviour, SimulatedInfinity);
                            B.learn(simplest_behaviour, SimulatedInfinity);

                            bool counterexample = A != B && Model::identical_behaviour(A, B, 2 * SequenceLength);
                            if (counterexample)                             // rejects the null hypothesis
                                return true;
                        }
                        return false;
                    };

                    ASSERT(different_model_instances_can_produce_identical_behaviour());
                }
            },
            {
                "#12 Denoising (The model outperforms the best trivial baseline predictor.)",
                RepeatForever,
                []() {
                    size_t model_score = 0, baseline_0_score = 0, baseline_1_score = 0;
                    const int num_of_experiments = 20, exposure_time = 5 * SequenceLength;   // plenty of time
                    for (int i = 0; i < num_of_experiments; ++i) {
                        const InputSequence seq(InputSequence::circular_random, SequenceLength);
                        const Input disruption = random<Input>(seq[1], seq.back());

                        Model A;
                        for (int i = 0; i < exposure_time; ++i)
                            A << seq;                                       // prior experience    

                        A << disruption;                                    // begin a novel situation
                        A << (seq | std::views::drop(1));

                        const Input& truth = seq.front();
                        model_score += utils::count_matching_bits(A(), truth);
                        baseline_0_score += utils::count_matching_bits(Input{}, truth);
                        baseline_1_score += utils::count_matching_bits(~Input{}, truth);
                    }

                    ASSERT(model_score > std::max(baseline_0_score, baseline_1_score));
                }
            },
            {
                "#13 Generalisation (The model performs above chance on previously unseen inputs.)",
                RepeatForever,
                []() {
                    size_t score = 0;
                    const int num_of_experiments = 20, k = 10;
                    for (int i = 0; i < num_of_experiments; ++i) {
                        Model rule_generator(Model::random, SimulatedInfinity);          // implements an unknown random rule
                        const auto train = rule_generator.generate(k * SequenceLength);  // split: first k parts for training
                        const auto truth = rule_generator.generate(1 * SequenceLength);  //        1 subsequent part for testing  

                        Model A;
                        A << train;

                        score += utils::count_matching_bits(A.generate(truth.size()), truth);
                    }
                    const size_t random_guess = num_of_experiments * SequenceLength * BitsPerInput / 2;

                    ASSERT(score > random_guess);
                }
            },
            {
                "#14 Real-time liveness (The model completes each input-driven transition within bounded time.)",
                RepeatForever,
                []() {
                    auto total_update_time = [](Model& M, const Model::InputSequence& sequence) -> size_t {
                        const auto start = std::chrono::high_resolution_clock::now();

                        M << sequence;

                        const auto end = std::chrono::high_resolution_clock::now();
                        return (size_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    };
                    auto conduct_experiments = [&](size_t num) {
                        std::vector<std::pair<size_t, size_t>> results; results.reserve(num);

                        while (results.size() < num) {
                            const InputSequence sequence(InputSequence::random, SimulatedInfinity);

                            Model A;
                            Model B(Model::random, SimulatedInfinity);

                            results.emplace_back(
                                total_update_time(A, sequence),
                                total_update_time(B, sequence)
                            );
                        }

                        return results;
                    };

                    const int num_of_experiments = 100;
                    const auto results = conduct_experiments(num_of_experiments);

                    ASSERT(not utils::consistently_greater_second_value(results));
                }
            }
        };
            };
        }
    }
