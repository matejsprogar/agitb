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



#define ASSERT(expression) (void)((!!(expression)) || \
                            (std::cerr << std::format("\n{} in {}:{}\n{}\n\n", red("\nAssertion failed"), __FILE__, __LINE__, #expression), \
                            exit(-1), 0))

namespace sprogar {
    namespace AGI {
        // AGITB environment settings
        const size_t SimulatedInfinity = 5000;

        // AGITB settings : temporal patterns with seven inputs of ten bits each
        const time_t SequenceLength = 7;        // \eta
        const size_t BitsPerInput = 10;         // \omega
        const int Repeat100x = 100;
        const int RepeatOnce = 1;

        template <typename SystemUnderEvaluation>
        class TestBed
        {
            using Input = std::bitset<BitsPerInput>;
            using InputSequence = utils::InputSequence<Input>;
            using Model = utils::Model<SystemUnderEvaluation, Input>;

        public:
            static void run()
            {
                std::clog << "Artificial General Intelligence Testbed\n\n";

                const std::string go_back(10, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;

                    for (int i = 1; i <= repetitions; ++i) {
                        std::clog << i << '/' << repetitions << go_back;

                        test();
                    }
                }

                std::clog << green("\nPASS\n");
            }

        private:
            static inline const std::vector<std::tuple<std::string, int, void(*)()>> testbed =
            {
                {
                    "#1 Bias-free start (All models begin in a completely blank, bias-free state.)",
                    Repeat100x,
                    []() {
                        Model A;

                        ASSERT(A == Model{});				    // The initial state represents absence of bias.
                        ASSERT(A.prediction() == Input{});	    // No spikes indicate an unbiased initial prediction.
                    }
                },
                {
                    "#2 Bias (A change in state indicates bias.)",
                    Repeat100x,
                    []() {
                        Model A;
                        A << random<Input>();

                        ASSERT(A != Model{});
                    }
                },
                {
                    "#3 Determinism (Identical experiences produce an identical state.)",
                    Repeat100x,
                    []() {
                        const InputSequence random_experience(InputSequence::random, SimulatedInfinity);

                        Model A, B;
                        A << random_experience;
                        B << random_experience;

                        ASSERT(A == B);
                    }
                },
                {
                    "#4 Sensitivity (Inequivalent models remain inequivalent under identical inputs.)",
                    Repeat100x,
                    []() {
                        const Input p = random<Input>();
                        const InputSequence random_experience(InputSequence::random, SimulatedInfinity);

                        Model A, B;
                        A << p << random_experience;
                        B << ~p << random_experience;

                        ASSERT(A != B);
                    }
                },
                {
                    "#5 Time (The input order is inherently temporal and crucial to the process.)",
                    Repeat100x,
                    []() {
                        const Input a = random<Input>();
                        const Input b = random<Input>(a);     // a & b == Input{}

                        Model A, B;
                        A << a << b;
                        B << b << a;

                        ASSERT(A != B || a == b);
                    }
                },
                {
                    "#6 Absolute refractory period (Each spike (1) must be followed by a no-spike (0).)",
                    Repeat100x,
                    []() {
                        const Input x = random<Input>();
                        const InputSequence no_consecutive_spikes = { x, ~x };
                        const InputSequence consecutive_spikes = { x, x };

                        Model A, B;

                        ASSERT(A.learn(no_consecutive_spikes, SimulatedInfinity));
                        ASSERT(not B.learn(consecutive_spikes, SimulatedInfinity) || x == Input{});
                    }
                },
                {
                    "#7 Temporal adaptability (The model can adapt to and predict temporal patterns of varying lengths.)",
                    Repeat100x,
                    []() {
                        const InputSequence trivial_problem(InputSequence::trivial, SequenceLength);
                        const InputSequence longer_trivial_problem(InputSequence::trivial, SequenceLength + 1);
                        Model A;

                        ASSERT(A.learn(trivial_problem, SimulatedInfinity));
                        ASSERT(A.learn(longer_trivial_problem, SimulatedInfinity));
                    }
                },
                {
                    "#8 Stagnation (You can't teach an old dog new tricks.)",
                    Repeat100x,
                    []() {
                        auto indefinitely_adaptable = [&](Model& dog) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                InputSequence learnable_trick = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);

                                if (not dog.learn(learnable_trick, SimulatedInfinity))
                                    return false;
                            }
                            return true;
                        };

                        Model A;

                        ASSERT(not indefinitely_adaptable(A));
                    }
                },
                {
                    "#9 Content sensitivity (Adaptation time depends on the content of the input sequence.)",
                    Repeat100x,
                    []() {
                    // Null Hypothesis: Adaptation time is independent of the input sequence content
                    auto adaptation_time_depends_on_the_content_of_the_input_sequence = [=]() -> bool {
                        Model B;
                        const InputSequence base_sequence = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);
                        const time_t base_time = B.time_to_repeat(base_sequence, SimulatedInfinity);
                        for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                            const InputSequence new_pattern(InputSequence::circular_random, SequenceLength);

                            if (new_pattern != base_sequence) {
                                Model A;
                                time_t time = A.time_to_repeat(new_pattern, SimulatedInfinity);
                                if (base_time != time)
                                    return true;                            // rejects the null hypothesis
                            }
                        }
                        return false;
                    };

                    ASSERT(adaptation_time_depends_on_the_content_of_the_input_sequence());
                }
            },
            {
                "#10 Context sensitivity (Adaptation time depends on the state of the model.)",
                Repeat100x,
                []() {
                    // Null Hypothesis: Adaptation time is independent of the state of the model
                    auto adaptation_time_depends_on_state_of_the_model = [&]() -> bool {
                        const InputSequence target_sequence = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);
                        Model B;
                        const time_t base_time = B.time_to_repeat(target_sequence, SimulatedInfinity);
                        for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                            Model A(Model::random, SequenceLength);

                            if (A != Model{}) {
                                time_t time = A.time_to_repeat(target_sequence, SimulatedInfinity);
                                if (base_time != time)                    // rejects the null hypothesis
                                    return true;
                            }
                        }
                        return false;
                    };

                    ASSERT(adaptation_time_depends_on_state_of_the_model());
                }
            },
            {
                "#11 Unobservability (Distinct models may exhibit the same observable behaviour in some timeframe.)",
                Repeat100x,
                []() {
                    // Null Hypothesis: "Different models cannot produce identical behavior."
                    auto different_model_instances_can_produce_identical_behaviour = [&]() -> bool {
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
                "#12 Denoising (The model performs above chance on perturbed inputs.)",
                Repeat100x,
                []() {
                    size_t score = 0;
                    const int N = 20, exposure_time = 5 * SequenceLength;
                    for (int i = 0; i < N; ++i) {
                        const InputSequence seq(InputSequence::circular_random, SequenceLength);
                        const Input disruption = random<Input>(seq[1], seq.back());

                        Model A;
                        for (int i = 0; i < exposure_time; ++i)
                            A << seq;                                       // prior experience    

                        A << disruption;                                    // begin a novel situation
                        A << (seq | std::views::drop(1));

                        const Input& truth = seq.front();
                        score += utils::count_matching_bits(A.prediction(), truth);
                    }
                    const size_t random_guess = N * BitsPerInput / 2;

                    ASSERT(score > random_guess);
                }
            },
            {
                "#13 Generalization (The model performs above chance on previously unseen inputs.)",
                Repeat100x,
                []() {
                    size_t score = 0;
                    const int N = 20, k = 10;
                    for (int i = 0; i < N; ++i) {
                        Model A(Model::random, k * SequenceLength);         // R sets the unknown rule behind the data
                        const auto train = A.generate(k * SequenceLength);  // split: first k parts for training
                        const auto truth = A.generate(1 * SequenceLength);  //        1 subsequent part for testing  

                        Model B;
                        B << train;

                        score += utils::count_matching_bits(B.generate(SequenceLength), truth);
                    }
                    const size_t random_guess = N * SequenceLength * BitsPerInput / 2;

                    ASSERT(score > random_guess);
                }
            },
            {
                "#14 Latency (The model shall operate within a bounded latency.)",
                RepeatOnce,
                []() {
                    std::clog << yellow("Manual validation required:\n");

                    std::clog << "Can the Model, in principle, produce a prediction within a bounded latency? [y/n]\n";
                    int answer = std::getchar();

                    ASSERT(answer == 'y' or answer == 'Y');
                }
            }
            };
        };
    }
}
