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
        const size_t Repeat100x = 100;
        const size_t RepeatOnce = 1;

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

                    for (size_t i = 1; i <= repetitions; ++i) {
                        std::clog << i << '/' << repetitions << go_back;

                        test();
                    }
                }

                std::clog << green("\nPASS\n");
            }

        private:
            static inline const std::vector<std::tuple<std::string, size_t, void(*)()>> testbed =
            {
                {
                    "#1 Bias-free start (All models begin in a completely blank, bias-free state.)",
                    RepeatOnce,
                    []() {
                        Model A;

                        ASSERT(A == Model{});				        // blank models are equal
                        ASSERT(A.get_prediction() == Input{});	    // first prediction: {0,0,0,0,0,0,0,0,0,0}
                    }
                },
                {
                    "#2 Bias (A change in state indicates bias.)",
                    Repeat100x,
                    []() {
                        const Input x = random<Input>();
                        Model A;
                        A << x;

                        ASSERT(A != Model{});
                        ASSERT(A.get_prediction() == x);
                    }
                },
                {
                    "#3 Injective determinism (Models are deterministic and sensitive)",
                    Repeat100x,
                    []() {
                        auto deterministic = []() {
                            const InputSequence random_experience(InputSequence::random, SimulatedInfinity);

                            Model A, B;
                            A << random_experience;
                            B << random_experience;

                            return A == B;
                        };
                        auto sensitive = []() {
                            const Input p = random<Input>();
                            const InputSequence random_experience(InputSequence::random, SimulatedInfinity);

                            Model A, B;
                            A << p << random_experience;
                            B << ~p << random_experience;

                            return A != B;
                        };
                        
                        ASSERT(deterministic() and sensitive());
                    }
                },
                {
                    "#4 Time (System evolution depends on input order.)",
                    Repeat100x,
                    []() {
                        const Input x1 = random<Input>();
                        const Input x2 = random<Input>();

                        Model A, B;
                        A << x1 << x2;
                        B << x2 << x1;

                        ASSERT(A != B || x1 == x2);
                        ASSERT(A.get_prediction() != B.get_prediction() || x1 == x2);
                    }
                },
                {
                    "#5 Absolute refractory period (Each spike (1) must be followed by a no-spike (0).)",
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
                    "#6 Temporal adaptability (The model must be able to learn sequences with varying cycle lengths.)",
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
                    "#7 Stagnation (After a time performance necessarily drops.)",
                    Repeat100x,
                    []() {
                        auto indefinitely_learnable = [&](Model& A) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                InputSequence learnable_trick = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);

                                if (not A.learn(learnable_trick, SimulatedInfinity))
                                    return false;
                            }
                            return true;
                        };
                        auto minimal_learning_ability = [&](Model& A) -> bool {
                            InputSequence short_trick(InputSequence::circular_random, 2);

                            return A.learn(short_trick, SimulatedInfinity);
                        };

                        Model A;

                        ASSERT(not indefinitely_learnable(A));
                        ASSERT(minimal_learning_ability(A));
                    }
                },                
                {
                    "#8 Content sensitivity (Adaptation time depends on the content of the input sequence.)",
                    Repeat100x,
                    []() {
                    // Null Hypothesis: Adaptation time is independent of the input sequence content
                    auto adaptation_time_depends_on_the_content_of_the_input_sequence = [=]() -> bool {
                        Model B;
                        const InputSequence base_sequence = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);
                        const time_t base_time = B.time_to_repeat(base_sequence, SimulatedInfinity);
                        for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                            const InputSequence sequence(InputSequence::circular_random, SequenceLength);

                            if (sequence != base_sequence) {
                                Model A;
                                time_t time = A.time_to_repeat(sequence, SimulatedInfinity);
                                if (base_time != time)                      // rejects the null hypothesis
                                    return true;
                            }
                        }
                        return false;
                    };

                    ASSERT(adaptation_time_depends_on_the_content_of_the_input_sequence());
                }
            },
                {
                    "#9 Context sensitivity (Adaptation time depends on the state of the model.)",
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
                    "#10 Unobservability (Distinct model instances may exhibit the same observable behaviour in some timeframe.)",
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
                    "#11 Denoising (The model performs above chance on perturbed inputs.)",
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
                            score += utils::count_matching_bits(A.get_prediction(), truth);
                        }
                        const size_t random_guess = N * BitsPerInput / 2;

                        ASSERT(score > random_guess);
                    }
                },
                {
                    "#12 Generalization (The model performs above chance on previously unseen inputs.)",
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
                }
            };
        };
    }
}
