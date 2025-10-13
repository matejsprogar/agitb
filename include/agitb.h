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

        // AGITB settings
        const time_t SequenceLength = 7;
        const size_t BitsPerInput = 10;
        const int Repeat100x = 100;
        const int RepeatOnce = 1;

        template <typename SystemUnderEvaluation, typename Input = std::bitset<BitsPerInput>>
        class TestBed
        {
            using InputSequence = utils::InputSequence<Input>;
            using Model = utils::Model<SystemUnderEvaluation, Input>;

        public:
            static void run()
            {
                std::clog << "Artificial General Intelligence Testbed\n\n";
                std::clog << "Testing with temporal patterns with " << SequenceLength << " inputs:\n";

                const std::string go_back(50, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;
                
                    for (int i = 1; i <= repetitions; ++i) {
                        std::clog << i << '/' << repetitions << '\t' << go_back;
                
                        test();
                    }
                }

                std::clog << green("\nPASS\n");
            }

        private:
            static inline const std::vector<std::tuple<std::string, int, void(*)()>> testbed =
            {
                {
                    "#1 Unbiased start (All cortices begin in a completely blank, bias-free state.)",
                    Repeat100x,
                    []() {
                        Model M;

                        ASSERT(M == Model{});				    // The initial state represents absence of bias.
                        ASSERT(M.prediction() == Input{});	    // No spikes indicate an unbiased initial prediction.
                    }
                },
                {
                    "#2 Bias (A change in state indicates bias.)",
                    Repeat100x,
                    []() {
                        Model M;
                        M << random<Input>();

                        ASSERT(M != Model{});
                    }
                },
                {
                    "#3 Determinism (Identical experiences produce an identical state.)",
                    Repeat100x,
                    []() {
                        const InputSequence experience = InputSequence(InputSequence::random, SimulatedInfinity);

                        Model M1, M2;
                        M1 << experience;
                        M2 << experience;

                        ASSERT(M1 == M2);
                    }
                },
                {
                    "#4 Sensitivity (The model exhibits chaos-like sensitivity to initial input.)",
                    Repeat100x,
                    []() {
                        const Input p = random<Input>();
                        const InputSequence experience = InputSequence(InputSequence::random, SimulatedInfinity);

                        Model M1, M2;
                        M1 << p << experience;
                        M2 << ~p << experience;

                        ASSERT(M1 != M2);
                    }
                },
                {
                    "#5 Time (The input order is inherently temporal and crucial to the process.)",
                    Repeat100x,
                    []() {
                        const Input in_1 = random<Input>();
                        const Input in_2 = random<Input>(in_1);     // in_1 & in_2 == Input{}

                        Model M1, M2;
                        M1 << in_1 << in_2;
                        M2 << in_2 << in_1;

                        ASSERT(M1 != M2 || in_1 == in_2);
                    }
                },
                {
                    "#6 Refractory period (Each spike (1) must be followed by a no-spike (0).)",
                    Repeat100x,
                    []() {
                        const Input p = random<Input>();
                        const InputSequence no_consecutive_spikes = { p, ~p };
                        const InputSequence consecutive_spikes = { p, p };
                
                        Model M1, M2;
                
                        ASSERT(M1.adapt(no_consecutive_spikes, SimulatedInfinity));
                        ASSERT(not M2.adapt(consecutive_spikes, SimulatedInfinity) || p == Input{});
                    }
                },
                {
                    "#7 Temporal adaptability (The model can adapt to and predict temporal patterns of varying lengths.)",
                    Repeat100x,
                    []() {
                        Model M;
                        ASSERT(M.adapt(InputSequence(InputSequence::trivial_problem, SequenceLength), SimulatedInfinity));
                        ASSERT(M.adapt(InputSequence(InputSequence::trivial_problem, 1 + SequenceLength), SimulatedInfinity));
                    }
                },
                {
                    "#8 Stagnation (You can't teach an old dog new tricks.)",
                    Repeat100x,
                    []() {
                        auto indefinitely_adaptable = [&](Model& dog) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                InputSequence learnable_trick = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);

                                if (not dog.adapt(learnable_trick, SimulatedInfinity))
                                    return false;
                            }
                            return true;
                        };
                
                        Model M;
                
                        ASSERT(not indefinitely_adaptable(M));
                    }
                },
                {
                    "#9 Content sensitivity (Adaptation time depends on the content of the input sequence.)",
                    Repeat100x,
                    []() {
                        // Null Hypothesis: Adaptation time is independent of the input sequence content
                        auto adaptation_time_depends_on_the_content_of_the_input_sequence = [=]() -> bool {
                            Model B{};
                            const InputSequence base_sequence = learnable_random_sequence<Model>(SequenceLength, SimulatedInfinity);
                            const time_t base_time = B.time_to_repeat(base_sequence, SimulatedInfinity);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                const InputSequence new_pattern = InputSequence(InputSequence::circular_random, SequenceLength);
                                
                                if (new_pattern != base_sequence) {
                                    Model M{};
                                    time_t new_time = M.time_to_repeat(new_pattern, SimulatedInfinity);
                                    if (base_time != new_time)
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
                                Model M = Model(Model::random, SequenceLength);
                                
                                if (M != Model{}) {
                                    time_t m_time = M.time_to_repeat(target_sequence, SimulatedInfinity);
                                    if (base_time != m_time)                    // rejects the null hypothesis
                                        return true;
                                }
                            }
                            return false;
                        };
                
                        ASSERT(adaptation_time_depends_on_state_of_the_model());
                    }
                },
                {
                    "#11 Unobservability (Distinct cortices may exhibit the same observable behaviour in some timeframe.)",
                    Repeat100x,
                    []() {
                        // Null Hypothesis: "Different cortices cannot produce identical behavior."
                        auto different_model_instances_can_produce_identical_behaviour = [&]() -> bool {
                            const InputSequence trivial_behaviour = { Input{}, Input{} };
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                Model M, R = Model(Model::random, SequenceLength);
                                M.adapt(trivial_behaviour, SimulatedInfinity);
                                R.adapt(trivial_behaviour, SimulatedInfinity);
                
                                bool counterexample = M != R && Model::identical_behaviour(M, R, 2 * SequenceLength);
                                if (counterexample)                             // rejects the null hypothesis
                                    return true;
                            }
                            return false;
                        };
                
                        ASSERT(different_model_instances_can_produce_identical_behaviour());
                    }
                },
                {
                    "#12 Denoising (A model can recall a sequence despite perturbations.)",
                    Repeat100x,
                    []() {
                        size_t score = 0;
                        const int N = 20, exposure_time = 5 * SequenceLength;
                        for (int i = 0; i < N; ++i) {
                            const InputSequence seq = InputSequence(InputSequence::circular_random, SequenceLength);
                            const Input disruption = random<Input>(seq[1], seq.back());

                            Model M;
                            for (int i = 0; i < exposure_time; ++i)
                                M << seq;                                       // prior experience    
                                
                            M << disruption;                                    // begin a novel situation
                            M << (seq | std::views::drop(1));

                            const Input& truth = seq.front();
                            score += utils::count_matching_bits(M.prediction(), truth);
                        }
                        const size_t random_guess = N * BitsPerInput / 2;
                
                        ASSERT(score > random_guess);
                    }
                },                
                {
                    "#13 Generalization (The model handles unseen situations.)",
                    Repeat100x,
                    []() {
                        size_t score = 0;
                        const int N = 20, k = 10;
                        for (int i = 0; i < N; ++i) {
                            Model R = Model(Model::random, k * SequenceLength); // R sets the rule behind the data
                            const auto train = R.generate(k * SequenceLength);  // split: first k parts for training
                            const auto truth = R.generate(1 * SequenceLength);  //     1 subsequent part for testing  
                            
                            Model M;
                            M << train;
                            
                            score += utils::count_matching_bits(M.generate(SequenceLength), truth);
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
