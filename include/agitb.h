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
        const time_t SequenceLength = 7;        // \eta
        const size_t BitsPerInput = 10;         // \omega
        const size_t Repeat100x = 100;
        const size_t RepeatForever = SimulatedInfinity;
        const size_t RepeatOnce = 1;
        enum mode { exhaustive = 0, fast = 1};

        template <typename SystemUnderEvaluation>
        class TestBed
        {
            using Input = std::bitset<BitsPerInput>;
            using InputSequence = utils::InputSequence<Input>;
            using Model = utils::Model<SystemUnderEvaluation, Input>;

        public:
            static bool run(mode _mode=exhaustive)
            {
                std::clog << "Artificial General Intelligence Testbed\n\n";

                const std::string go_back(10, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;
                
                    const size_t N = repeats(_mode, repetitions);
                    for (size_t i = 1; i <= N; ++i) {
                        std::clog << i << '/' << N << go_back;
                
                        test();
                    }
                }

                std::clog << green("\nPASS\n");
                return true;
            }

        private:
            static size_t repeats(mode _mode, size_t repetitions) { return _mode == exhaustive ? repetitions : std::min(repetitions, Repeat100x); }
            static inline const std::vector<std::tuple<std::string, size_t, void(*)()>> testbed =
            {
                {
                    "#1 Uninformed start (All instances of a given model type begin transitioning from an identical initial configuration.)",
                    RepeatOnce,
                    []() {
                        Model A;

                        ASSERT(A == Model{});				        // A_0 == B_0
                    }
                },
                {
                    "#2 Model transition (Every exposure to input produces a model transition.)",
                    Repeat100x,
                    []() {
                        Model A;
                        A << random<Input>();

                        ASSERT(A != Model{});                       // A_1 != A_0
                    }
                },
                {
                    "#3 Determinism (Model evolution is deterministic with respect to input.)",
                    RepeatForever,
                    []() {
                        auto deterministic = []() {
                            const InputSequence random_experience(InputSequence::random, SimulatedInfinity);

                            Model A, B;
                            A << random_experience;
                            B << random_experience;

                            return A == B;
                        };

                        ASSERT(deterministic());
                    }
                },
                {
                    "#4 Update injectivity (Distinct models remain distinct under any input.)",
                    RepeatForever,
                    []() {
                        auto sensitive = []() {
                            const Input p = random<Input>();
                            const InputSequence random_experience(InputSequence::random, SimulatedInfinity);

                            Model A, B;                             // A_0, B_0
                            A << p << random_experience;            // A_1, A_5001
                            B << ~p << random_experience;           // B_1, B_5001

                            return A != B;                          // A_5001 != B_5001
                        };
                        
                        ASSERT(sensitive());
                    }
                },
                {
                    "#5 Time (Model evolution depends on input order.)",
                    RepeatForever,
                    []() {
                        const Input x = random<Input>();

                        Model A, B;
                        A << x << ~x;
                        B << ~x << x;

                        ASSERT(A != B);
                    }
                },
                {
                    "#6 Absolute refractory period (A model can learn a cyclic sequence only if the sequence satisfies the absolute refractory-period constraint.)",
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
                    "#7 Temporal adaptability (The model must be able to learn sequences with varying cycle lengths.)",
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
                    "#8 Bounded learnability (Teaching tricks to an old dog.)",
                    RepeatForever,
                    []() {
                        auto limited_learnability = [&](Model& A) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                InputSequence learnable_trick = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);

                                if (not A.learn(learnable_trick, SimulatedInfinity))
                                    return true;
                            }
                            return false;
                        };
                        auto length_2_sequences_universally_learnable = [&](Model& A) -> bool {
                            const size_t nontrivial_length = 2;
                            InputSequence any_short_trick(InputSequence::circular_random, nontrivial_length);

                            return A.learn(any_short_trick, SimulatedInfinity);
                        };

                        Model A;

                        ASSERT(limited_learnability(A));    // The model has limited learnability.
                        ASSERT(length_2_sequences_universally_learnable(A));  // All admissible length-2 sequences are universally learnable.
                    }
                },                
                {
                    "#9 Content sensitivity (Adaptation time is input dependent.)",
                    RepeatForever,
                    []() {
                    // Null Hypothesis: Adaptation time is independent of the input sequence content
                    auto adaptation_time_is_input_dependent = [=]() -> bool {
                        Model B;
                        const InputSequence base_sequence = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
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

                    ASSERT(adaptation_time_is_input_dependent());
                }
            },
                {
                    "#10 Context sensitivity (Adaptation time is model dependent.)",
                    RepeatForever,
                    []() {
                        // Null Hypothesis: Adaptation time is independent of the model
                        auto adaptation_time_is_model_dependent = [&]() -> bool {
                            const InputSequence target_sequence = Model::learnable_random_sequence(SequenceLength, SimulatedInfinity);
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

                        ASSERT(adaptation_time_is_model_dependent());
                    }
                },
                {
                    "#11 Unobservability (Distinct models may be observationally indistinguishable.)",
                    RepeatForever,
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
                    "#12 Denoising (The model outperforms the best trivial baseline predictor.)",
                    RepeatForever,
                    []() {
                        size_t model_score = 0, baseline_0_score = 0, baseline_1_score = 0;
                        const int N = 20, exposure_time = 5 * SequenceLength;   // plenty of time
                        for (int i = 0; i < N; ++i) {
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
                        const int N = 20, k = 10;
                        for (int i = 0; i < N; ++i) {
                            Model rule_generator(Model::random, 1000*SequenceLength);        // implements an unknown random rule
                            const auto train = rule_generator.generate(k * SequenceLength);  // split: first k parts for training
                            const auto truth = rule_generator.generate(1 * SequenceLength);  //        1 subsequent part for testing  

                            Model A;
                            A << train;

                            score += utils::count_matching_bits(A.generate(truth.size()), truth);
                        }
                        const size_t random_guess = N * SequenceLength * BitsPerInput / 2;

                        ASSERT(score > random_guess);
                    }
                },
                {
                    "#14 Liveness (The model completes each input-driven transition within bounded time.)",
                    RepeatForever,
                    []() {
                        auto state_update_time = [](Model& M, const Model::InputSequence& sequence) -> size_t {
                            const auto start = std::chrono::high_resolution_clock::now();

                            M << sequence;

                            const auto end = std::chrono::high_resolution_clock::now();
                            return (size_t)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                        };
                        auto experiments = [&](size_t N) {
                            std::vector<std::pair<size_t, size_t>> results; results.reserve(N);
                            
                            while (results.size() < N) {
                                const InputSequence sequence(InputSequence::random, SimulatedInfinity);

                                Model A;
                                Model B(Model::random, SimulatedInfinity);

                                results.emplace_back(
                                    state_update_time(A, sequence),   // empty model's time
                                    state_update_time(B, sequence)    // complex model's time
                                );
                            }

                            return results;
                        };

                        const int N = 100;
                        const auto results = experiments(N);

                        ASSERT(not utils::consistently_greater_second_value(results));
                    }
                }
            };
        };
    }
}
