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
            using Model = utils::Model<SystemUnderEvaluation, Input, SimulatedInfinity>;

            enum test_repetitions { RepeatOnce = 1, Repeat10x = 10, Repeat100x = 100, RepeatForever = SimulatedInfinity };

        public:
            // Runs all tests from the testbed using the specified test mode.
            static bool run(size_t repetitions_override = 0)
            {
                std::clog << "Artificial General Intelligence Testbed\n";
                
                std::clog << "\n\nRunning 12 tests...\n";
                const std::string go_back(20, '\b');
                for (const auto& [info, repetitions, test] : testbed) {
                    std::clog << info << std::endl;

                    const size_t test_repetitions = repetitions_override == 0 ? repetitions : std::min((size_t)repetitions, (size_t)repetitions_override);
                    for (size_t r = 1; r <= test_repetitions; ++r) {
                        std::clog << r << '/' << test_repetitions << "   " << go_back;

                        utils::rng_seed = utils::rng();
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
                    RepeatForever,
                    []() {
                        const Model R(Model::random);

                        for (const Input& x : all_distinct_inputs) {
                            Model A = R, B = R;
                            A << x;
                            B << x;

                            ASSERT(A == B);
                        }
                    }
                },
                {
                    // Each input leaves a permanent internal trace.
                    "#3 Trace", 
                    RepeatOnce,
                    []() {
                        Model A;

                        const Input zeros = Input{};                            // edge case
                        std::vector<Model> trajectory;
                        trajectory.reserve(SimulatedInfinity);
                        while (trajectory.size() < SimulatedInfinity) {
                            trajectory.push_back(A);
                            A << zeros;

                            ASSERT(std::find(trajectory.begin(), trajectory.end(), A) == trajectory.end());
                        }
                    }
                },
                {
                    // Model evolution depends on input order.
                    "#4 Time",
                    Repeat100x,
                    []() {
                        Model A(Model::random);

                        auto complementary_inputs = [](const Input& x) { return x.count() <= BitsPerInput / 2; };
                        for (const Input& x : all_distinct_inputs | std::views::filter(complementary_inputs)) {
                            Model _A = A, _B = A;
                            _A << x << ~x;
                            _B << ~x << x;

                            ASSERT(_A != _B);
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

                            ASSERT(A.learn(no_consecutive_spikes));
                            ASSERT(not B.learn(consecutive_spikes) || !spikes);
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
                                const InputSequence learnable_sequence = Model::learnable_random_sequence(SequenceLength);

                                if (not A.learn(learnable_sequence))
                                    return true;
                            }
                            return false;
                        };
                        auto universal_learnability_of_admissible_length_2_sequences = [](const Model& A) -> bool {
                            auto admissible = [](const Input& x1, const Input& x2) -> bool { return (x1 & x2).none(); };

                            for (const Input& x1 : all_distinct_inputs) {
                                for (const Input& x2 : all_distinct_inputs) {
                                    if (!admissible(x1, x2))
                                        continue;

                                    const InputSequence admissible_length_2_sequence = { x1, x2 };
                                    Model B = A;
                                    if (!B.learn(admissible_length_2_sequence))
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
                            const time_t time_base_seq = A.time_to_learn(base_seq);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                const InputSequence seq(InputSequence::circular_random, SequenceLength);    // admissible by construction

                                if (seq != base_seq) {
                                    Model B;
                                    const time_t time_seq = B.time_to_learn(seq);
                                    const bool seq_learnable = time_seq != SimulatedInfinity;
                                    if (seq_learnable and time_seq != time_base_seq)                         // rejects the null hypothesis
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
                            const InputSequence seq = Model::learnable_random_sequence(SequenceLength);
                            Model A;
                            const time_t A_time = A.time_to_learn(seq);
                            for (size_t attempts = 0; attempts < SimulatedInfinity; ++attempts) {
                                Model B(Model::random);                                                     // even if A == B by chance, a vast majority of 
                                                                                                            // other models will differ from A
                                time_t B_time = B.time_to_learn(seq);
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
                        auto corrupt = [](const Input& x, const Input& x_prev, const Input& x_next) -> std::optional<Input> {
                            const Input corruptible_bits = ~(x_prev | x_next);
                            return corruptible_bits.any() ? std::optional<Input>{ x ^ corruptible_bits } : std::nullopt;
                        };
                        const Input zeros = Input{}, ones = zeros;
                        size_t model_score = 0, baseline_0_score = 0, baseline_1_score = 0;
                        const int num_of_runs = 20;                                 // within each of 5,000 trials
                        const int n = 5;                                            // informing context length
                        for (int i = 0; i < num_of_runs; ++i) {
                            const InputSequence reality(InputSequence::circular_random, SequenceLength);
                            const Input true_elt = reality[0];
                            if (const auto corrupted_elt = corrupt(true_elt, reality.back(), reality[1])) {
                                Model A;
                                for (int j = 0; j < n; ++j)
                                    A << reality;                                   // inform the model about the reality

                                // feed the old reality after the noisy sample or else a continuously learning model may begin generalising
                                A << *corrupted_elt << (reality | std::views::drop(1));

                                model_score += utils::match_score(A.get_prediction(), true_elt);
                                baseline_0_score += utils::match_score(zeros, true_elt);
                                baseline_1_score += utils::match_score(ones, true_elt);
                            }
                            else
                                i -= 1;
                        }
                        const size_t baseline = std::max(baseline_0_score, baseline_1_score);

                        ASSERT(model_score > baseline);
                    }
                },
                {
                    // A continuous learner can predict the unseen continuation of a complex stream.
                    "#11 Generalisation",
                    RepeatForever,
                    []() {
                        auto generate = [](size_t length) {
                            Model G;
                            Input last = random<Input>();
                            for (size_t attempt = 1; attempt <= SimulatedInfinity; ++attempt) {
                                G << last;
                                Model X = G;
                                InputSequence seq = X.generate(length);
                                if (not utils::is_periodic(seq))
                                    return std::make_pair(seq, X.get_prediction());
                                last = random<Input>(last);
                            }
                            bool can_generate_nonperiodic_sequences = false;
                            ASSERT(can_generate_nonperiodic_sequences);
                        };
                        const int experience_len = 1 * SequenceLength;
                        const int num_of_runs = 20;                                 // 1/20 is enough
                        for (int i = 0; i < num_of_runs; ++i) {
                            auto [experience, continuation] = generate(experience_len);

                            Model A;
                            A << experience;

                            const auto prediction = A.get_prediction();

                            if (continuation == prediction)                         // 1/1024 lucky chance
                                return;
                        }
                        bool can_generalise_once = false;
                        ASSERT(can_generalise_once);
                    }
                },
                {
                    // Each model update completes within a fixed wall-clock time bound, independent of the input history.
                    "#12 Real-time liveness",
                    RepeatForever,
                    []() {
                        static const time_t min_chunk_duration_us = 200;
                        static const size_t chunk_count = 100;
                        static const double jitter_tolerance = 4.0;

                        auto autotune_chunk_size = [=]() -> size_t {
                            const size_t tuning_samples = 11;
                            InputSequence chunk(InputSequence::random, 2ull);
                            while (true) {
                                std::vector<time_t> time_probes(tuning_samples);
                                for (time_t& time : time_probes) {
                                    Model M;
                                    time = utils::time_it([&]() { M << chunk; });
                                }
                                const auto [median, _] = utils::percentiles(time_probes);
                                if (median >= 2 * min_chunk_duration_us)
                                    break;
                                chunk = InputSequence(InputSequence::random, 2 * chunk.size());
                            }
                            return chunk.size();
                        };
                        auto assert_live_on = [&](auto make_chunk) {
                            std::vector<time_t> times;
                            times.reserve(chunk_count);

                            Model M;
                            while (times.size() < chunk_count) {
                                const time_t dt = utils::time_it([&]() { M << make_chunk(); });
                                times.push_back(dt);

                                static const time_t absolute_ceiling = 1'000 * min_chunk_duration_us;   // ~0.2 s/chunk
                                ASSERT(dt <= absolute_ceiling);
                            }

                            const double growth_tolerance = 1.25;              // allow 25% benign drift due to noise and other factors
                            auto early = times
                                | std::views::take(chunk_count / 2)
                                | std::views::transform([&](time_t tm) { return (time_t)(tm * growth_tolerance); });
                            auto late = times
                                | std::views::drop(chunk_count / 2);
                            const bool growing = utils::consistently_greater_second_value(early, late);
                            ASSERT(not growing);

                            const auto [median, p95] = utils::percentiles(times);
                            ASSERT(median >= min_chunk_duration_us);            // meaningful measurements
                            ASSERT(p95 <= median * jitter_tolerance);           // bounded worst case
                        };

                        auto periodic_chunk = [](const InputSequence& motif, size_t chunk_size) {
                            InputSequence chunk; chunk.reserve(chunk_size);
                            for (size_t k = 0; k < chunk_size; ++k)
                                chunk.push_back(motif[k % motif.size()]);

                            chunk.back() = chunk.back() & ~chunk.front();       // ARP
                            return chunk;
                        };

                        static const size_t chunk_size = autotune_chunk_size();
                        assert_live_on([&]() { return InputSequence(InputSequence::random, chunk_size); });
                        assert_live_on([&]() { return InputSequence(InputSequence::trivial, chunk_size); });

                        for (size_t i = 0; i < 10; ++i) {
                            const size_t pattern_period = utils::random(2, 4 * SequenceLength);
                            const InputSequence motif(InputSequence::circular_random, pattern_period);
                            assert_live_on([&]() { return periodic_chunk(motif, chunk_size); });
                        }
                    } 
                }
            };
        };
    }
}
