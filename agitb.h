/*
 * Copyright 2024 Matej Sprogar <matej.sprogar@gmail.com>
 * 
 * This file is part of AGITB - Artificial General Intelligence Testbed.
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
#include <cassert>
#include <random>
#include <vector>
#include <functional>

#include "concepts.h"
#include "helpers.h"


#define ASSERT(expression) (void)((!!(expression)) || (std::cerr << red("Assertion failed ") \
	<< __FILE__ << "\nLine " << __LINE__ << ": " << #expression << std::endl, exit(-1), 0))



namespace sprogar {
    namespace AGI {
        using namespace std;

        template <typename Cortex, typename Pattern, size_t SimulatedInfinity = 500>
            requires InputPredictor<Cortex, Pattern> and BitProvider<Pattern>
        class Testbed
        {
        public:
            static void run()
            {
                const time_t temporal_sequence_length = achievable_sequence_length();

                clog << "Artificial Intelligence Testbed:\n"
                     << "Conducting tests on temporal sequences of " << temporal_sequence_length << " patterns\n\n";

                for (const auto& test : testbed)
                    test(temporal_sequence_length);

                clog << green("PASS") << endl << endl;
            }

        private:
            using time_t = size_t;
            template<typename T> using TemporalSequence = std::vector<T>;

            static time_t achievable_sequence_length()
            {
                for (time_t length = 2; length < SimulatedInfinity; ++length) {
                    Cortex C;
                    const TemporalSequence<Pattern> input = generate_circular_random_sequence(length);
                    if (!adapt(C, input))
                        return length - 1;
                }
                return SimulatedInfinity;
            }

            // Each bit in the pattern is set randomly unless explicitly required to remain off.
            template<std::same_as<Pattern>... Patterns>
            static Pattern generate_random_pattern(const Patterns&... off)
            {
                static thread_local std::mt19937 generator{ std::random_device{}() };
                static std::bernoulli_distribution bd(0.5);

                Pattern pattern;
                for (size_t i = 0; i < Pattern::size(); ++i)
                    if (!(false | ... | off[i]))
                        pattern[i] = bd(generator);

                return pattern;
            }
            static TemporalSequence<Pattern> generate_random_sequence(time_t temporal_sequence_length)
            {
                assert(temporal_sequence_length > 0);
                TemporalSequence<Pattern> sequence;
                sequence.reserve(temporal_sequence_length);

                sequence.push_back(generate_random_pattern());
                while (sequence.size() < temporal_sequence_length)
                    sequence.push_back(generate_random_pattern(sequence.back()));

                return sequence;
            }
            static TemporalSequence<Pattern> generate_circular_random_sequence(time_t circle_length)
            {
                assert(circle_length > 1);
                TemporalSequence<Pattern> sequence = generate_random_sequence(circle_length);

                sequence.pop_back();
                sequence.push_back(generate_random_pattern(sequence.back(), sequence.front()));

                return sequence;
            }
            static TemporalSequence<Pattern> generate_random_learnable_sequence(time_t temporal_sequence_length)
            {
                while (true) {
                    Cortex C;
                    TemporalSequence<Pattern> sequence = generate_circular_random_sequence(temporal_sequence_length);
                    if (adapt(C, sequence))
                        return sequence;
                }
            }
            static Cortex generate_random_cortex(time_t random_strength)
            {
                Cortex C;
                C << generate_random_sequence(random_strength);
                return C;
            }

            static TemporalSequence<Pattern> behaviour(Cortex& C, time_t output_size = SimulatedInfinity)
            {
                TemporalSequence<Pattern> predictions;
                predictions.reserve(output_size);

                while (predictions.size() < output_size) {
                    predictions.push_back(C.predict());
                    C << predictions.back();
                }
                return predictions;
            }
            static TemporalSequence<Pattern> predict(Cortex& C, const TemporalSequence<Pattern>& inputs)
            {
                TemporalSequence<Pattern> predictions;
                predictions.reserve(inputs.size());

                for (const Pattern& in : inputs) {
                    predictions.push_back(C.predict());
                    C << in;
                }
                return predictions;
            }
            static time_t time_to_repeat(Cortex& C, const TemporalSequence<Pattern>& inputs)
            {
                for (time_t time = 0; time < SimulatedInfinity; time += inputs.size()) {
                    if (predict(C, inputs) == inputs)
                        return time;
                }
                return SimulatedInfinity;
            }
            static bool adapt(Cortex& C, const TemporalSequence<Pattern>& inputs)
            {
                return time_to_repeat(C, inputs) < SimulatedInfinity;
            }


            static inline const std::vector<void (*)(time_t)> testbed =
            {
                [](time_t) {
                    clog << "#1 Genesis (The system starts from a truly blank state, free of bias.)\n";

                    Cortex C;

                    ASSERT(C == Cortex{});  // Requires deep comparison in operator==
                },
                [](time_t) {
                    clog << "#2 Knowledge (Bias emerges from the inputs.)\n";

                    Cortex C;
                    C << generate_random_pattern();

                    ASSERT(C != Cortex{});
                },
                [](time_t) {
                    clog << "#3 Determinism (Equal state implies equal life.)\n";
                    const TemporalSequence<Pattern> life = generate_random_sequence(SimulatedInfinity);

                    Cortex C, D;
                    C << life;
                    D << life;

                    ASSERT(C == D);
                },
                [](time_t) {
                    clog << "#4 Time (The order of inputs is crucial.)\n";
                    const Pattern pattern = generate_random_pattern(), patteRn = helpers::mutate(pattern);

                    Cortex C, D;
                    C << pattern << patteRn;
                    D << patteRn << pattern;

                    ASSERT(C != D);
                },
                [](time_t) {
                    clog << "#5 Sensitivity (The cortex behaves as a chaotic system.)\n";
                    const Pattern initial_condition = generate_random_pattern(), mutated_condition = helpers::mutate(initial_condition);
                    const TemporalSequence<Pattern> life = generate_random_sequence(SimulatedInfinity);

                    Cortex C, D;
                    C << initial_condition << life;
                    D << mutated_condition << life;

                    ASSERT(C != D);
                },
                [](time_t) {
                    clog << "#6 RefractoryPeriod (Each spike (1) must be followed by a no-spike (0).)\n";
                    const Pattern no_spikes, single_spike = helpers::mutate(no_spikes);
                    const TemporalSequence<Pattern> no_consecutive_spikes = { single_spike, no_spikes };
                    const TemporalSequence<Pattern> consecutive_spikes = { single_spike, single_spike };

                    Cortex C, D;

                    ASSERT(adapt(C, no_consecutive_spikes));
                    ASSERT(not adapt(D, consecutive_spikes));
                },
                [](time_t temporal_sequence_length) {
                    clog << "#7 Scalability (The system can adapt to predict longer sequences.)\n";
                    auto can_adapt_to_longer_sequences = [&]() -> bool {
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            Cortex C;
                            const TemporalSequence<Pattern> longer_sequence = generate_circular_random_sequence(temporal_sequence_length + 1);
                            if (adapt(C, longer_sequence))
                                return true;
                        }
                        return false;
                    };

                    ASSERT(can_adapt_to_longer_sequences());
                },
                [](time_t temporal_sequence_length) {
                    clog << "#8 Stagnation (You can't teach an old dog new tricks.)\n";
                    auto indefinitely_adaptable = [&](Cortex& dog) -> bool {
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            TemporalSequence<Pattern> new_trick = generate_random_learnable_sequence(temporal_sequence_length);
                            if (not adapt(dog, new_trick))
                                return false;
                        }
                        return true;
                    };

                    Cortex C;

                    ASSERT(not indefinitely_adaptable(C));
                },
                [](time_t temporal_sequence_length) {
                    clog << "#9 Input (Learning time depends on the input sequence content.)\n";
                    // Null Hypothesis: Learning time is independent of the input sequence
                    auto learning_time_can_differ_across_sequences = [=]() -> bool {
                        Cortex D;
                        const time_t default_time = time_to_repeat(D, generate_circular_random_sequence(temporal_sequence_length));
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            TemporalSequence<Pattern> random_sequence = generate_circular_random_sequence(temporal_sequence_length);
                            Cortex C;
                            time_t random_time = time_to_repeat(C, random_sequence);
                            if (default_time != random_time)
                                return true;
                        }
                        return false;
                    };
                    
                    ASSERT(learning_time_can_differ_across_sequences());   // rejects the null hypothesis
                },
                [](time_t temporal_sequence_length) {
                    clog << "#10 Experience (Learning time depends on the state of the cortex.)\n";
                    // Null Hypothesis: Learning time is independent of the state of the cortex
                    auto learning_time_can_differ_across_cortices = [&]() -> bool {
                        Cortex D;
                        const TemporalSequence<Pattern> target_sequence = generate_random_learnable_sequence(temporal_sequence_length);
                        const time_t default_time = time_to_repeat(D, target_sequence);
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            Cortex R = generate_random_cortex(temporal_sequence_length);
                            time_t random_time = time_to_repeat(R, target_sequence);
                            if (default_time != random_time)
                                return true;
                        }
                        return false;
                    };

                    ASSERT(learning_time_can_differ_across_cortices());   // rejects the null hypothesis
                },
                [](time_t temporal_sequence_length) {
                    clog << "#11 Unobservability (Different internal states can produce identical behaviour.)\n";
                    // Null Hypothesis: "Different cortices cannot produce identical behavior."
                    auto behaviour_can_be_identical_across_cortices = [&]() -> bool {
                        const time_t nontrivial_problem_size = 2;

                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            const TemporalSequence<Pattern> target_behaviour = generate_random_learnable_sequence(nontrivial_problem_size);
                            Cortex C, R = generate_random_cortex(temporal_sequence_length);
                            adapt(C, target_behaviour);
                            adapt(R, target_behaviour);

                            ASSERT(C != R);
                            if (behaviour(C) == behaviour(R))
                                return true;    // C != R && behaviour(C) == behaviour(R)
                        }
                        return false;
                    };

                    ASSERT(behaviour_can_be_identical_across_cortices());   // rejects the null hypothesis
                },
                [](time_t temporal_sequence_length) {
                    clog << "#12 Advantage (Adapted models predict more accurately.)\n";
                    
                    size_t average_adapted_score = 0, average_unadapted_score = 0;
                    for (time_t time = 0; time < SimulatedInfinity; ++time) {
                        const TemporalSequence<Pattern> facts = generate_random_learnable_sequence(temporal_sequence_length);
                        const Pattern disruption = generate_random_pattern(), expectation = facts[0];

                        Cortex A;
                        adapt(A, facts);
                        A << disruption << facts;
                        average_adapted_score += helpers::count_matches(A.predict(), expectation);

                        Cortex U;
                        U << disruption << facts;
                        average_unadapted_score += helpers::count_matches(U.predict(), expectation);
                    }

                    ASSERT(average_adapted_score > average_unadapted_score);
                }            
            };
        };
    }
}