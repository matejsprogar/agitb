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
#include "utils.h"

#define ASSERT(expression) (void)((!!(expression)) || (throw AGI::Error(__FILE__, __LINE__, #expression), 0))



namespace sprogar {
    namespace AGI {
        using std::vector;

        template <typename Cortex, typename Pattern, size_t SimulatedInfinity = 500>
            requires InputPredictor<Cortex, Pattern> and BitProvider<Pattern>
        class Testbed
        {
            using util = TestbedUtils<Cortex, Pattern, SimulatedInfinity>;

        public:
            static void run()
            {
                const time_t temporal_sequence_length = achievable_sequence_length();

                std::clog << "Artificial Intelligence Testbed:\n"
                     << "Conducting tests on temporal sequences of " << temporal_sequence_length << " patterns\n\n";

                try {
                    for (const auto& test : testbed)
                        test(temporal_sequence_length);

                    std::clog << green("PASS") << std::endl << std::endl;
                }
                catch (const Error& err) {
                    std::clog << red("Assertion failed") << err.what() << std::endl;
                    exit(-1);
                }
            }
            static time_t achievable_sequence_length()
            {
                for (time_t length = 2; length < SimulatedInfinity; ++length) {
                    Cortex C;
                    const vector<Pattern> input = util::circular_random_sequence(length);
                    if (!util::adapt(C, input))
                        return length - 1;
                }
                return SimulatedInfinity;
            }

        private:
            static inline const vector<void (*)(time_t)> testbed =
            {
                [](time_t) {
                    std::clog << "#1 Genesis (The system starts from a truly blank state, free of bias.)\n";

                    Cortex C;

                    ASSERT(C == Cortex{});  // Requires deep comparison in operator==
                },
                [](time_t) {
                    std::clog << "#2 Knowledge (Bias emerges from the inputs.)\n";

                    Cortex C;
                    C << util::random_pattern();

                    ASSERT(C != Cortex{});
                },
                [](time_t) {
                    std::clog << "#3 Determinism (Equal state implies equal life.)\n";
                    const vector<Pattern> life = util::random_sequence(SimulatedInfinity);

                    Cortex C, D;
                    C << life;
                    D << life;

                    ASSERT(C == D);
                },
                [](time_t) {
                    std::clog << "#4 Time (The input order is inherently temporal and crucial to the process.)\n";
                    const Pattern pattern = util::random_pattern();
                    const Pattern patteRn = util::mutate(pattern);

                    Cortex C, D;
                    C << pattern << patteRn;
                    D << patteRn << pattern;

                    ASSERT(C != D);
                },
                [](time_t) {
                    std::clog << "#5 Sensitivity (The cortex exhibits chaos-like sensitivity to initial conditions.)\n";
                    const Pattern initial_condition = util::random_pattern();
                    const Pattern mutated_condition = util::mutate(initial_condition);
                    const vector<Pattern> life = util::random_sequence(SimulatedInfinity);

                    Cortex C, D;
                    C << initial_condition << life;
                    D << mutated_condition << life;

                    ASSERT(C != D);
                },
                [](time_t) {
                    std::clog << "#6 RefractoryPeriod (Each spike (1) must be followed by a no-spike (0).)\n";
                    const Pattern no_spikes;
                    const Pattern single_spike = util::mutate(no_spikes);
                    const vector<Pattern> no_consecutive_spikes = { single_spike, no_spikes };
                    const vector<Pattern> consecutive_spikes = { single_spike, single_spike };

                    Cortex C, D;

                    ASSERT(util::adapt(C, no_consecutive_spikes));
                    ASSERT(not util::adapt(D, consecutive_spikes));
                },
                [](time_t temporal_sequence_length) {
                    std::clog << "#7 Scalability (The system can adapt to predict also longer sequences.)\n";
                    
                    util::learnable_random_sequence(temporal_sequence_length + 1);  // throw
                    
                    ASSERT(true);   // nothrow
                },
                [](time_t temporal_sequence_length) {
                    std::clog << "#8 Stagnation (You can't teach an old dog new tricks.)\n";
                    auto indefinitely_adaptable = [&](Cortex& dog) -> bool {
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            vector<Pattern> new_trick = util::learnable_random_sequence(temporal_sequence_length);
                            if (not util::adapt(dog, new_trick))
                                return false;
                        }
                        return true;
                    };

                    Cortex C;

                    ASSERT(not indefinitely_adaptable(C));
                },
                [](time_t temporal_sequence_length) {
                    std::clog << "#9 Input (Learning time depends on the input sequence content.)\n";
                    // Null Hypothesis: Learning time is independent of the input sequence
                    auto learning_time_can_differ_across_sequences = [=]() -> bool {
                        Cortex D;
                        const time_t default_time = util::time_to_repeat(D, util::circular_random_sequence(temporal_sequence_length));
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            vector<Pattern> random_sequence = util::circular_random_sequence(temporal_sequence_length);
                            Cortex C;
                            time_t random_time = util::time_to_repeat(C, random_sequence);
                            if (default_time != random_time)
                                return true;
                        }
                        return false;
                    };
                    
                    ASSERT(learning_time_can_differ_across_sequences());   // rejects the null hypothesis
                },
                [](time_t temporal_sequence_length) {
                    std::clog << "#10 Experience (Learning time depends on the state of the cortex.)\n";
                    // Null Hypothesis: Learning time is independent of the state of the cortex
                    auto learning_time_can_differ_across_cortices = [&]() -> bool {
                        const vector<Pattern> target_sequence = util::learnable_random_sequence(temporal_sequence_length);

                        Cortex D;
                        const time_t default_time = util::time_to_repeat(D, target_sequence);
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            Cortex R = util::random_cortex(temporal_sequence_length);
                            time_t random_time = util::time_to_repeat(R, target_sequence);
                            if (default_time != random_time)
                                return true;
                        }
                        return false;
                    };

                    ASSERT(learning_time_can_differ_across_cortices());   // rejects the null hypothesis
                },
                [](time_t temporal_sequence_length) {
                    std::clog << "#11 Unobservability (Different internal states can produce identical behaviour.)\n";
                    // Null Hypothesis: "Different cortices cannot produce identical behavior."
                    auto behaviour_can_be_identical_across_cortices = [&]() -> bool {
                        const time_t nontrivial_problem_size = 2;

                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            const vector<Pattern> target_behaviour = util::learnable_random_sequence(nontrivial_problem_size);
                            
                            Cortex C, R = util::random_cortex(temporal_sequence_length);
                            util::adapt(C, target_behaviour);
                            util::adapt(R, target_behaviour);

                            ASSERT(C != R);
                            if (util::behaviour(C) == util::behaviour(R))
                                return true;    // C != R && util::behaviour(C) == util::behaviour(R)
                        }
                        return false;
                    };

                    ASSERT(behaviour_can_be_identical_across_cortices());   // rejects the null hypothesis
                },
                [](time_t temporal_sequence_length) {
                    std::clog << "#12 Advantage (Adapted models predict more accurately.)\n";
                    
                    size_t average_adapted_score = 0, average_unadapted_score = 0;
                    for (time_t time = 0; time < SimulatedInfinity; ++time) {
                        const vector<Pattern> facts = util::learnable_random_sequence(temporal_sequence_length);
                        const Pattern disruption = util::random_pattern();
                        const Pattern expectation = facts[0];

                        Cortex A;
                        util::adapt(A, facts);
                        A << disruption << facts;
                        average_adapted_score += util::count_matches(A.predict(), expectation);

                        Cortex U;
                        U << disruption << facts;
                        average_unadapted_score += util::count_matches(U.predict(), expectation);
                    }

                    ASSERT(average_adapted_score > average_unadapted_score);
                }            
            };
        };
    }
}