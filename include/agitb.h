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
#include <algorithm>
#include <chrono>

using namespace std::literals::chrono_literals;

#include "concepts.h"
#include "utils.h"

#define ASSERT(expression) (void)((!!(expression)) || \
                            (std::cerr << std::format("\n{} in {}:{}\n{}\n\n", red("\nAssertion failed"), __FILE__, __LINE__, #expression), \
                            exit(-1), 0))


namespace sprogar {
    namespace AGI {
        using std::string;

        template <typename TCortex, typename TInput, size_t SimulatedInfinity = 5000>
        class TestBed
        {
            using Input = utils::Input<TInput>;
            using Cortex = utils::Cortex<TCortex, Input, SimulatedInfinity>;
            using Sequence = Cortex::Sequence;

        public:
            static void run(time_t temporal_pattern_length, size_t test_repetitions = 100)
            {
                std::clog << "Artificial General Intelligence Testbed\n\n";
                std::clog << "Testing with temporal patterns with " << temporal_pattern_length << " inputs:\n";

                const string go_back(50, '\b');
                for (const auto& [info, test] : testbed) {
                    std::clog << info << std::endl;

                    for (size_t r = 1; r <= test_repetitions; ++r) {
                        std::clog << r << '/' << test_repetitions << go_back;

                        test(temporal_pattern_length);
                    }
                }

                std::clog << green("\nPASS\n");
            }

        private:
            static inline const std::vector<std::pair<string, void(*)(time_t)>> testbed =
            {
                {
                    "#1 Genesis (All cortices begin in a completely blank, bias-free state.)",
                    [](time_t) {
                        Cortex C;

                        ASSERT(C == Cortex{});				    // The initial state represents absence of bias.
                        ASSERT(C.prediction() == Input{});	    // No spikes indicate an unbiased initial prediction.
                    }
                },
                {
                    "#2 Bias (A change in state indicates bias.)",
                    [](time_t) {
                        Cortex C;
                        C << Input::random();

                        ASSERT(C != Cortex{});
                    }
                },
                {
                    "#3 Determinism (Identical experiences produce an identical state.)",
                    [](time_t) {
                        const Sequence experience = Sequence::random(SimulatedInfinity);

                        Cortex C, D;
                        C << experience;
                        D << experience;

                        ASSERT(C == D);
                    }
                },
                {
                    "#4 Sensitivity (The cortex exhibits chaos-like sensitivity to initial input.)",
                    [](time_t) {
                        const Input p = Input::random();
                        const Sequence life = Sequence::random(SimulatedInfinity);

                        Cortex C, D;
                        C << p << life;
                        D << ~p << life;

                        ASSERT(C != D);
                    }
                },
                {
                    "#5 Time (The input order is inherently temporal and crucial to the process.)",
                    [](time_t) {
                        const Input in_1 = Input::random();
                        const Input in_2 = Input::random(in_1);

                        Cortex C, D;
                        C << in_1 << in_2;
                        D << in_2 << in_1;

                        ASSERT(C != D || in_1 == in_2);
                    }
                },
                {
                    "#6 RefractoryPeriod (Each spike (1) must be followed by a no-spike (0).)",
                    [](time_t) {
                        const Input p = Input::random();
                        const Sequence no_consecutive_inputs = { p, Input::random(p) };
                        const Sequence consecutive_inputs = { p, p };

                        Cortex C, D;

                        ASSERT(C.adapt(no_consecutive_inputs));
                        ASSERT(not D.adapt(consecutive_inputs) || p == Input{});
                    }
                },
                {
                    "#7 Adaptability (The model can adapt to and predict patterns of unknown lengths.)",
                    [](time_t temporal_pattern_length) {
                        Cortex C;
                        ASSERT(C.adapt(Sequence::trivial_pattern(temporal_pattern_length)));
                        ASSERT(C.adapt(Sequence::trivial_pattern(temporal_pattern_length + 1)));
                    }
                },
                {
                    "#8 Stagnation (You can't teach an old dog new tricks.)",
                    [](time_t temporal_pattern_length) {
                        auto indefinitely_adaptable = [&](Cortex& dog) -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                Sequence new_trick = Cortex::adaptable_random_pattern(temporal_pattern_length);
                                if (not dog.adapt(new_trick))
                                    return false;
                            }
                            return true;
                        };

                        Cortex C;

                        ASSERT(not indefinitely_adaptable(C));
                    }
                },
                {
                    "#9 Content sensitivity (Adaptation time depends on the content of the input sequence.)",
                    [](time_t temporal_pattern_length) {
                        // Null Hypothesis: Adaptation time is independent of the input sequence content
                        auto adaptation_time_depends_on_the_content_of_the_input_sequence = [=]() -> bool {
                            Cortex B;
                            const Sequence base_pattern = Sequence::nontrivial_circular_random(temporal_pattern_length);
                            const time_t base_time = B.time_to_repeat(base_pattern);
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                const Sequence new_pattern = Sequence::nontrivial_circular_random(temporal_pattern_length);
                                if (new_pattern != base_pattern) {
                                    Cortex R;
                                    time_t new_time = R.time_to_repeat(new_pattern);
                                    if (base_time != new_time)
                                        return true;
                                }
                            }
                            return false;
                        };

                        ASSERT(adaptation_time_depends_on_the_content_of_the_input_sequence());   // rejects the null hypothesis
                    }
                },
                {
                    "#10 Context sensitivity (Adaptation time depends on the state of the cortex.)",
                    [](time_t temporal_pattern_length) {
                        // Null Hypothesis: Adaptation time is independent of the state of the cortex
                        auto adaptation_time_depends_on_state_of_the_cortex = [&]() -> bool {
                            const Sequence target_pattern = Cortex::adaptable_random_pattern(temporal_pattern_length);
                            Cortex B;
                            const time_t base_time = B.time_to_repeat(target_pattern);
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                Cortex N = Cortex::random();
                                if (N != Cortex{}) {
                                    time_t new_time = N.time_to_repeat(target_pattern);
                                    if (base_time != new_time)
                                        return true;
                                }
                            }
                            return false;
                        };

                        ASSERT(adaptation_time_depends_on_state_of_the_cortex());   // rejects the null hypothesis
                    }
                },
                {
                    "#11 Unobservability (Distinct cortices may exhibit the same observable behaviour.)",
                    [](time_t) {
                        // Null Hypothesis: "Different cortices cannot produce identical behavior."
                        auto different_cortex_instances_can_produce_identical_behaviour = [&]() -> bool {
                            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                                const Sequence trivial_behaviour = { Input{}, Input{} };

                                Cortex C{}, D = Cortex::random();
                                C.adapt(trivial_behaviour);
                                D.adapt(trivial_behaviour);

                                bool counterexample = C != D && Cortex::identical_behaviour(C, D, SimulatedInfinity);
                                if (counterexample)                         // rejects the null hypothesis
                                    return true;
                            }
                            return false;
                        };

                        ASSERT(different_cortex_instances_can_produce_identical_behaviour());
                    }
                },
                {
                    "#12 Generalisation (On average, adapted models exhibit the strongest generalisation.)",
                    [](time_t temporal_pattern_length) {
                        size_t adapted_score = 0, unadapted_score = 0;
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            const Sequence facts = Cortex::adaptable_random_pattern(temporal_pattern_length);
                            const Input disruption = Input::random();
                            const Input expectation = facts[0];

                            Cortex A;
                            A.adapt(facts);
                            A << disruption << facts;
                            adapted_score += Input::count_matches(A.prediction(), expectation);

                            Cortex U;
                            U << disruption << facts;
                            unadapted_score += Input::count_matches(U.prediction(), expectation);
                        }
                        const size_t random_guess = SimulatedInfinity * Input{}.size() / 2;

                        ASSERT(adapted_score > unadapted_score);
                        ASSERT(adapted_score > random_guess);
                    }
                },
                {
                    "#13 Bounded Prediction Latency (Cortex architecture allows bounded reaction times.)",
                    [](time_t) {
                        ASSERT(false);  // no automated test available - change manually
                    }
                }

            };
        };
    }
}
