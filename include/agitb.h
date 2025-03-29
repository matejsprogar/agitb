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


#include "concepts.h"
#include "utils.h"

#define ASSERT(expression) (void)((!!(expression)) || \
                            (std::cerr << std::format("\n{} in {}:\n{}\n\n", red("\nAssertion failed"), __FILE__, #expression), \
                            exit(-1), 0))


namespace sprogar {
namespace AGI {
    using std::string;

    template <typename CortexType, typename InputSample, size_t SimulatedInfinity = 1000, size_t Repetitions = 100>
    requires InputPredictor<CortexType, InputSample> && Indexable<InputSample>
        class TestBed
    {
        using Input = utils::Input<InputSample>;
        using Sequence = utils::Sequence<Input>;
        using Cortex = utils::Cortex<CortexType, Input, Sequence, SimulatedInfinity>;
        using Misc = utils::Misc<Cortex, Input, Sequence, SimulatedInfinity>;

    public:
        static void run(time_t temporal_pattern_length)
        {
            std::clog << "Artificial General Intelligence Test Bed\n\n";
            std::clog << "Testing with temporal patterns of " << temporal_pattern_length << " inputs:\n";

            for (const auto& [info, test] : testbed) {
                std::clog << info << std::endl;
                repeat(test, temporal_pattern_length);
            }
            std::clog << green("\nPASS\n");
        }

    private:
        static void repeat(void (*test)(time_t), const time_t temporal_pattern_length)
        {
            const string go_back(50, '\b');

            for (size_t r = 1; r <= Repetitions; ++r) {
                std::clog << r << '/' << Repetitions << go_back;

                test(temporal_pattern_length);
            }
        }

        static inline const std::vector<std::pair<string, void(*)(time_t)>> testbed =
        {
            {
                "#1 Genesis (All cortices begin in a completely blank, bias-free state.)",
                [](time_t) {
                    Cortex C;
                    
                    ASSERT(C == Cortex{});				// The initial state represents absence of bias.
                    ASSERT(C.predict() == Input{});	    // No spikes indicate an unbiased initial prediction.
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
                    const Sequence seq = Sequence::circular_random(2);

                    Cortex C, D;
                    C << seq[0] << seq[1];
                    D << seq[1] << seq[0];

                    ASSERT(C != D || seq[0] == seq[1]);
                }
            },
            {
                "#6 RefractoryPeriod (Each spike (1) must be followed by a no-spike (0).)",
                [](time_t) {
                    const Input p = Input::random();
                    const Sequence no_consecutive_spikes = { p, Input::random(p) };
                    const Sequence consecutive_spikes = { p, p };

                    Cortex C, D;

                    ASSERT(C.adapt(no_consecutive_spikes));
                    ASSERT(not D.adapt(consecutive_spikes) || p == Input{});
                }
            },
            {
                "#7 TemporalFlexibility (The model can adapt to and predict patterns of varying lengths.)",
                [](time_t temporal_pattern_length) {
                    ASSERT(Misc::adaptable_random_pattern(temporal_pattern_length)); 
                    ASSERT(Misc::adaptable_random_pattern(temporal_pattern_length + 1)); 
                }
            },
            {
                "#8 Stagnation (You can't teach an old dog new tricks.)",
                [](time_t temporal_pattern_length) {
                    auto indefinitely_adaptable = [&](Cortex& dog) -> bool {
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            Sequence new_trick = Misc::adaptable_random_pattern(temporal_pattern_length);
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
                "#9 Unsupervised (Adaptation time depends on the content of the input sequence.)",
                [](time_t temporal_pattern_length) {
                    // Null Hypothesis: Adaptation time is independent of the input sequence
                    auto adaptation_time_depends_on_the_content_of_the_input_sequence = [=]() -> bool {
                        Cortex B;
                        const time_t base_pattern_time = B.time_to_repeat(Sequence::circular_random(temporal_pattern_length));
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            Sequence another = Sequence::circular_random(temporal_pattern_length);
                            Cortex R;
                            time_t another_pattern_time = R.time_to_repeat(another);
                            if (base_pattern_time != another_pattern_time)
                                return true;
                        }
                        return false;
                    };

                    ASSERT(adaptation_time_depends_on_the_content_of_the_input_sequence());   // rejects the null hypothesis
                }
            },
            {
                "#10 Knowledge (Adaptation time depends on the state of the cortex.)",
                [](time_t temporal_pattern_length) {
                    // Null Hypothesis: Adaptation time is independent of the state of the cortex
                    auto adaptation_time_depends_on_state_of_the_cortex = [&]() -> bool {
                        const Sequence target_pattern = Misc::adaptable_random_pattern(temporal_pattern_length);
                        Cortex B;
                        const time_t base_time = B.time_to_repeat(target_pattern);
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            Cortex O = Cortex::random();
                            time_t other_cortex_time = O.time_to_repeat(target_pattern);
                            if (base_time != other_cortex_time)
                                return true;
                        }
                        return false;
                    };

                    ASSERT(adaptation_time_depends_on_state_of_the_cortex());   // rejects the null hypothesis
                }
            },
            {
                "#11 Unobservability (Different cortex instances can produce identical behaviour.)",
                [](time_t temporal_pattern_length) {
                    // Null Hypothesis: "Different cortices cannot produce identical behavior."
                    auto different_cortex_instances_can_produce_identical_behaviour = [&]() -> bool {
                        for (time_t time = 0; time < SimulatedInfinity; ++time) {
                            const Sequence trivial_behaviour = { Input{}, Input{} };

                            Cortex C{}, D = Cortex::random();
                            C.adapt(trivial_behaviour);
                            D.adapt(trivial_behaviour);

                            bool counterexample_found = C != D && C.behaviour() == D.behaviour();
                            if (counterexample_found)
                                return true;
                        }
                        return false;
                    };

                    ASSERT(different_cortex_instances_can_produce_identical_behaviour());   // rejects the null hypothesis
                }
            },
            {
                "#12 Generalisation (Adapted models predict more accurately.)",
                [](time_t temporal_pattern_length) {
                    const size_t random_guess = SimulatedInfinity * Input::size() / 2;
                    size_t adapted_score = 0, unadapted_score = 0;
                    for (time_t time = 0; time < SimulatedInfinity; ++time) {
                        const Sequence facts = Misc::adaptable_random_pattern(temporal_pattern_length);
                        const Input disruption = Input::random();
                        const Input expectation = facts[0];

                        Cortex A;
                        A.adapt(facts);
                        A << disruption << facts;
                        adapted_score += Input::count_matches(A.predict(), expectation);

                        Cortex U;
                        U << disruption << facts;
                        unadapted_score += Input::count_matches(U.predict(), expectation);
                    }

                    ASSERT(adapted_score > unadapted_score);
                    ASSERT(adapted_score > random_guess);
                }
            }
        };
    };
}
}
