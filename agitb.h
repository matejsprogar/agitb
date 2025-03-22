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
#include <vector>


#include "concepts.h"
#include "utils.h"

#define ASSERT(expression) (void)((!!(expression)) || (throw AGI::Error(__FILE__, __LINE__, #expression), 0))



namespace sprogar {
namespace AGI {
using std::string;

template <typename XCortex, typename XPattern, size_t SimulatedInfinity = 1000>
	requires InputPredictor<XCortex, XPattern> && Indexable<XPattern>
class Testbed
{
	using Pattern = utils::Pattern<XPattern>;
	using Sequence = utils::Sequence<Pattern>;
	using Cortex = utils::Cortex<XCortex, Pattern, Sequence, SimulatedInfinity>;
	using Misc = utils::Misc<Cortex, Pattern, Sequence, SimulatedInfinity>;

public:
	static void run(time_t temporal_sequence_length, size_t repetitions = 100)
	{
		std::clog << "Artificial General Intelligence Testbed\n\n";
		std::clog << "Testing with temporal sequences of " << temporal_sequence_length << " patterns:\n";

		for (const auto& [info, test] : testbed) {
			std::clog << info << std::endl;
			repeat(test, temporal_sequence_length, repetitions);
		}
				
		std::clog << green("\nPASS\n");
	}

private:
	static void repeat(void (*test)(time_t), const time_t temporal_sequence_length, size_t repetitions)
	{
		const string go_back(50, '\b');

		try {
			for (size_t r = 1; r <= repetitions; ++r) {
				std::clog<< r << '/' << repetitions << go_back;

				test(temporal_sequence_length);
			}
		}
		catch (const Error& err) {
			std::clog << red("\nAssertion failed") << err.what() << std::endl;
			exit(-1);
		}
	}

	static inline const std::vector<std::pair<string, void(*)(time_t)>> testbed =
	{
		{
			"#1 Genesis (All cortices begin in a completely blank, bias-free state.)",
			[](time_t) {
				Cortex C;

				ASSERT(C == Cortex{});				// The initial state represents absence of bias.
				ASSERT(C.predict() == Pattern{});	// No spikes indicate an unbiased initial prediction.
			}
		},
		{
			"#2 Knowledge (A change in state indicates bias.)",
			[](time_t) {
				Cortex C;
				C << Pattern::random();

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
				const Pattern p = Pattern::random();
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
				const Pattern p = Pattern::random();
				const Sequence no_consecutive_spikes = { p, Pattern::random(p) };
				const Sequence consecutive_spikes = { p, p };

				Cortex C, D;

				ASSERT(C.adapt(no_consecutive_spikes));
				ASSERT(not D.adapt(consecutive_spikes) || p == Pattern{});
			}
		},
		{
			"#7 Scalability (The system can adapt to predict also longer sequences.)",
			[](time_t temporal_sequence_length) {
				Misc::adaptable_random_sequence(temporal_sequence_length + 1);  // throw?

				ASSERT(true);   // nothrow
			}
		},
		{
			"#8 Stagnation (You can't teach an old dog new tricks.)",
			[](time_t temporal_sequence_length) {
				auto indefinitely_adaptable = [&](Cortex& dog) -> bool {
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						Sequence new_trick = Misc::adaptable_random_sequence(temporal_sequence_length);
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
			"#9 Input (Adaptation time depends on the input sequence content.)",
			[](time_t temporal_sequence_length) {
				// Null Hypothesis: Adaptation time is independent of the input sequence
				auto adaptation_time_can_differ_across_sequences = [=]() -> bool {
					Cortex D;
					const time_t default_time = D.time_to_repeat(Sequence::circular_random(temporal_sequence_length));
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						Sequence random = Sequence::circular_random(temporal_sequence_length);
						Cortex R;
						time_t random_time = R.time_to_repeat(random);
						if (default_time != random_time)
							return true;
					}
					return false;
				};

				ASSERT(adaptation_time_can_differ_across_sequences());   // rejects the null hypothesis
			}
		},
		{
			"#10 Experience (Adaptation time depends on the state of the cortex.)",
			[](time_t temporal_sequence_length) {
				// Null Hypothesis: Adaptation time is independent of the state of the cortex
				auto adaptation_time_depends_on_state = [&]() -> bool {
					const Sequence target_sequence = Misc::adaptable_random_sequence(temporal_sequence_length);
					Cortex D;
					const time_t default_time = D.time_to_repeat(target_sequence);
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						Cortex O = Cortex::random();
						time_t other_time = O.time_to_repeat(target_sequence);
						if (default_time != other_time)
							return true;
					}
					return false;
				};

				ASSERT(adaptation_time_depends_on_state());   // rejects the null hypothesis
			}
		},
		{
			"#11 Unobservability (Different internal states can produce identical Cortex::behaviour.)",
			[](time_t temporal_sequence_length) {
				// Null Hypothesis: "Different cortices cannot produce identical behavior."
				auto cortices_can_match_behavior = [&]() -> bool {
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
                        const Sequence trivial_behaviour = { Pattern{}, Pattern{} };

						Cortex C{}, D = Cortex::random();
						C.adapt(trivial_behaviour);
						D.adapt(trivial_behaviour);

						bool counterexample_found = C != D && C.behaviour() == D.behaviour();
						if (counterexample_found)
							return true;
					}
					return false;
				};

				ASSERT(cortices_can_match_behavior());   // rejects the null hypothesis
			}
		},
		{
			"#12 Advantage (Adapted models predict more accurately.)",
			[](time_t temporal_sequence_length) {
				const size_t random_guess = SimulatedInfinity * Pattern::size() / 2;
				size_t adapted_score = 0, unadapted_score = 0;
				for (time_t time = 0; time < SimulatedInfinity; ++time) {
					const Sequence facts = Misc::adaptable_random_sequence(temporal_sequence_length);
					const Pattern disruption = Pattern::random();
					const Pattern expectation = facts[0];

					Cortex A;
					A.adapt(facts);
					A << disruption << facts;
					adapted_score += Pattern::count_matches(A.predict(), expectation);

					Cortex U;
					U << disruption << facts;
					unadapted_score += Pattern::count_matches(U.predict(), expectation);
				}

				ASSERT(adapted_score > unadapted_score);
				ASSERT(adapted_score > random_guess);
			}
		}
	};
};
}
}