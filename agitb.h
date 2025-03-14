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
	requires InputPredictor<Cortex, Pattern>and BitProvider<Pattern>
class Testbed
{
	using util = TestbedUtils<Cortex, Pattern, SimulatedInfinity>;

public:
	static void run()
	{
		std::clog << "Artificial General Intelligence Testbed\n\n";

		testbed[5].second(0);

		for (const auto& test : testbed) {
			std::clog << test.first << std::endl;
			repeat(test.second);
		}
				
		std::clog << green("\nPASS\n");
	}

private:
	static void repeat(void (*test)(time_t))
	{
		const std::string back(50, '\b');

		try {
			for (time_t t = 1; t <= SimulatedInfinity; ++t) {
				std::clog<< t << '/' << SimulatedInfinity << back ;

				const time_t temporal_sequence_length = estimate_test_difficulty();
				test(temporal_sequence_length);
			}
		}
		catch (const Error& err) {
			std::clog << red("\nAssertion failed") << err.what() << std::endl;
			exit(-1);
		}
	}
	static time_t estimate_test_difficulty()
	{
		for (time_t difficulty = 2; difficulty < SimulatedInfinity; ++difficulty) {
			Cortex C;
			const vector<Pattern> input = util::circular_random_sequence(difficulty);
			if (!util::adapt(C, input))
				return difficulty - 1;
		}
		return SimulatedInfinity;
	}

	static inline const vector<std::pair<const char*, void(*)(time_t)>> testbed =
	{
		{
			"#1 Genesis (All cortices begin in a completely blank, bias-free state.)",
			[](time_t) {
				Cortex C;

				ASSERT(C == Cortex{});	// unbiased start
			}
		},
		{
			"#2 Knowledge (A change in state indicates bias.)",
			[](time_t) {
				const Pattern bias = util::random_pattern();

				Cortex C;
				C << bias;

				ASSERT(C != Cortex{});
			}
		},
		{
			"#3 Determinism (Identical experiences produce an identical state.)",
			[](time_t) {
				const vector<Pattern> experience = util::random_sequence(SimulatedInfinity);

				Cortex C, D;
				C << experience;
				D << experience;

				ASSERT(C == D);
			}
		},
		{
			"#4 Sensitivity(The cortex exhibits chaos-like sensitivity to initial conditions.)",
			[](time_t) {
				const Pattern initial_condition = util::random_pattern();
				const Pattern inverted_condition = util::invert(initial_condition);
				const vector<Pattern> life = util::random_sequence(SimulatedInfinity);

				Cortex C, D;
				C << initial_condition << life;
				D << inverted_condition << life;

				ASSERT(C != D);
			}
		},
		{
			"#5 Time (The input order is inherently temporal and crucial to the process.)",
			[](time_t) {
				const vector<Pattern> two_patterns = util::random_sequence(2);

				Cortex C, D;
				C << two_patterns[0] << two_patterns[1];
				D << two_patterns[1] << two_patterns[0];

				ASSERT(C != D);
			}
		},
		{
			"#6 RefractoryPeriod (Each spike (1) must be followed by a no-spike (0).)",
			[](time_t) {
				const Pattern no_spikes;
				const Pattern single_spike = util::random_spike();
				const vector<Pattern> no_consecutive_spikes = { single_spike, no_spikes };
				const vector<Pattern> consecutive_spikes = { single_spike, single_spike };

				Cortex C, D;

				ASSERT(util::adapt(C, no_consecutive_spikes));
				ASSERT(not util::adapt(D, consecutive_spikes));
			}
		},
		{
			"#7 Scalability (The system can adapt to predict also longer sequences.)",
			[](time_t temporal_sequence_length) {
				util::adaptable_random_sequence(temporal_sequence_length + 1);  // throw

				ASSERT(true);   // nothrow
			}
		},
		{
			"#8 Stagnation (You can't teach an old dog new tricks.)",
			[](time_t temporal_sequence_length) {
				auto indefinitely_adaptable = [&](Cortex& dog) -> bool {
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						vector<Pattern> new_trick = util::adaptable_random_sequence(temporal_sequence_length);
						if (not util::adapt(dog, new_trick))
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

				ASSERT(adaptation_time_can_differ_across_sequences());   // rejects the null hypothesis
			}
		},
		{
			"#10 Experience (Adaptation time depends on the state of the cortex.)",
			[](time_t temporal_sequence_length) {
				// Null Hypothesis: Adaptation time is independent of the state of the cortex
				auto adaptation_time_can_differ_across_cortices = [&]() -> bool {
					const vector<Pattern> target_sequence = util::adaptable_random_sequence(temporal_sequence_length);
					Cortex C;
					const time_t default_time = util::time_to_repeat(C, target_sequence);
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						Cortex D = util::random_cortex();
						time_t random_time = util::time_to_repeat(D, target_sequence);
						if (default_time != random_time)
							return true;
					}
					return false;
				};

				ASSERT(adaptation_time_can_differ_across_cortices());   // rejects the null hypothesis
			}
		},
		{
			"#11 Unobservability (Different internal states can produce identical behaviour.)",
			[](time_t temporal_sequence_length) {
				// Null Hypothesis: "Different cortices cannot produce identical behavior."
				auto behaviour_can_be_identical_across_cortices = [&]() -> bool {
					const time_t nontrivial_problem_length = 2;
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						const vector<Pattern> target_behaviour = util::adaptable_random_sequence(nontrivial_problem_length);

						Cortex C{}, D = util::random_cortex();
						util::adapt(C, target_behaviour);
						util::adapt(D, target_behaviour);

						ASSERT(C != D);
						if (util::behaviour(C) == util::behaviour(D))
							return true;    // C != D && behaviour(C) == behaviour(D)
					}
					return false;
				};

				ASSERT(behaviour_can_be_identical_across_cortices());   // rejects the null hypothesis
			}
		},
		{
			"#12 Advantage (Adapted models predict more accurately.)",
			[](time_t temporal_sequence_length) {
				const size_t random_guess = SimulatedInfinity * Pattern::size() / 2;
				size_t adapted_score = 0, unadapted_score = 0;
				for (time_t time = 0; time < SimulatedInfinity; ++time) {
					const vector<Pattern> facts = util::adaptable_random_sequence(temporal_sequence_length);
					const Pattern disruption = util::random_pattern();
					const Pattern expectation = facts[0];

					Cortex A;
					util::adapt(A, facts);
					A << disruption << facts;
					adapted_score += util::count_matches(A.predict(), expectation);

					Cortex U;
					U << disruption << facts;
					unadapted_score += util::count_matches(U.predict(), expectation);
				}

				ASSERT(adapted_score > unadapted_score);
				ASSERT(adapted_score > random_guess);
			}
		}
	};
};
}
}