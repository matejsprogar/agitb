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

#include <string>
#include <format>
#include <algorithm>
#include <ranges>
#include <random>

#include "concepts.h"

namespace sprogar {

	inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
	inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }

	namespace AGI {
		inline namespace utils {
			using time_t = size_t;
			using std::vector;
			using std::string;

			class Error : public std::runtime_error
			{
			public:
				Error(const string& file, int line, const string& msg) : std::runtime_error{ std::format(" in {}\nLine {}: {}", file, line, msg) } {}
			};

			template <Indexable Pattern>
				requires (!HasUnaryTilde<Pattern>)
			Pattern operator ~(const Pattern& pattern)
			{
				Pattern bitwise_not{};
				for (size_t i = 0; i < Pattern::size(); ++i)
					bitwise_not[i] = !pattern[i];
				return bitwise_not;
			}

			template <Indexable XPattern>
			class Pattern : public XPattern
			{
			public:
				Pattern() : XPattern() {}
				Pattern(const XPattern& src) : XPattern(src) {}
				Pattern(XPattern&& src) : XPattern(std::move(src)) {}


				// Count the number of matching bits between two patterns.
				static size_t count_matches(const Pattern& a, const Pattern& b)
				{
					return std::ranges::count_if(std::views::iota(0ul, Pattern::size()), [&](size_t i) { return a[i] == b[i]; });
				}

				// Returns a pattern with spikes at random positions, except where explicitly required to have none.
				template<std::same_as<Pattern>... Patterns>
				static Pattern random(const Patterns&... off)
				{
					static std::mt19937 rng{ std::random_device{}() };
					static std::bernoulli_distribution bd(0.5);

					Pattern pattern;
					for (size_t i = 0; i < Pattern::size(); ++i)
						if (!(false | ... | off[i]))
							pattern[i] = bd(rng);

					return pattern;
				}
			};

			template <typename Pattern>
			class Sequence : public std::vector<Pattern>
			{
			public:
				template<typename... Args>
				Sequence(Args&&... args) : std::vector<Pattern>(std::forward<Args>(args)...) {}
				Sequence(std::initializer_list<Pattern> il) : std::vector<Pattern>(il) {}

				// Returns a random sequence of patterns with a specified length.
				static Sequence random(time_t temporal_sequence_length)
				{
					if (0 == temporal_sequence_length)
						return Sequence{};

					Sequence sequence{};
					sequence.reserve(temporal_sequence_length);

					sequence.push_back(Pattern::random());
					while (sequence.size() < temporal_sequence_length)
						sequence.push_back(Pattern::random(sequence.back()));

					return sequence;
				}

				// Returns a random sequence of patterns with a specified length, exhibiting a circular property 
				// where the first pattern incorporates refractory periods for the last pattern in the sequence.
				static Sequence circular_random(time_t circle_length)
				{
					if (circle_length < 2)
						return Sequence{ circle_length, Pattern{} };

					Sequence sequence = Sequence::random(circle_length);

					sequence.pop_back();
					sequence.push_back(Pattern::random(sequence.back(), sequence.front()));

					return sequence;
				}
			};

			template <typename XCortex, Indexable Pattern, typename Sequence, size_t SimulatedInfinity>
				requires InputPredictor<XCortex, Pattern>
			class Cortex : public XCortex
			{
			public:
				template<typename... Args>
				Cortex(Args&&... args) : XCortex(std::forward<Args>(args)...) {}

				// Creates a randomly initialized cortex object.
				static Cortex random()
				{
					const time_t arbitrary_random_strength = 10;

					Cortex C;
					C << Sequence::random(arbitrary_random_strength);
					return C;
				}

				// Iteratively feeds the cortex its own predictions and returns the resulting predictions over a specified timeframe.
				Sequence behaviour(time_t timeframe = SimulatedInfinity)
				{
					Sequence predictions{};
					predictions.reserve(timeframe);

					while (predictions.size() < timeframe) {
						predictions.push_back(predict());
						*this << predictions.back();
					}
					return predictions;
				}

				// Adapts the cortex to the given input sequence and returns the time required to achieve perfect prediction.
				time_t time_to_repeat(const Sequence& inputs)
				{
					for (time_t time = 0; time < SimulatedInfinity; time += inputs.size()) {
						if (predict(inputs) == inputs)
							return time;
					}
					return SimulatedInfinity;
				}

				// Adapts the cortex to the given input sequence and returns true if perfect prediction is achieved.
				bool adapt(const Sequence& inputs)
				{
					return time_to_repeat(inputs) < SimulatedInfinity;
				}

				Pattern predict() const { return XCortex::predict(); }
				// Cortex& operator << (const Pattern& p) { (XCortex&)(*this) << (p); return *this; }
				friend Cortex& operator << (Cortex& C, const Pattern& p) { (XCortex&)C << p; return C; }

				// Sequentially feeds each element of the range to the target.
				template <std::ranges::range Range>
				Cortex& operator << (Range&& range)
				{
					for (auto&& elt : range)
						*this << elt;
					return *this;
				}

			private:
				// Modifies the cortex by processing the given inputs and returns its corresponding predictions.
				Sequence predict(const Sequence& inputs)
				{
					Sequence predictions{};
					predictions.reserve(inputs.size());

					for (const Pattern& in : inputs) {
						predictions.push_back(predict());
						*this << in;
					}
					return predictions;
				}
			};

			template <typename Cortex, typename Pattern, typename Sequence, size_t SimulatedInfinity>
			class Misc {
			public:
				// Returns an adaptable non-empty random sequence of patterns with a specified length.
				static Sequence adaptable_random_sequence(time_t temporal_sequence_length)
				{
					const Sequence empty_sequence{ temporal_sequence_length, Pattern{} };
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						Cortex C{};
						Sequence sequence = Sequence::circular_random(temporal_sequence_length);
						if (sequence == empty_sequence)
							continue;

						if (C.adapt(sequence)) // not every circular sequence is inherently adaptable.
							return sequence;
					}
					throw Error(__FILE__, __LINE__, std::format("Unable to find a {}-pattern sequence for adaptation.", temporal_sequence_length));
				}
			};
		}
	}
}