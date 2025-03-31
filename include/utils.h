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

			template <Indexable TInput>
			class Input : public TInput
			{
			public:
				Input() : TInput() {}
				Input(const TInput& src) : TInput(src) {}
				Input(const Input& src) : TInput(src) {}
				Input(Input&& src) : TInput(std::move(src)) {}
				Input& operator=(const Input& src) { if (this != &src) TInput::operator=(src); return *this; }

				// Count the number of matching bits between two inputs.
				static size_t count_matches(const Input& a, const Input& b)
				{
					return std::ranges::count_if(std::views::iota(0ul, Input::size()), [&](size_t i) { return a[i] == b[i]; });
				}

				// Returns an input with spikes at random positions, except where explicitly required to have none.
				template<std::same_as<Input>... Inputs>
				static Input random(const Inputs&... off)
				{
					static std::mt19937 rng{ std::random_device{}() };
					static std::bernoulli_distribution bd(0.5);

					Input input;
					for (size_t i = 0; i < Input::size(); ++i)
						if (!(false | ... | off[i]))
							input[i] = bd(rng);

					return input;
				}
			};

			template <Indexable TInput>
				requires (!HasUnaryTilde<TInput>)
			auto operator ~(const Input<TInput>& input)
			{
				Input<TInput> bitwise_not{};
				for (size_t i = 0; i < TInput::size(); ++i) 
                    bitwise_not[i] = !input[i];
				return bitwise_not;
			}

			template <typename Input>
			class Sequence : public std::vector<Input>
			{
			public:
				template<typename... Args>
				Sequence(Args&&... args) : std::vector<Input>(std::forward<Args>(args)...) {}
				Sequence(std::initializer_list<Input> il) : std::vector<Input>(il) {}

                bool operator !() const { return std::vector<Input>::empty(); }

				// Returns a random sequence of inputs with a specified length.
				static Sequence random(time_t temporal_pattern_length)
				{
					if (0 == temporal_pattern_length)
						return Sequence{};

					Sequence sequence{};
					sequence.reserve(temporal_pattern_length);

					sequence.push_back(Input::random());
					while (sequence.size() < temporal_pattern_length)
						sequence.push_back(Input::random(sequence.back()));

					return sequence;
				}

				// Returns a random sequence of inputs with a specified length, exhibiting a circular property 
				// where the first input incorporates refractory periods for the last input in the sequence.
				static Sequence circular_random(time_t circle_length)
				{
					if (circle_length < 2)
						return Sequence{ circle_length, Input{} };

					Sequence sequence = Sequence::random(circle_length);

					sequence.pop_back();
					sequence.push_back(Input::random(sequence.back(), sequence.front()));

					return sequence;
				}
			};

			template <typename TCortex, Indexable Input, typename Sequence, size_t SimulatedInfinity>
				requires InputPredictor<TCortex, Input>
			class Cortex : public TCortex
			{
			public:
                Cortex() : TCortex() {}
				Cortex(Cortex&& src) : TCortex(std::move(src)) {}
				Cortex(const TCortex& src) : TCortex(src) {}
				Cortex& operator=(const Cortex& src) { if (this != &src) TCortex::operator=(src); return *this; }
            
				template<typename... Args>
				Cortex(Args&&... args) : TCortex(std::forward<Args>(args)...) {}

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

				Input predict() const { return TCortex::predict(); }
				Cortex& operator << (const Input& p) { (TCortex&)(*this) << (p); return *this; }

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

					for (const Input& in : inputs) {
						predictions.push_back(predict());
						*this << in;
					}
					return predictions;
				}
			};

			template <typename Cortex, typename Input, typename Sequence, size_t SimulatedInfinity>
			class Misc {
			public:
				// Returns an adaptable random sequence of non-empty inputs with a specified length.
				static Sequence adaptable_random_pattern(time_t temporal_pattern_length)
				{
					const Sequence empty_sequence{temporal_pattern_length, Input{}};
					for (time_t time = 0; time < SimulatedInfinity; ++time) {
						Cortex C{};
						Sequence sequence = Sequence::circular_random(temporal_pattern_length);
						if (sequence == empty_sequence)
							continue;

						if (C.adapt(sequence)) // not every circular sequence is inherently adaptable.
							return sequence;
					}
                    std::cerr << std::format("\n{} No temporal pattern spanning {} inputs was found.\n\n", red("Error:"), temporal_pattern_length);
                    exit(-1);
				}
			};
		}
	}
}