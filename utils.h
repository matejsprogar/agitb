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

#include <format>
#include <algorithm>
#include <ranges>

#include "concepts.h"

namespace sprogar {

    inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
    inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }

    namespace AGI {
        using time_t = size_t;

    
        template <typename T, std::ranges::range Range>
            requires InputPredictor<T, std::ranges::range_value_t<Range>>
        T& operator << (T& target, Range&& range) {
            for (auto&& elt : range)
                target << elt;
            return target;
        }

        template <typename Cortex, typename Pattern, template<typename> typename TemporalSequence, size_t SimulatedInfinity>
            requires InputPredictor<Cortex, Pattern> and BitProvider<Pattern>
        class TestbedUtils
        {
        public:
            static size_t count_matches(const Pattern& a, const Pattern& b)
            {
                return std::ranges::count_if(std::views::iota(0ul, Pattern::size()), [&](size_t i) { return a[i] == b[i]; });
            }
            static Pattern mutate(Pattern pattern)
            {
                static std::uniform_int_distribution<size_t> dist(0, Pattern::size() - 1);
                const size_t random_index = dist(rng);

                pattern[random_index] = !pattern[random_index];
                return pattern;
            }

            // Each bit in the pattern is set randomly unless explicitly required to remain off.
            template<std::same_as<Pattern>... Patterns>
            static Pattern generate_random_pattern(const Patterns&... off)
            {
                //static thread_local std::mt19937 generator{ std::random_device{}() };
                static std::bernoulli_distribution bd(0.5);

                Pattern pattern;
                for (size_t i = 0; i < Pattern::size(); ++i)
                    if (!(false | ... | off[i]))
                        pattern[i] = bd(rng);

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

        private:
            inline static std::mt19937 rng{ std::random_device{}() };
        };
    }
}