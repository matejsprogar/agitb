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

#include "concepts.h"

namespace sprogar {

    inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
    inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }

    namespace AGI {
        using time_t = size_t;
        using std::vector;
        using std::string;
        
        const int impossible_task = 42;
    
        template <typename T, std::ranges::range Range>
            requires InputPredictor<T, std::ranges::range_value_t<Range>>
        T& operator << (T& target, Range&& range) {
            for (auto&& elt : range)
                target << elt;
            return target;
        }
        
        class Error : public std::runtime_error
        {
        public:
            Error(const string& file, int line, const string& msg) : std::runtime_error{std::format(" in {}\nLine {}: {}", file, line, msg)} {}
        };

        template <typename Cortex, typename Pattern, size_t SimulatedInfinity>
            requires InputPredictor<Cortex, Pattern> and BitProvider<Pattern>
        class TestbedUtils
        {
            
        public:
            // Count the number of matching bits between two patterns.
            static size_t count_matches(const Pattern& a, const Pattern& b)
            {
                return std::ranges::count_if(std::views::iota(0ul, Pattern::size()), [&](size_t i) { return a[i] == b[i]; });
            }
            
            // Mutate a pattern by randomly flipping a single bit.
            static Pattern mutate(Pattern pattern)
            {
                static std::uniform_int_distribution<size_t> dist(0, Pattern::size() - 1);

                const size_t random_index = dist(rng);
                pattern[random_index] = !pattern[random_index];
                return pattern;
            }

            // Returns a pattern where each bit is set randomly unless explicitly required to remain off.
            template<std::same_as<Pattern>... Patterns>
            static Pattern random_pattern(const Patterns&... off)
            {
                static std::bernoulli_distribution bd(0.5);

                Pattern pattern;
                for (size_t i = 0; i < Pattern::size(); ++i)
                    if (!(false | ... | off[i]))
                        pattern[i] = bd(rng);

                return pattern;
            }
            
            // Returns a random sequence of patterns with a specified length.
            static vector<Pattern> random_sequence(time_t temporal_sequence_length)
            {
                assert(temporal_sequence_length > 0);
                
                vector<Pattern> sequence;
                sequence.reserve(temporal_sequence_length);

                sequence.push_back(random_pattern());
                while (sequence.size() < temporal_sequence_length)
                    sequence.push_back(random_pattern(sequence.back()));

                return sequence;
            }
            
            // Returns a random sequence of patterns with a specified length, exhibiting a circular property 
            // where the first pattern incorporates refractory periods for the last pattern in the sequence.
            static vector<Pattern> circular_random_sequence(time_t circle_length)
            {
                assert(circle_length > 1);
                
                vector<Pattern> sequence = random_sequence(circle_length);

                sequence.pop_back();
                sequence.push_back(random_pattern(sequence.back(), sequence.front()));

                return sequence;
            }
            
            // Returns a learnable random sequence of patterns with a specified length.
            static vector<Pattern> learnable_random_sequence(time_t temporal_sequence_length)
            {
                for (time_t time = 0; time < SimulatedInfinity; ++time) {
                    Cortex C;
                    vector<Pattern> sequence = circular_random_sequence(temporal_sequence_length);
                    
                    if (adapt(C, sequence)) // not every circular sequence is inherently learnable.
                        return sequence;
                }
                throw Error(__FILE__, __LINE__, std::format("Unable to find a learnable sequence of {} patterns.", temporal_sequence_length));
            }
            
            // Creates a randomly initialized cortex object.
            static Cortex random_cortex(time_t random_strength)
            {
                Cortex C;
                C << random_sequence(random_strength);
                return C;
            }

            // Feeds both cortices their own predictions for a simulated infinite duration and returns true only if all predictions remain identical.
            static bool equal_behaviour(Cortex& A, Cortex& B)
            {
                for (time_t time = 0; time < SimulatedInfinity; ++time) {
                    const Pattern prediction = A.predict();
                    if (prediction != B.predict())
                        return false;
                    A << prediction;
                    B << prediction;
                }
                return true;
            }
            
            // Modifies the cortex by processing the given inputs and returns its corresponding predictions.
            static vector<Pattern> predict(Cortex& C, const vector<Pattern>& inputs)
            {
                vector<Pattern> predictions;
                predictions.reserve(inputs.size());

                for (const Pattern& in : inputs) {
                    predictions.push_back(C.predict());
                    C << in;
                }
                return predictions;
            }
            
            // Adapts the cortex to the given input sequence and returns the time required to achieve perfect prediction.
            static time_t time_to_repeat(Cortex& C, const vector<Pattern>& inputs)
            {
                for (time_t time = 0; time < SimulatedInfinity; time += inputs.size()) {
                    if (predict(C, inputs) == inputs)
                        return time;
                }
                return SimulatedInfinity;
            }
            
            // Adapts the cortex to the given input sequence and returns true if it achieves 100% accurate prediction.
            static bool adapt(Cortex& C, const vector<Pattern>& inputs)
            {
                return time_to_repeat(C, inputs) < SimulatedInfinity;
            }

        private:
            inline static thread_local std::mt19937 rng{ std::random_device{}() };
        };
    }
}