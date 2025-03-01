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
        inline namespace helpers {

            template <typename T, std::ranges::range Range>
                requires InputPredictor<T, std::ranges::range_value_t<Range>>
            T& operator << (T& target, Range&& range) {
                for (auto&& elt : range)
                    target << elt;
                return target;
            }
            
            template <BitProvider Pattern>
            size_t count_matches(const Pattern& a, const Pattern& b)
            {
                return std::ranges::count_if(std::views::iota(0ul, Pattern::size()), [&](size_t i) { return a[i] == b[i]; });
            }

            template <BitProvider Pattern>
            Pattern mutate(Pattern pattern)
            {
                static std::mt19937 rng{ std::random_device{}() };
                static std::uniform_int_distribution<size_t> dist(0, Pattern::size() - 1);
                const size_t random_index = dist(rng);

                pattern[random_index] = !pattern[random_index];
                return pattern;
            }
        }
    }
}