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
#include <bitset>
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

    template<typename T>
    std::vector<T> shuffle(const std::vector<T>& vec)
    {
        std::vector<T> result(vec);

        std::mt19937 rng{ std::random_device{}() };
        std::ranges::shuffle(result, rng);

        return result;
    }

    // Count the number of matching bits between two inputs.
    template <typename Input>
    size_t count_matches(const Input& a, const Input& b)
    {
        return std::ranges::count_if(std::views::iota(0ul, Input{}.size()), [&](size_t i) { return a[i] == b[i]; });
    }

    // Returns an input with spikes at random positions, except where explicitly required to have none.
    template<typename Input, typename... Inputs>
    Input random(const Inputs&... off)
    {
        static std::mt19937 rng{ std::random_device{}() };
        static std::bernoulli_distribution bd(0.5);

        Input input;
        for (size_t i = 0; i < Input{}.size(); ++i)
            if (!(false | ... | off[i]))
                input[i] = bd(rng);

        return input;
    }

    template <Indexable Input>
    requires (!HasUnaryTilde<Input>)
    auto operator ~(const Input& input)
    {
        Input bitwise_not{};
        for (size_t i = 0; i < Input{}.size(); ++i)
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
        bool is_trivial() const {
            static const Input no_spikes;
            for (const Input& in : *this)
                if (in != no_spikes)
                    return false;
            return true;
        }

        // Returns a random sequence of inputs with a specified length.
        static Sequence random(time_t length)
        {
            if (0 == length)
                return Sequence{};

            Sequence sequence{};
            sequence.reserve(length);

            sequence.push_back(utils::random<Input>());
            while (sequence.size() < length)
                sequence.push_back(utils::random<Input>(sequence.back()));

            return sequence;
        }

        // Returns a random sequence of inputs with a specified length, exhibiting a circular property 
        // where the first input incorporates refractory periods for the last input in the sequence.
        static Sequence circular_random(time_t length)
        {
            if (length < 2)
                return Sequence{ length, Input{} };

            Sequence sequence = Sequence::random(length);

            sequence.pop_back();
            sequence.push_back(utils::random<Input>(sequence.back(), sequence.front()));

            return sequence;
        }
        static Sequence nontrivial_circular_random(time_t length)
        {
            while (true) {
                auto sequence = circular_random(length);
                if (!sequence.is_trivial())
                    return sequence;
            }
        }
    };

    template <typename TCortex, typename Input, size_t SimulatedInfinity, size_t MaxAdaptationTime>
    requires InputPredictor<TCortex, Input> and Indexable<Input>
    class Cortex : public TCortex
    {
    public:
        using Sequence = utils::Sequence<Input>;

        Cortex() = default;
        Cortex(const Cortex& src) = default;
        Cortex(Cortex&& src) = default;
        Cortex& operator=(const Cortex& src) = default;
    
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
            for (time_t time = 0; time < MaxAdaptationTime; time += inputs.size()) {
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
            requires std::same_as<std::ranges::range_value_t<Range>, Input>
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

    template <typename Cortex, size_t MaxAdaptationTime>
    class Misc {
    public:
        using Sequence = Cortex::Sequence;
        // Returns a random, adaptable and non-trivial sequence with the specified period.
        static Sequence adaptable_random_pattern(time_t pattern_period)
        {
            for (time_t time = 0; time < MaxAdaptationTime; ++time) {
                Sequence sequence = Sequence::nontrivial_circular_random(pattern_period);
                if (period(sequence) != pattern_period)
                    continue;

                Cortex C{};
                if (C.adapt(sequence)) // not every circular sequence is inherently adaptable.
                    return sequence;
            }
            return Sequence{};
        }
    private:
        static size_t period(const Sequence& sequence)
        {
            const size_t n = sequence.size();
            for (size_t period = 1; period <= n / 2; ++period) {
                bool is_periodic = true;
                for (size_t i = period; i < n; ++i) {
                    if (sequence[i] != sequence[i - period]) {
                        is_periodic = false;
                        break;
                    }
                }
                if (is_periodic) return period;
            }
            return n;
        }
    };
}
}
}