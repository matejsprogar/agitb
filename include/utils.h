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

namespace sprogar {

inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }
inline std::string yellow(const char* msg) { return std::format("\033[93m{}\033[0m", msg); }

namespace AGI {
inline namespace utils {
    using time_t = size_t;

    template <typename Input>
    size_t count_matches(const Input& a, const Input& b)
    {
        //return std::ranges::count_if(std::views::iota(0ul, InputWidth), [&](size_t i) { return a[i] == b[i]; });
        return Input{}.size() - (a ^ b).count();
    }

    // Returns an input with spikes at random positions, except where explicitly required to have none.
    template<typename Input, typename... Inputs>
    requires (std::same_as<Input, Inputs> && ...)
    Input random(const Inputs&... turn_off)
    {
        static std::mt19937 rng{ std::random_device{}() };
        static std::bernoulli_distribution bd(0.5);

        Input input{};
        for (size_t i = 0; i < Input{}.size(); ++i)
            if (!(false | ... | turn_off[i]))
                input[i] = bd(rng);

        return input;
    }

    template <typename Input>
    class InputSequence : public std::vector<Input>
    {
    public:
        enum random_tag { random = 0 };
        enum circular_random_tag { circular_random = 0 };
        enum trivial_problem_tag { trivial_problem = 0 };

        InputSequence(std::initializer_list<Input> il) : std::vector<Input>(il) {}

        // constructs a random sequence of inputs with a specified length.
        InputSequence(random_tag, time_t length)
        {
            if (0 == length)
                return;

            using base = std::vector<Input>;
            base::reserve(length);

            base::push_back(utils::random<Input>());
            while (base::size() < length)
                base::push_back(utils::random<Input>(base::back()));
        }
        // constructs a random sequence of inputs with a specified length, exhibiting a circular property 
        // where the first input incorporates refractory periods for the last input in the sequence.
        InputSequence(circular_random_tag, time_t length) : InputSequence(random, length) {
            using base = std::vector<Input>;

            base::pop_back();
            base::push_back(utils::random<Input>(base::back(), base::front()));
        }

        // constructs a simple, easily adaptable sequence of inputs with a specified length.
        InputSequence(trivial_problem_tag, time_t length)
        {
            using base = std::vector<Input>;

            base::resize( length );
            base::back() = ~Input{};                // [{0...0}, {0...0}, ..., {0...0}, {1...1}]
        }
    };

    template <class Cortex>
    InputSequence<typename Cortex::InputType> adaptable_random_sequence(time_t length, time_t timeframe)
    {
        using InputSequence = InputSequence<typename Cortex::InputType>;
        auto period = [](const InputSequence& sequence)
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
            };

        // ensure the sequence is adaptable and has the desired period
        for (time_t time = 0; time < timeframe; ++time) {
            InputSequence seq{ InputSequence::circular_random, length };
            if (period(seq) != length)
                continue;

            Cortex C;
            if (C.adapt(seq, timeframe))   // not every circular sequence is inherently adaptable.
                return seq;
        };
        return InputSequence{}; // failed to find an adaptable InputSequence
    }




    
    template <typename Cortex, typename Input>
    concept InputPredictor = std::regular<Cortex> && requires(Cortex cortex, const Cortex ccortex, const Input input)
    {
        { cortex << input } -> std::convertible_to<Cortex&>;
       // { ccortex.prediction() } -> std::convertible_to<Input>;
    };    

    template <typename TCortex, typename Input>
    requires InputPredictor<TCortex, Input>
    class Cortex
    {
        TCortex cortex;
    public:
        using InputSequence = utils::InputSequence<Input>;
        using InputType = Input;

        enum random_tag { random = 0 };

        Cortex() = default;
        Cortex(const Cortex& src) = default;
        Cortex(Cortex&& src) = default;
        Cortex& operator=(const Cortex& src) = default;
        bool operator==(const Cortex& rhs) const = default;// { return cortex == rhs.cortex; }
    
        template<typename... Args>
        Cortex(Args&&... args) : cortex(std::forward<Args>(args)...) {}

        // Constructs a randomly initialized cortex object.
        Cortex(random_tag) : Cortex()
        {
            const time_t arbitrary_random_strength = 10;

            *this << InputSequence(InputSequence::random, arbitrary_random_strength);
        }

        // Iteratively feeds each cortex its own predictions and returns true if predictions match over a specified timeframe.
        static bool identical_behaviour(Cortex& A, Cortex& B, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; ++time) {
                if (A.prediction() != B.prediction())
                    return false;
                A << A.prediction();
                B << B.prediction();
            }
            return A.prediction() == B.prediction();
        }

        // Adapts the cortex to the given input sequence and returns the time required to achieve perfect prediction.
        time_t time_to_repeat(const InputSequence& inputs, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; time += inputs.size()) {
                if (predict(inputs) == inputs)
                    return time;
            }
            return timeframe;
        }

        // Adapts the cortex to the given input sequence and returns true if perfect prediction is achieved.
        bool adapt(const InputSequence& inputs, time_t timeframe)
        {
            return time_to_repeat(inputs, timeframe) < timeframe;
        }

        Cortex& operator << (const Input& p) { cortex << p; return *this; }
        Input prediction() const { return cortex.prediction(); }

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
        InputSequence predict(const InputSequence& inputs)
        {
            InputSequence predictions{};
            predictions.reserve(inputs.size());

            for (const Input& in : inputs) {
                predictions.push_back(prediction());
                *this << in;
            }
            return predictions;
        }
    };

    }
}
}
