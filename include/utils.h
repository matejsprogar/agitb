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

#include "requirements.h"

namespace sprogar {

inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }

namespace AGI {
inline namespace utils {
    using time_t = size_t;

    // Count the number of matching bits between two inputs.
    template <typename TInput>
    requires (Indexable<TInput>)
    class Input : public TInput {
        public:
            using TInput::reference;
            using TInput::operator[];
            using TInput::size;

            Input() = default;
            Input(const TInput& src) : TInput(src) {}
            Input(const Input&) = default;
            Input(Input&&) = default;
            
            Input& operator = (const Input& oth) = default;
            Input& operator = (Input&& oth) = default;
            
            friend bool operator ==(const Input& a, const Input& b) { 
                return static_cast<const TInput&>(a) == static_cast<const TInput&>(b); 
            }

            static size_t count_matches(const TInput& a, const Input& b)
            {
                return std::ranges::count_if(std::views::iota(0ul, Input{}.size()), [&](size_t i) { return a[i] == b[i]; });
            }

            // Returns an input with spikes at random positions, except where explicitly required to have none.
            template<typename... Inputs>
            static Input random(const Inputs&... off)
            {
                static std::mt19937 rng{ std::random_device{}() };
                static std::bernoulli_distribution bd(0.5);

                Input input;
                for (size_t i = 0; i < Input{}.size(); ++i)
                    if (!(false | ... | off[i]))
                        input[i] = bd(rng);

                return input;
            }
            
            friend Input operator ~(const Input& self)
            requires (!HasUnaryTilde<TInput>)
            {
                Input bitwise_not{};
                for (size_t i=0; i<Input{}.size(); ++i)
                    bitwise_not[i] = !self[i];
                return bitwise_not;
            }
            
            typename TInput::reference operator [](const size_t idx) { return TInput::operator[](idx); }
            bool operator [](const size_t idx) const { return TInput::operator[](idx); }
    };

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

            sequence.push_back(Input::random());
            while (sequence.size() < length)
                sequence.push_back(Input::random(sequence.back()));

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
            sequence.push_back(Input::random(sequence.back(), sequence.front()));

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
        static Sequence trivial_pattern(time_t temporal_pattern_length)
        {
            Sequence sequence;
            sequence.resize( temporal_pattern_length );
            sequence.back() = ~Input{};

            return sequence;    // [{0...0}, {0...0}, ..., {0...0}, {1...1}]
        }
        size_t period() const
        {
            const Sequence& sequence = *this;
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

    template <typename TCortex, typename Input, size_t SimulatedInfinity>
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
                predictions.push_back(prediction());
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

        Cortex& operator << (const Input& p) { (TCortex&)(*this) << (p); return *this; }
        Input prediction() const { return TCortex::prediction(); }

        // Sequentially feeds each element of the range to the target.
        template <std::ranges::range Range>
            requires std::same_as<std::ranges::range_value_t<Range>, Input>
        Cortex& operator << (Range&& range)
        {
            for (auto&& elt : range)
                *this << elt;
            return *this;
        }

        static Sequence adaptable_random_pattern(time_t temporal_pattern_length)
        {
            for (time_t time = 0; time < SimulatedInfinity; ++time) {
                Sequence sequence = Sequence::nontrivial_circular_random(temporal_pattern_length);
                if (sequence.period() != temporal_pattern_length)
                    continue;

                Cortex C;
                if (C.adapt(sequence)) // not every circular sequence is inherently adaptable.
                    return sequence;
            }
            return Sequence{};
        }
    private:
        // Modifies the cortex by processing the given inputs and returns its corresponding predictions.
        Sequence predict(const Sequence& inputs)
        {
            Sequence predictions{};
            predictions.reserve(inputs.size());

            for (const Input& in : inputs) {
                predictions.push_back(prediction());
                *this << in;
            }
            return predictions;
        }
    };
    
    template <typename Cortex>
    bool predictions_in_constant_time(Cortex &C, const int K = 10, const double EPS = 0.15)
    {
        using namespace std::chrono;
        using Input = Cortex::Sequence::value_type;
        
        auto time_once = [&](Cortex& C, const Input& x) {
            auto t0 = high_resolution_clock::now(); 
            C << x;
            auto t1 = high_resolution_clock::now();
            return t1 - t0;
            
        };
        auto is_flat = [=](auto tm1, auto tm2) -> bool {
            auto r = tm1 / tm2;
            return std::fabs(r - 1.0) <= EPS;
        };

        C << Input{};
        
        Input x{};
        auto prev = time_once(C, x), time0 = prev;
        for (int i = 0; i < K; ++i) {
            x = Input::random(x);
            auto curr = time_once(C, x);
            if (!is_flat(curr, prev)) 
                return false;
            prev = curr;
        }

        auto timeK = prev;
        return is_flat(time0, timeK);
    }

    }
}
}
