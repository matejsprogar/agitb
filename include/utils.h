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

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <numeric>
#include <bitset>
#include <format>
#include <algorithm>
#include <ranges>
#include <random>
#include <cassert>

namespace sprogar {

#define ASSERT(expression) (void)((!!(expression)) || \
                            (std::cerr << std::format("\n\n{} in {}:{}\n{}\n\nrng_seed: {}\n", \
                                red("Assertion failed"), __FILE__, __LINE__, #expression, utils::rng_seed), \
                            exit(-1), 0))

inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }
inline std::string yellow(const char* msg) { return std::format("\033[93m{}\033[0m", msg); }

namespace AGI {
inline namespace utils {
    using time_t = size_t;

    constexpr time_t Infinity = std::numeric_limits<time_t>::max();

    static unsigned rng_seed = std::random_device{}();
    static std::mt19937 rng(rng_seed);

    template <typename M, typename T>
    concept InputPredictor = std::regular<M>
        and requires(M c, const T& t)
    {
        { c(t) } -> std::convertible_to<T>;
    };

    template <size_t BitsPerInput>
    size_t match_score(const std::bitset<BitsPerInput>& a, const std::bitset<BitsPerInput>& b)
    {
        return BitsPerInput - (a ^ b).count();
    }

    template <std::ranges::input_range R1, std::ranges::input_range R2>
    size_t match_score(const R1& r1, const R2& r2)
    {
        size_t count = 0;
        for (const auto& [x1, x2] : std::views::zip(r1, r2))
            count += match_score(x1, x2);

        return count;
    }
    
    bool random(double p) { 
        std::bernoulli_distribution bd(p);
        return bd(rng); 
    }
    size_t random(size_t min, size_t max_inclusive)
    {
        std::uniform_int_distribution<size_t> dist(min, max_inclusive);
        return dist(rng);
    }
    // Returns an input with spikes at random positions with given probability, except where explicitly required to have none.
    template<typename Input, typename... Inputs>
    requires (std::same_as<Input, Inputs> && ...)
    Input random_p(double p, const Inputs&... turn_off)
    {
        Input input{};
        for (size_t i = 0; i < Input{}.size(); ++i)
            if (!(false | ... | turn_off[i]))
                input[i] = random(p);

        return input;
    }
    // Returns an input with spikes at random positions, except where explicitly required to have none.
    template<typename Input, typename... Inputs>
        requires (std::same_as<Input, Inputs> && ...)
    Input random(const Inputs&... turn_off)
    {
        return random_p<Input>(0.5, turn_off...);
    }


    template <typename Input>
    class InputSequence : public std::vector<Input>
    {
        using base = std::vector<Input>;
    public:
        enum random_tag { random = 0 };
        enum circular_random_tag { circular_random = 0 };
        enum trivial_tag { trivial = 0 };

        InputSequence() {}
        InputSequence(std::initializer_list<Input> il) : std::vector<Input>(il) {}

        // constructs a random sequence of inputs with a specified length.
        InputSequence(random_tag, size_t length, Input start=utils::random<Input>())
        {
            if (0 == length)
                return;

            base::reserve(length);

            base::push_back(start);
            while (base::size() < length)
                base::push_back(utils::random<Input>(base::back()));
        }
        // constructs a random sequence of inputs with a specified length, exhibiting a circular property 
        // where the first input incorporates refractory periods for the last input in the sequence.
        InputSequence(circular_random_tag, size_t length, Input start=utils::random<Input>()) : InputSequence(random, length, start) {
            base::pop_back();
            base::push_back(utils::random<Input>(base::back(), base::front()));
        }

        // constructs a sequencerhythm x y x y ... z with a specified length
        InputSequence(trivial_tag, size_t length)
        {
            base::resize(length);
            base::back() = Input(1);
        }
     };

    template <typename ModelUnderTest, typename InputType, size_t SimulatedInfinity>
    requires InputPredictor<ModelUnderTest, InputType>
    class Model
    {
        
    public:
        using Input = InputType;
        using InputSequence = utils::InputSequence<Input>;

        enum random_tag { random = 0 };

        Model() = default;
        Model(const Model& src) = default;
        Model(Model&& src) = default;
        Model& operator=(const Model& src) = default;
        bool operator==(const Model& rhs) const = default;

        //template<typename... Args>
        //Model(Args&&... args) : model(std::forward<Args>(args)...) {}

        // Constructs a randomly initialized model by feeding it with random inputs.
        Model(random_tag, const time_t warm_up) : Model()
        {
            *this << InputSequence(InputSequence::random, warm_up);
        }
        Model(random_tag) : Model(random, utils::random(0, SimulatedInfinity))
        {
        }
        
        //////////////
        Input operator ()(const Input& p) { return current_prediction = model(p); }
        Model& operator << (const Input& p) { current_prediction = model(p); return *this; }
        ////////////////
        const Input& get_prediction() const { return current_prediction; }
        const Input& operator ()() const { return current_prediction; }

        // Sequentially feeds each element of the range to the target.
        template <std::ranges::range Range>
            //requires std::same_as<std::ranges::range_value_t<Range>, Input>
        Model& operator << (Range&& range)
        {
            for (auto&& elt : range)
                *this << elt;
            return *this;
        }

        auto learn_anything(const size_t length)
        {
            const Model starting_point = *this;
            for (size_t attempt = 0; attempt < SimulatedInfinity; ++attempt) {
                const InputSequence seq(InputSequence::circular_random, length);
                const time_t time = time_to_learn(seq);
                if (time < Infinity)
                    return std::make_pair(time, seq);

                Model M;
                if (M.learn(seq))
                    break;
                
                *this = starting_point;
            }
            return std::make_pair(Infinity, InputSequence());
        }

        // Constructs an adult: a model that has reached its capacity (see #6a) and then lived as long again, but never
        // shorter than a minimal life, so that a model cannot shorten its own test by failing early.
        // Every run lives the same life, which keeps failures reproducible; the caller's random state is left untouched.
        static Model adult(const size_t length)
        {
            const auto rng_state = rng;
            rng.seed();

            Model M;

            for (size_t adaptations=0; 
                M.learn_anything(length).first != Infinity and adaptations < SimulatedInfinity; 
                ++adaptations) 
            {}

            rng = rng_state;
            return M;
        }

        // Adapts the model to the given input sequence and returns the number of timesteps needed to learn the sequence.
        time_t time_to_learn(const InputSequence& inputs)
        {
            for (size_t iteration = 0; iteration < SimulatedInfinity; ++iteration) {
                if (process(inputs) == inputs)
                    return iteration * inputs.size();
            }
            return Infinity;
        }

        // Adapts the model to the given input sequence and returns true if perfect prediction is achieved.
        bool learn(const InputSequence& inputs)
        {
            return time_to_learn(inputs) < Infinity;
        }

        // Adapts the model to the given input sequence until it predicts the sequence perfectly twice in a row, so that
        // what it learned remains active; returns true if this happens within SimulatedInfinity passes.
        bool master(const InputSequence& inputs)
        {
            bool perfect_before = false;
            for (size_t pass = 0; pass < SimulatedInfinity; ++pass) {
                const bool perfect = process(inputs) == inputs;
                if (perfect and perfect_before)
                    return true;
                perfect_before = perfect;
            }
            return false;
        }

        bool behaves_identically(Model& B)
        {
            Input x = utils::random<Input>();
            for (size_t i = 0; i < SimulatedInfinity; ++i)
            {
                if ((*this)(x) != B(x))
                    return false;
                x = utils::random<Input>(x);
            }
            return true;
        }

        // Feeds the model its own predictions to generate a sequence of predictions.
        auto generate(size_t length)
        {
            return std::views::iota(std::size_t{ 0 }, length)
                | std::views::transform([&](std::size_t) {
                const Input prediction = get_prediction();
                *this << prediction;
                return prediction;
                    });
        }

    private:
        ModelUnderTest model;
        Input current_prediction;
        
        // Modifies the model by processing the given inputs and returns its corresponding predictions.
        InputSequence process(const InputSequence& inputs)
        {
            InputSequence predictions; predictions.reserve(inputs.size());

            for (const Input& in : inputs) {
                predictions.push_back(get_prediction());
                *this << in;
            }
            return predictions;
        }
    };
 
    // Generates a simple world governed by a single rule: at each time step, every bit of the input moves one position
    // to the right; bit 0 falls off and a new bit enters at bit 9 every 7 steps. Channels 7 apart therefore always fire
    // together, so learning which channel follows which cannot tell the rule apart from coincidences; the problem is to
    // apply the rule to an input never seen before.
    template <typename Input>
    struct world_generator
    {
        static auto generate()
        {
            const InputSequence<Input> world = {
                0b1000000100,
                0b0100000010,
                0b0010000001,
                0b0001000000,
                0b0000100000,
                0b0000010000,
                0b0000001000
            };
            const Input prefix        = 0b0010000100;
            const Input continuation  = 0b0001000010;

            return std::make_tuple( world, prefix, continuation );
        }
    };

    template <typename Func>
    time_t time_it(Func&& f) 
    {
        const auto start = std::chrono::steady_clock::now();
        f();
        const auto stop = std::chrono::steady_clock::now();
        return (time_t)std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }
}   // utils
}   // AGI
}   // sprogar
