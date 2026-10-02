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
        enum rhythm_tag { rhythm = 0 };

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

        // constructs a rhythm x y x y ... z with a specified length
        InputSequence(rhythm_tag, size_t length)
        {
            const Input x = utils::random<Input>(), 
                        y = utils::random<Input>(x), 
                        z = utils::random<Input>(x, y);
                        
            for (size_t i = 0; i + 1 < length; ++i)
                base::push_back(i % 2 ? y : x);
            base::push_back(z);
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

            size_t adaptations = 0;
            for (bool saturated = false; !saturated; adaptations += 1) {
                auto [time, _] = M.learn_anything(length);
                saturated = time == Infinity;
            }

            const size_t minimal_life = 50;
            for (size_t times = adaptations + 1; times < std::max(minimal_life, 2 * adaptations); ++times)     // successful or not
                M.learn(InputSequence(InputSequence::circular_random, length));

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
 
    // Generates a simple world governed by a single rule that rotates bits in an input by 1 position at each time step.
    // The problem is to generalise when a rotation increases to 2 positions.
    // The world is described using 
    template <typename Input>
    class sequence_generator
    {
        static constexpr size_t L = Input{}.size();
        static Input rotate(const Input& x, size_t k) { k %= L; return k == 0 ? x : (x << k) | (x >> (L - k)); }
        
        typedef Input(*operation)(const Input&, size_t);
        const operation fun = rotate;

        const int world_rotation = 1, test_rotation = 3;
        InputSequence<Input> world;

        public:
        sequence_generator()
        {
            const Input x0 = { 0b0101010001 };
            world.push_back(x0);
            do {
                world.push_back(rotate(world.back(), world_rotation));
            } 
            while (world.back() != x0);
            world.pop_back();
        }
        auto generate() const
        {
            const size_t prefix_size = 1, size_t continuation_size = 1;
            InputSequence<Input> prefix = { fun(world.back(), test_rotation) };
            while (prefix.size() < prefix_size)
                prefix.push_back(fun(prefix.back(), test_rotation));

            InputSequence<Input> continuation = { fun(prefix.back(), test_rotation) };
            while (continuation.size() < continuation_size)
                continuation.push_back( fun(continuation.back(), test_rotation) );

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
