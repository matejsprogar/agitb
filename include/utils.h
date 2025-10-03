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

    template <typename M, typename T>
    concept InputPredictor = std::regular<M>
        && requires(M c, const M cc, const T t)
    {
        { c << t } -> std::convertible_to<M&>;
        { cc.prediction() } -> std::convertible_to<T>;
    };
    template <typename T>
    concept Indexable = std::regular<T> && requires(T t, const T c)
    {
        { t[size_t{}] } -> std::convertible_to<typename T::reference>;
        { c[size_t{}] } -> std::convertible_to<bool>;
        { c.size() } -> std::convertible_to<size_t>;
    };

    template <size_t BitsPerInput>
    size_t count_matching_bits(const std::bitset<BitsPerInput>& a, const std::bitset<BitsPerInput>& b)
    {
        return BitsPerInput - (a ^ b).count();
    }

    template <Indexable T>
    size_t count_matching_bits(const T& a, const T& b)
    {
        return std::ranges::count_if(std::views::iota(0ul, a.size()), [&](size_t i) { return a[i] == b[i]; });
    }

    template <std::ranges::input_range R1, std::ranges::input_range R2>
    size_t count_matching_bits(const R1& r1, const R2& r2)
    {
        auto it1 = std::ranges::begin(r1);
        auto it2 = std::ranges::begin(r2);
        auto end1 = std::ranges::end(r1);

        size_t count = 0;
        for (; it1 != end1; ++it1, ++it2) {
            count += count_matching_bits(*it1, *it2);
        }
        return count;
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
        using base = std::vector<Input>;
    public:
        enum random_tag { random = 0 };
        enum circular_random_tag { circular_random = 0 };
        enum trivial_problem_tag { trivial_problem = 0 };

        InputSequence() {}
        InputSequence(std::initializer_list<Input> il) : std::vector<Input>(il) {}

        // constructs a random sequence of inputs with a specified length.
        InputSequence(random_tag, time_t length)
        {
            if (0 == length)
                return;

            base::reserve(length);

            base::push_back(utils::random<Input>());
            while (base::size() < length)
                base::push_back(utils::random<Input>(base::back()));
        }
        // constructs a random sequence of inputs with a specified length, exhibiting a circular property 
        // where the first input incorporates refractory periods for the last input in the sequence.
        InputSequence(circular_random_tag, time_t length) : InputSequence(random, length) {
            base::pop_back();
            base::push_back(utils::random<Input>(base::back(), base::front()));
        }

        // constructs a simple, easily adaptable sequence of inputs with a specified length.
        InputSequence(trivial_problem_tag, time_t length)
        {
            base::resize( length );
            base::back() = ~Input{};                // [{0...0}, {0...0}, ..., {0...0}, {1...1}]
        }
    };


    template <typename Model>
    InputSequence<typename Model::Input> learnable_random_sequence(const size_t length, time_t timeframe)
    {
        using InputSequence = InputSequence<typename Model::Input>;

        for (time_t time = 0; time < timeframe; time += length) {
            const InputSequence in = InputSequence(InputSequence::circular_random, length);
            Model M;
            if (M.adapt(in, timeframe))
                return in;
        }
        std::cerr << red("Error:") << " Could not find a learnable sequence.\n";
        exit(-1);
    }


    template <typename ModelUnderTest, typename InputType>
    requires InputPredictor<ModelUnderTest, InputType> and Indexable<InputType>
    class Model
    {
        ModelUnderTest model;
    public:
        using Input = InputType;
        using InputSequence = utils::InputSequence<Input>;

        enum random_tag { random = 0 };

        Model() = default;
        Model(const Model& src) = default;
        Model(Model&& src) = default;
        Model& operator=(const Model& src) = default;
        bool operator==(const Model& rhs) const = default;// { return model == rhs.model; }
    
        template<typename... Args>
        Model(Args&&... args) : model(std::forward<Args>(args)...) {}

        // Constructs a randomly initialized model object by feedng it with random inputs.
        Model(random_tag, const time_t random_initialization_strength) : Model()
        {
            *this << InputSequence(InputSequence::random, random_initialization_strength);
        }

        // Iteratively feeds each model its own predictions and returns true if predictions match over a specified timeframe.
        static bool identical_behaviour(Model& A, Model& B, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; ++time) {
                const auto expectation = A.prediction();
                if (expectation != B.prediction())
                    return false;
                A << expectation;
                B << expectation;
            }
            return A.prediction() == B.prediction();
        }

        // Adapts the model to the given input sequence and returns the time required to achieve perfect prediction.
        time_t time_to_repeat(const InputSequence& inputs, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; time += inputs.size()) {
                if (process(inputs) == inputs)
                    return time;
            }
            return timeframe;
        }

        // Adapts the model to the given input sequence and returns true if perfect prediction is achieved.
        bool adapt(const InputSequence& inputs, time_t timeframe)
        {
            return time_to_repeat(inputs, timeframe) < timeframe;
        }

        Model& operator << (const Input& p) { model << p; return *this; }
        Input prediction() const { return model.prediction(); }

        // Sequentially feeds each element of the range to the target.
        template <std::ranges::range Range>
            requires std::same_as<std::ranges::range_value_t<Range>, Input>
        Model& operator << (Range&& range)
        {
            for (auto&& elt : range)
                model << elt;
            return *this;
        }
        
        //auto generate(size_t length)
        //{
        //    auto seq = std::views::iota(size_t{0}, length)
        //     | std::views::transform([&](size_t) {
        //           auto value = prediction();
        //           model << value;
        //           return value;
        //       });
        //    return seq;
        //}
        InputSequence generate(size_t length)
        {
            InputSequence seq;
            seq.reserve(length);
            while (seq.size() < length) {
                seq.push_back(prediction());
                model << seq.back();
            }
            return seq;
        }

    private:
        // Modifies the model by processing the given inputs and returns its corresponding predictions.
        InputSequence process(const InputSequence& inputs)
        {
            InputSequence predictions{};
            predictions.reserve(inputs.size());

            for (const Input& in : inputs) {
                predictions.push_back(model.prediction());
                model << in;
            }
            return predictions;
        }
    };

    }
}
}
