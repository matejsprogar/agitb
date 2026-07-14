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
    using time_t = std::time_t;

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

        template<typename... Args>
        InputSequence(Args&&... args) : base(std::forward<Args>(args)...) {}

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

        // constructs a simple, easily adaptable sequence of inputs with a specified length.
        InputSequence(trivial_tag, size_t length)
        {
            base::resize( length );
            base::back() = ~Input{};                // [{0...0}, {0...0}, ..., {0...0}, {1...1}]
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

        // Sequentially feeds each element of the range to the target.
        template <std::ranges::range Range>
            //requires std::same_as<std::ranges::range_value_t<Range>, Input>
        Model& operator << (Range&& range)
        {
            for (auto&& elt : range)
                *this << elt;
            return *this;
        }

        static InputSequence learnable_random_sequence(const size_t length)
        {
            for (time_t time = 0; time < SimulatedInfinity; time += length) {
                const InputSequence in = InputSequence(InputSequence::circular_random, length);
                Model M;
                if (M.learn(in))
                    return in;
            }

            const bool learned_at_least_one_sequence = false;
            ASSERT(learned_at_least_one_sequence);
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

    template <typename T>   
    auto median(const std::vector<T> vec)
    {
        const size_t n = vec.size();
        if (n == 0) return T{};

        std::vector<T> sorted = vec;
        std::sort(sorted.begin(), sorted.end());

        return (n % 2 == 1) ? sorted[n/2] : (sorted[n/2 - 1] + sorted[n/2]) / 2;
    }

    /*
    * Mann-Kendall trend test (one-sided, tie- and continuity-corrected).
    * Input: observations in TEMPORAL order. Returns the normal-approximation
    * z-score for a monotone INCREASING trend; z > threshold is significant growth.
    * Same conservative thresholds as elsewhere (3.090 = 0.1%).
    */
    bool mann_kendall_grow(const std::vector<time_t>& V, const double mann_kendall_significance_threshold = 3.090)
    {
        const int n = (int)V.size();
        if (n < 3) return false;

        long long S = 0;
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                S += (V[j] > V[i]) - (V[j] < V[i]);

        std::vector<time_t> sorted(V);
        std::sort(sorted.begin(), sorted.end());
        double tie_term = 0.0;
        for (int i = 0; i < n; ) {
            int j = i + 1;
            while (j < n && sorted[j] == sorted[i]) ++j;
            const auto t = j - i;
            if (t > 1) tie_term += (double)t * (t - 1) * (2 * t + 5);
            i = j;
        }
        const double variance = ((double)n * (n - 1) * (2 * n + 5) - tie_term) / 18.0;
        if (variance <= 0.0) return false;

        const double numerator = S > 0 ? (double)(S - 1) : (S < 0 ? (double)(S + 1) : 0.0);
        return numerator / std::sqrt(variance) > mann_kendall_significance_threshold;
    }

    /*
    * Sen's slope: the median of all pairwise slopes (x_j - x_i)/(j - i), i < j.
    * A robust estimate of trend magnitude in value-per-step; pairs with the
    * Mann-Kendall test to answer "how large is the trend", not just "is there one".
    */
    double sen_slope(const std::vector<time_t>& V)
    {
        const int n = (int)V.size();
        std::vector<double> slopes;
        slopes.reserve((size_t)n * (n - 1) / 2);
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                slopes.push_back(((double)V[j] - (double)V[i]) / (double)(j - i));
        if (slopes.empty()) return 0.0;

        return median(slopes);
    }

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
