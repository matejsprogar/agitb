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

namespace sprogar {

inline std::string red(const char* msg) { return std::format("\033[91m{}\033[0m", msg); }
inline std::string green(const char* msg) { return std::format("\033[92m{}\033[0m", msg); }
inline std::string yellow(const char* msg) { return std::format("\033[93m{}\033[0m", msg); }

namespace AGI {
inline namespace utils {
    using time_t = size_t;

    static const auto random_seed = std::random_device{}();

    template <typename M, typename T>
    concept InputPredictor = std::regular<M>
        && requires(M c, const T& t)
    {
        { c(t) } -> std::convertible_to<T>;
    };

    template <size_t BitsPerInput>
    size_t count_matching_bits(const std::bitset<BitsPerInput>& a, const std::bitset<BitsPerInput>& b)
    {
        return BitsPerInput - (a ^ b).count();
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

    inline size_t random(size_t hi)
    {
        static std::mt19937 rng{ random_seed+1 };
        static std::uniform_int_distribution<size_t> dist(0, hi);
        return dist(rng);
    }
    
    // Returns an input with spikes at random positions, except where explicitly required to have none.
    template<typename Input, typename... Inputs>
    requires (std::same_as<Input, Inputs> && ...)
    Input random(const Inputs&... turn_off)
    {
        static std::mt19937 rng{ random_seed };
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
        enum trivial_tag { trivial = 0 };

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
        InputSequence(trivial_tag, time_t length)
        {
            base::resize( length );
            base::back() = ~Input{};                // [{0...0}, {0...0}, ..., {0...0}, {1...1}]
        }
    };

    template <typename ModelUnderTest, typename InputType>
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
        Model(random_tag, const time_t random_initialization_strength) : Model()
        {
            *this << InputSequence(InputSequence::random, random_initialization_strength);
        }
        
        //////////////
        Input operator ()(const Input& p) { return cached_prediction = model(p); }
        Model& operator << (const Input& p) { cached_prediction = model(p); return *this; }
        ////////////////
        Input operator()() const { return cached_prediction; }

        // Sequentially feeds each element of the range to the target.
        template <std::ranges::range Range>
            requires std::same_as<std::ranges::range_value_t<Range>, Input>
        Model& operator << (Range&& range)
        {
            for (auto&& elt : range)
                *this << elt;
            return *this;
        }

        static InputSequence learnable_random_sequence(const size_t length, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; time += length) {
                const InputSequence in = InputSequence(InputSequence::circular_random, length);
                Model M;
                if (M.learn(in, timeframe))
                    return in;
            }
            std::cerr << red("Error:") << " Could not find a learnable sequence.\n";
            exit(-1);
        }

        // Iteratively feeds each model its own predictions and returns true if predictions match over a specified timeframe.
        static bool identical_behaviour(Model& A, Model& B, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; ++time) {
                const auto prediction = A();
                if (prediction != B())
                    return false;
                A << prediction;
                B << prediction;
            }
            return A() == B();
        }

        // Adapts the model to the given input sequence and returns the time required to achieve perfect prediction.
        time_t time_to_learn(const InputSequence& inputs, time_t timeframe)
        {
            for (time_t time = 0; time < timeframe; time += inputs.size()) {
                if (process(inputs) == inputs)
                    return time;
            }
            return timeframe;
        }

        // Adapts the model to the given input sequence and returns true if perfect prediction is achieved.
        bool learn(const InputSequence& inputs, time_t timeframe)
        {
            return time_to_learn(inputs, timeframe) < timeframe;
        }
        
        // Feeds the model its own predictions to generate a sequence of predictions.
        InputSequence generate(size_t length)
        {
            InputSequence seq; seq.reserve(length);
            while (seq.size() < length) {
                seq.push_back(cached_prediction);
                cached_prediction = seq.back();
            }
            return seq;
        }

    private:
        ModelUnderTest model;
        Input cached_prediction;
        
        // Modifies the model by processing the given inputs and returns its corresponding predictions.
        InputSequence process(const InputSequence& inputs)
        {
            InputSequence predictions; predictions.reserve(inputs.size());

            for (const Input& in : inputs) {
                predictions.push_back(cached_prediction);
                *this << in;
            }
            return predictions;
        }
    };

 /**
 * Tests whether the second of two paired sequences of elapsed times is consistently 
 * worse (i.e., larger) than the first, using a one-sided Wilcoxon signed-rank test.
 *
 * This function implements a one-sided Wilcoxon signed-rank test on paired data and
 * returns a boolean indicating whether there is statistically significant evidence
 * that values in the second sequence (B) tend to be greater than the corresponding
 * values in the first sequence (A). It intentionally ignores effect size and
 * variability; it answers only whether the direction of the difference is stable
 * across pairs.
 *
 * The test is:
 *   - Paired (each observation in A corresponds to one in B)
 *   - Non-parametric (no distributional assumptions)
 *   - Robust to outliers and heavy-tailed noise
 *   - Directional (specifically tests for B > A)
 *
 * Given paired observations (A_i, B_i), the Wilcoxon signed-rank test evaluates the
 * null hypothesis that the median of the paired differences (B_i - A_i) is zero,
 * against the alternative hypothesis that the median of the paired difference is positive.
 *
 * Return value:
 *  (1) false if fewer than 10 non-zero paired differences are available;
 *  (2) true if there is statistically significant evidence that B > A
 *      (z-score exceeds the one-sided threshold);
 *  (3) false otherwise.
 *
 * Parameters:
 *  AB
 *      A vector of paired observations (A_i, B_i).
 *
 *  one_sided_z_threshold
 *      Threshold applied to the z-score from the normal approximation.
 *      Common one-sided values:
 *      = 3.090  very conservative (0.1% significance) = AGITB setting
 *      = 2.326  strong evidence   (1% significance)
 *      = 1.645  standard choice   (5% significance)    
 **/
    bool consistently_greater_second_value(const std::vector<std::pair<size_t, size_t>>& AB,
        const double one_sided_z_threshold = 3.090)
    {
        struct SignedAbsDiff { size_t abs_diff; int sign; };
        std::vector<SignedAbsDiff> diffs; diffs.reserve(AB.size());

        for (auto [a, b] : AB) {
            if (a == b) continue;
            if (b > a)  diffs.emplace_back(b - a, +1);
            else        diffs.emplace_back(a - b, -1);
        }

        const int n = (int)diffs.size();
        const int min_nonzero_pairs = 10;
        if (n < min_nonzero_pairs) return false;

        std::sort(diffs.begin(), diffs.end(),
            [](const auto& x, const auto& y) { return x.abs_diff < y.abs_diff; });

        double Wplus = 0.0, tieCorr = 0.0;
        for (int i = 0; i < n; ) {
            int j = i + 1;
            while (j < n && diffs[j].abs_diff == diffs[i].abs_diff)
                ++j;
            int t = j - i;
            double avgRank = 0.5 * ((i + 1) + j);
            for (int k = i; k < j; ++k)
                if (diffs[k].sign > 0)
                    Wplus += avgRank;
            if (t > 1)
                tieCorr += (double)t * ((double)t * t - 1); // t^3-t
            i = j;
        }

        const double mu = n * (n + 1.0) / 4.0;
        const double var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - tieCorr / 48.0;
        if (var <= 0.0) return false;

        const double cc = (Wplus > mu) ? 0.5 : 0.0;
        double z = (Wplus - mu - cc) / std::sqrt(var);

        return z > one_sided_z_threshold;   // true => evidence that B tends to be greater than A
    }
}   // utils
}   // AGI
}   // sprogar
