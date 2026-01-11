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
    size_t match_score(const std::bitset<BitsPerInput>& a, const std::bitset<BitsPerInput>& b)
    {
        return BitsPerInput - (a ^ b).count();
    }

    template <std::ranges::input_range R1, std::ranges::input_range R2>
    size_t match_score(const R1& r1, const R2& r2)
    {
        auto it1 = std::ranges::begin(r1);
        auto it2 = std::ranges::begin(r2);
        auto end1 = std::ranges::end(r1);

        size_t count = 0;
        for (; it1 != end1; ++it1, ++it2) {
            count += match_score(*it1, *it2);
        }
        return count;
    }

    inline size_t random_warm_up_time(time_t hi)
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
        enum structured_tag { structured = 0 };

        InputSequence() {}
        InputSequence(std::initializer_list<Input> il) : std::vector<Input>(il) {}

        //template<typename... Args>
        //Model(Args&&... args) : model(std::forward<Args>(args)...) {}

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

        // deterministically constructs a batch of structured inputs 
        InputSequence(structured_tag, time_t length, const size_t id)
        {
            base::reserve(length+1);

            const size_t predefined_patterns = 8;
            const size_t L = Input{}.size();
            const Input half_bits_set = (~Input()) >> (L / 2);              // 0b0000011111 @ L=10

            const int choice = id % predefined_patterns;
            const Input arp = choice != 7 ? Input{} : ~half_bits_set;   // absolute refractory-period

            while (base::size() < length) {
                const size_t shift = (base::size() / 2) % L;
                switch (choice)
                {
                case 0: base::push_back(Input{}); break;                    // 0 0 0 0 0...
                case 1: base::push_back(~Input{}); break;                   // ~0 0 ~0 0 ~0...
                case 2: base::push_back(Input{1}); break;                   // 1 0 1 0 1...
                case 3: base::push_back(Input{2}); break;                   // 2 0 2 0 2...
                case 4: base::push_back(Input{1ull << shift}); break;       // 1 0 2 0 4...
                case 5: base::push_back(Input{3ull << shift}); break;       // 3 0 6 0 12...
                case 6: base::push_back(half_bits_set); break;              // h 0 h 0 h...
                case 7: base::push_back(half_bits_set); break;              // h ~h h ~h h...
                }
                base::push_back(arp);
            }
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
        Model(random_tag, const time_t warm_up) : Model()
        {
            *this << InputSequence(InputSequence::random, warm_up);
        }
        
        //////////////
        Input operator ()(const Input& p) { return current_prediction = model(p); }
        Model& operator << (const Input& p) { current_prediction = model(p); return *this; }
        ////////////////
        Input operator()() const { return current_prediction; }

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
            std::cerr << "Error: Couldn't find a learnable sequence.\n";
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

        // Adapts the model to the given input sequence and returns the learning time in atomic steps required to achieve perfect prediction.
        time_t time_to_learn(const InputSequence& inputs, time_t timeframe)
        {
            for (time_t tau = 0; tau < timeframe; tau += inputs.size()) {
                if (process(inputs) == inputs)
                    return tau;
            }
            return timeframe;
        }

        // Adapts the model to the given input sequence and returns true if perfect prediction is achieved.
        bool learn(const InputSequence& inputs, time_t timeframe)
        {
            return time_to_learn(inputs, timeframe) < timeframe;
        }
        
        // Feeds the model its own predictions to generate a sequence of predictions.
        auto generate(size_t length)
        {
            return std::views::iota(std::size_t{ 0 }, length)
                | std::views::transform([&](std::size_t) {
                    Model& model = *this;
                    const Input prediction = model();
                    model << prediction;
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
                predictions.push_back(current_prediction);
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
 * that values in the second sequence (V2) tend to be greater than the corresponding
 * values in the first sequence (V1). It intentionally ignores effect size and
 * variability; it answers only whether the direction of the difference is stable
 * across pairs.
 *
 * The test is:
 *   - Paired (each observation in V1 corresponds to one in V2)
 *   - Non-parametric (no distributional assumptions)
 *   - Robust to outliers and heavy-tailed noise
 *   - Directional (specifically tests for V2 > V1)
 *
 * Given paired observations (V1_i, V2_i), the Wilcoxon signed-rank test evaluates the
 * null hypothesis that the median of the paired differences (V2_i - V1_i) is zero,
 * against the alternative hypothesis that the median of the paired difference is positive.
 *
 * Return value:
 *  (1) false if fewer than 10 non-zero paired differences are available;
 *  (2) true if there is statistically significant evidence that V2 > V1
 *      (z-score exceeds the one-sided threshold);
 *  (3) false otherwise.
 *
 * Parameters:
 *  V1, V2: index-paired observations (V1_i, V2_i)
 *
 *  one_sided_z_threshold
 *      Threshold applied to the z-score from the normal approximation.
 *      Common one-sided values:
 *      = 3.090  very conservative (0.1% significance) = AGITB setting
 *      = 2.326  strong evidence   (1% significance)
 *      = 1.645  standard choice   (5% significance)    
 **/
    bool consistently_greater_second_value(const std::vector<time_t>& V1, const std::vector<time_t>& V2,
        const double one_sided_z_threshold = 3.090)
    {
        assert(V1.size() == V2.size());

        struct SignedAbsDiff { size_t abs_diff; int sign; };
        std::vector<SignedAbsDiff> diffs; diffs.reserve(V1.size());

        for (size_t i = 0; i < V1.size(); ++i) {
            const time_t v1 = V1[i], v2 = V2[i];
            if (v1 == v2) continue;
            if (v2 > v1)    diffs.emplace_back(v2 - v1, +1);
            else            diffs.emplace_back(v1 - v2, -1);
        }

        const int n = (int)diffs.size();
        const int min_nonzero_pairs = 20;
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

        return z > one_sided_z_threshold;   // true => evidence that V2 tends to be greater than V1
    }

    template <typename T>
    inline T median(const std::vector<T>& times)
    {
        std::vector<T> sorted_times = times;
        std::sort(sorted_times.begin(), sorted_times.end());
        const size_t n = sorted_times.size();
        if (n % 2 == 1)
            return sorted_times[n / 2];
        else
            return (sorted_times[n / 2 - 1] + sorted_times[n / 2]) / 2;
    }
}   // utils
}   // AGI
}   // sprogar
