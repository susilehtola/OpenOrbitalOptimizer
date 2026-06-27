/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once
#include "eigen_compat.hpp"
#include "types.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace OpenOrbitalOptimizer {
  namespace ConjugateGradients {

    /// Conjugate gradient optimization
    template<typename T>
    Vector<T> cg_optimize(const Vector<T> & x0,
                          const std::function<std::pair<T, Vector<T>>(Vector<T>)> & fx,
                          T f_tol = 1e-8,
                          T df_tol = 1e-6,
                          T x_tol = 100*std::numeric_limits<T>::epsilon(),
                          size_t max_iter = 1000) {
      (void)f_tol;
      /// Current estimate for parameters
      Vector<T> x = x0;
      /// Evaluate initial point
      auto current_point = fx(x);

      /// Old search direction
      Vector<T> current_gradient, previous_gradient;
      Vector<T> current_direction, previous_direction;
      for(size_t iteration = 0; iteration < max_iter; iteration++) {
        /// Update the gradients
        previous_gradient = current_gradient;
        current_gradient = current_point.second;

        /// Form the search direction
        previous_direction = current_direction;
        current_direction = -current_gradient;
        if(iteration>0) {
          /// Use Polak-Ribière rule, falling back to steepest descent if the previous gradient vanished
          T denom = previous_gradient.dot(previous_gradient);
          if(denom > std::numeric_limits<T>::min()) {
            T gamma = current_gradient.dot(current_gradient - previous_gradient) / denom;
            current_direction += gamma * previous_direction;
          }
        }
        if(current_direction.dot(current_gradient) > T{0})
          /// Reset bad search direction
          current_direction = -current_gradient;

        /// Convergence check
        if(current_gradient.norm() < df_tol)
          break;

        /// Helper for line search
        std::function<T(T)> evaluate_step = [x, fx, current_direction](T step) {
          return fx(Vector<T>(x + step * current_direction)).first;
        };

        /// Perform brute force line search
        Vector<T> step_lengths = logspace<T>(T{-6}, T{6}, 13);
        Vector<T> function_values(step_lengths.size());
        for(Index i = 0; i < step_lengths.size(); i++)
          function_values[i] = evaluate_step(step_lengths[i]);

        /// Find the minimum value
        Index minidx;
        function_values.minCoeff(&minidx);
        if(minidx==0 || minidx == step_lengths.size() - 1)
          throw std::logic_error("Issue in line search\n");

        /// Now we have bracketed the minimum, use golden section search.
        const T golden_ratio = (T{1} + std::sqrt(T{5})) / T{2};
        T right_bracket = step_lengths[minidx + 1];
        T left_bracket = step_lengths[minidx - 1];
        while(std::abs(right_bracket - left_bracket) > T{0.5} * x_tol * (left_bracket + right_bracket)) {
          T test_left = right_bracket - (right_bracket - left_bracket) / golden_ratio;
          T test_right = left_bracket + (right_bracket - left_bracket) / golden_ratio;
          if(evaluate_step(test_left) < evaluate_step(test_right)) {
            right_bracket = test_right;
          } else {
            left_bracket = test_left;
          }
        }
        T optimal_step = T{0.5} * (right_bracket + left_bracket);

        /// Update the current point
        x += optimal_step * current_direction;
        current_point = fx(x);
      }

      return x;
    }
  }
}
