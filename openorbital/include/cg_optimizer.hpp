/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once
#include <armadillo>

namespace OpenOrbitalOptimizer {
  namespace ConjugateGradients {

    /// Conjugate gradient optimization
    template<typename T>
    arma::Col<T> cg_optimize(const arma::Col<T> & x0, const std::function<std::pair<T,arma::Col<T>>(arma::Col<T>)> & fx, T f_tol = 1e-8, T df_tol = 1e-6, T x_tol = 100*std::numeric_limits<T>::epsilon(), size_t max_iter = 1000) {
      /// Current estimate for parameters
      auto x(x0);
      /// Evaluate initial point
      auto current_point = fx(x);

      /// Old search direction
      arma::Col<T> current_gradient, previous_gradient;
      arma::Col<T> current_direction, previous_direction;
      for(size_t iteration = 0; iteration < max_iter; iteration++) {
        /// Update the gradients
        previous_gradient = current_gradient;
        current_gradient = current_point.second;

        /// Form the search direction
        previous_direction = current_direction;
        current_direction = -current_gradient;
        if(iteration>0) {
          /// Use Polak-Ribière rule
          auto gamma = arma::dot(current_gradient,current_gradient-previous_gradient)/arma::dot(previous_gradient, previous_gradient);
          current_direction += gamma*previous_direction;
        }
        if(arma::dot(current_direction, current_gradient) > 0.0)
          /// Reset bad search direction
          current_direction = -current_gradient;

        //printf("Optimization iteration %i: gradient norm %e\n",iteration, arma::norm(current_gradient,2));
        /// Convergence check
        if(arma::norm(current_gradient,2)<df_tol)
          break;

        /// Helper for line search
        std::function<T(T)> evaluate_step = [x, fx, current_direction](T step) {
          return fx(x + step*current_direction).first;
        };

        /// Perform brute force line search
        auto step_lengths = arma::logspace<arma::Col<T>>(-6.0,6.0,13);
        arma::Col<T> function_values(step_lengths.n_elem);
        for(size_t i=0;i<step_lengths.n_elem;i++)
          function_values[i] = evaluate_step(step_lengths(i));

        /// Find the minimum value
        arma::uword minidx;
        function_values.min(minidx);
        if(minidx==0 || minidx==step_lengths.n_elem-1)
          throw std::logic_error("Issue in line search\n");

        /// Now we have bracketed the minimum, use golden section search.
        const double golden_ratio = (1+sqrt(5.0))/2.0;
        T right_bracket = step_lengths[minidx+1];
        T left_bracket = step_lengths[minidx-1];
        while(std::abs(right_bracket - left_bracket) > 0.5*x_tol*(left_bracket+right_bracket)) {
          //printf("left % e right % e length %e\n",left_bracket,right_bracket,std::abs(right_bracket - left_bracket));
          T test_left = right_bracket - (right_bracket-left_bracket) / golden_ratio;
          T test_right = left_bracket + (right_bracket-left_bracket) / golden_ratio;
          if(evaluate_step(test_left) < evaluate_step(test_right)) {
            right_bracket = test_right;
          } else {
            left_bracket = test_left;
          }
        }
        T optimal_step = 0.5*(right_bracket + left_bracket);

        /// Update the current point
        x += optimal_step * current_direction;
        current_point = fx(x);
      }

      return x;
    }
  }
}
