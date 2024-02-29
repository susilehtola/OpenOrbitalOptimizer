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

    /// Conjugate gradient optimization with given line search
    template<typename T>
    arma::Col<T> cg_optimize_limited_line(const arma::Col<T> & x0, const std::function<std::pair<T,arma::Col<T>>(arma::Col<T>)> & fx, T max_step, T f_tol = 1e-8, T df_tol = 1e-6, T x_tol = 100*std::numeric_limits<T>::epsilon(), size_t max_iter = 1000) {
      /// Current estimate for parameters
      auto x(x0);
      auto current_point = fx(x);
      auto current_energy = current_point.first;

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

        printf("Optimization iteration %i: gradient norm %e\n",iteration, arma::norm(current_gradient,2));
        /// Convergence check
        if(arma::norm(current_gradient,2)<df_tol) {
          printf("gradient norm = %e < df_tol = %e\n",arma::norm(current_gradient,2),df_tol);
          break;
        }

        // Helper for line search
        std::function<std::pair<T,T>(T)> evaluate_step = [x, fx, current_gradient, current_direction](T step) {
          auto eval = fx(x + step*current_direction);
          auto fval = eval.first;
          auto slope = arma::dot(eval.second, current_direction);
          return std::make_pair(fval, slope);
        };

        // Perform line search for zero gradient
        std::vector<std::tuple<T,T,T>> steps;
        steps.push_back(std::make_tuple((T) 0.0, current_point.first, arma::dot(current_gradient, current_direction)));
        if(std::get<2>(steps[0])>0.0)
          throw std::logic_error("Positive projection on gradient direction at origin!\n");

        // Find a flip in the gradient
        T step_length = max_step;
        while(true) {
          auto eval = evaluate_step(step_length);
          steps.push_back(std::make_tuple(step_length, eval.first, eval.second));
          printf("Evaluated step %e E = %.10f dEdstep=%e\n",step_length,eval.first,eval.second);
          // Stop when the sign of the gradient flips
          if(std::get<2>(steps[steps.size()-1]) > 0.0)
            break;
          step_length *= 10.0;
        }

        if(steps.size()==2) {
          // Initial step was long enough; retrace
          while(true) {
            step_length /= 10.0;
            auto eval = evaluate_step(step_length);
            steps.push_back(std::make_tuple(step_length, eval.first, eval.second));
            printf("Tracing back step %e E = %.10f dEdstep=%e\n",step_length,eval.first,eval.second);
            // Stop when the sign of the gradient flips to negative
            if(std::get<2>(steps[steps.size()-1]) < 0.0)
              break;
          }
        }
        // Sort array in step length
        std::sort(steps.begin(), steps.end(), [](const std::tuple<T,T,T> & x,const std::tuple<T,T,T> & y) {return std::get<0>(x) < std::get<0>(y);});

        // Find the entry with the first positive sign
        size_t flipind=0;
        for(flipind=1;flipind<steps.size();flipind++)
          if(std::get<2>(steps[flipind])>0.0)
            break;

        // Now we can do a binary search
        T left_step = std::get<0>(steps[flipind-1]);
        T right_step = std::get<0>(steps[flipind]);
        T optimal_step;
        do {
          optimal_step = 0.5*(right_step + left_step);
          auto eval = evaluate_step(optimal_step);
          if(std::abs(eval.second)<10*std::numeric_limits<T>::epsilon())
            // We are essentially at zero gradient
            break;
          printf("Evaluated step %e E = %.10f dEdstep=%e\n",optimal_step,eval.first,eval.second);
          if(eval.second>0.0) {
            right_step = optimal_step;
          } else if(eval.second<0.0) {
            left_step = optimal_step;
          } else // exact zero
            break;
        } while(std::abs(right_step - left_step) > 0.5*x_tol*(right_step+left_step));

        /// Update the current point
        x += optimal_step * current_direction;
        current_point = fx(x);

        T dE = current_energy - current_point.first;
        printf("Energy decrease in iteration %i %e\n",iteration,dE);
        if(current_energy - current_point.first < f_tol) {
          printf("Energy decrease in iteration %i %e < f_tol = %e, stopping\n",iteration,dE, f_tol);
          break;
        }
      }

      return x;
    }
  }
}
