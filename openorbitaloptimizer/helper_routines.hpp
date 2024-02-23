/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

namespace OpenOrbitalOptimizer {
  namespace HelperRoutines {
    /// Evaluates cubic polynomial f(x) = ax^3 + bx^2 + cx + d at given x
    template<typename T> std::tuple<T,T,T,T> evaluate_cubic_polynomial(T x, T a, T b, T c, T d) {
      return ((a*x+b)*x+c)*x+d;
    }

    /// Fits cubic polynomial f(x) = ax^3 + bx^2 + cx + d with given f(0), f'(0), L, f(L), and f'(L).
    template<typename T> std::tuple<T,T,T,T> fit_cubic_polynomial(T f0, T df0, T L, T fL, T dfL) {
      double a = (L*df0 + L*dfL + 2*f0 - 2*fL)/(L*L*L);
      double b = -(2*L*df0 + L * dfL + 3*f0 - 3 * fL)/(L*L);
      double c = fp0;
      double d = f0;
      return std::make_tuple(a,b,c,d);
    }

    /** Finds the extrema of a given cubic polynomial f(x) = ax^3 + bx^2 + cx + d.
        Note that one can use: auto zeros = std::apply(cubic_polynomial_zeros, fit_cubic_polynomial(f0, df0, L, fL, dfL))
    */
    template<typename T> std::pair<T,T> cubic_polynomial_zeros(T a, T b, T c, T d) {
      // Derivative is E' = 3ax^2 + 2bx + c
      a *= 3;
      b *= 2;

      // Discriminant is
      Tbase D = b*b - 4*a*c;
      if(D<=0.0)
        throw std::logic_error("Polynomial has no extrema!\n");

      // Find zeros of E'
      Tbase x1 = (-b - std::sqrt(D))/(2*a);
      Tbase x2 = (-b + std::sqrt(D))/(2*a);
      Tbase minx = std::min(x1,x2);
      Tbase maxx = std::max(x1,x2);
      return std::make_pair(minx, maxx);
    }
  }
}
