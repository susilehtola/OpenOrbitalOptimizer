/*
 *                This Source Code Form is subject to the
 *                terms of the Mozilla Public License, v. 2.0.
 *                If a copy of the MPL was not distributed
 *                with this file, You can obtain one at
 *                http://mozilla.org/MPL/2.0/.
 *
 *           Copyright (c) 2025 Susi Lehtola
 */

/// Opt-in glue making `_Float128` (the ISO C23 quad-precision type,
/// available in GCC 13+/Clang 18+ with libstdc++ >= 14) usable as a
/// Tbase for SCFSolver. libstdc++ already specialises
/// std::numeric_limits and provides std::abs/sqrt/exp/log/pow/isnan/...
/// for _Float128, but only when compiled as C++23 or newer (the
/// overloads are gated by __cpp_lib_extended_float). The missing pieces
/// are an Eigen::NumTraits specialisation and an ostream insertion
/// operator that resolves the otherwise-ambiguous lookup chain.
///
/// Requirements for consumers of this header:
///   * compile with -std=c++23 (or newer)
///   * link against libquadmath (system library for some helpers; Eigen
///     itself only needs the language type)
/// Include this header BEFORE the first Eigen header — the
/// operator<<(ostream, _Float128) declaration needs to be visible during
/// Eigen's two-phase template lookup, and that means it must precede
/// Eigen's IO.h definition.

#ifndef OPENORBITALOPTIMIZER_QUAD_SUPPORT_HPP
#define OPENORBITALOPTIMIZER_QUAD_SUPPORT_HPP

#include <limits>
#include <ostream>

/// libstdc++ does not provide a stream output operator for _Float128.
/// Without one, any ostream insertion is ambiguous because the implicit
/// conversions to float / double / long double all rank equally.
/// Provide an explicit overload routed through long double — precision
/// past ~19 digits is lost in print, but this is a development-time
/// diagnostic path, not a serialisation format.
inline std::ostream & operator<<(std::ostream & os, _Float128 x) {
  return os << static_cast<long double>(x);
}

#include <Eigen/Core>

namespace Eigen {
  template <>
  struct NumTraits<_Float128> : GenericNumTraits<_Float128> {
    typedef _Float128 Real;
    typedef _Float128 NonInteger;
    typedef _Float128 Nested;
    typedef _Float128 Literal;
    enum {
      IsComplex = 0,
      IsInteger = 0,
      IsSigned = 1,
      RequireInitialization = 0,
      ReadCost = 1,
      AddCost = 6,
      MulCost = 8
    };
    static inline Real epsilon()         { return std::numeric_limits<_Float128>::epsilon(); }
    static inline Real dummy_precision() { return Real{1e-30f128}; }
    static inline Real highest()         { return (std::numeric_limits<_Float128>::max)(); }
    static inline Real lowest()          { return std::numeric_limits<_Float128>::lowest(); }
    static inline int digits10()         { return std::numeric_limits<_Float128>::digits10; }
  };
}

#endif
