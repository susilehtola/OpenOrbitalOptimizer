

# File quad\_support.hpp

[**File List**](files.md) **>** [**openorbitaloptimizer**](dir_3072c93c56dfbbd2cb4eee0809487533.md) **>** [**quad\_support.hpp**](quad__support_8hpp.md)

[Go to the documentation of this file](quad__support_8hpp.md)


```C++
/*
 *                This Source Code Form is subject to the
 *                terms of the Mozilla Public License, v. 2.0.
 *                If a copy of the MPL was not distributed
 *                with this file, You can obtain one at
 *                http://mozilla.org/MPL/2.0/.
 *
 *           Copyright (c) 2025 Susi Lehtola
 */


#ifndef OPENORBITALOPTIMIZER_QUAD_SUPPORT_HPP
#define OPENORBITALOPTIMIZER_QUAD_SUPPORT_HPP

#include <limits>
#include <ostream>

#if !defined(__FLT128_MANT_DIG__) || __LDBL_MANT_DIG__ != __FLT128_MANT_DIG__
inline std::ostream & operator<<(std::ostream & os, _Float128 x) {
  return os << static_cast<long double>(x);
}
#endif

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
```


