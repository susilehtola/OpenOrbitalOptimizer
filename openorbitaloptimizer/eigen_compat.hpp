/*
 *                This Source Code Form is subject to the
 *                terms of the Mozilla Public License, v. 2.0.
 *                If a copy of the MPL was not distributed
 *                with this file, You can obtain one at
 *                http://mozilla.org/MPL/2.0/.
 *
 *           Copyright (c) 2025 Susi Lehtola
 */
#ifndef OPENORBITALOPTIMIZER_EIGEN_COMPAT_HPP
#define OPENORBITALOPTIMIZER_EIGEN_COMPAT_HPP

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace OpenOrbitalOptimizer {

  /// Real component type of a (possibly complex) scalar.
  template <class T>
  using RealOf = typename Eigen::NumTraits<T>::Real;

  /// Stack a vector of column vectors into one long column vector. Replaces
  /// arma::join_cols on Cols.
  template <class T>
  Vector<T> join_columns(const std::vector<Vector<T>> & parts) {
    Index total = 0;
    for (const auto & p : parts) total += p.size();
    Vector<T> out(total);
    Index off = 0;
    for (const auto & p : parts) {
      out.segment(off, p.size()) = p;
      off += p.size();
    }
    return out;
  }

  /// Vectorise a real-valued matrix to a column vector (column-major), with no
  /// real/imag splitting.
  template <class T>
  std::enable_if_t<!Eigen::NumTraits<T>::IsComplex, Vector<T>>
  vectorise_real_imag(const Matrix<T> & M) {
    return Eigen::Map<const Vector<T>>(M.data(), M.size());
  }

  /// Vectorise a complex-valued matrix into a real column vector by stacking
  /// the real part on top of the imaginary part. Mirrors the layout the SCF
  /// solver relies on for real-valued optimisation over complex orbital
  /// rotations.
  template <class T>
  std::enable_if_t<Eigen::NumTraits<T>::IsComplex, Vector<RealOf<T>>>
  vectorise_real_imag(const Matrix<T> & M) {
    using R = RealOf<T>;
    Vector<R> out(2 * M.size());
    // M is column-major, so the data pointer streams down columns; this
    // matches arma's storage layout.
    auto realview = Eigen::Map<const Matrix<R>>(reinterpret_cast<const R*>(M.data()),
                                                2, M.size());
    // realview row 0 is the real parts in memory order, row 1 the imag parts.
    out.head(M.size()) = realview.row(0).transpose();
    out.tail(M.size()) = realview.row(1).transpose();
    return out;
  }

  /// Inverse of vectorise_real_imag for the real case.
  template <class T>
  std::enable_if_t<!Eigen::NumTraits<T>::IsComplex, Matrix<T>>
  unvectorise_real_imag(const Vector<T> & v, Index rows, Index cols) {
    return Eigen::Map<const Matrix<T>>(v.data(), rows, cols);
  }

  /// Inverse of vectorise_real_imag for the complex case.
  template <class T>
  Matrix<std::complex<T>>
  unvectorise_real_imag_complex(const Vector<T> & v, Index rows, Index cols) {
    Matrix<std::complex<T>> out(rows, cols);
    const Index n = rows * cols;
    auto interleaved = Eigen::Map<Matrix<T>>(reinterpret_cast<T*>(out.data()), 2, n);
    interleaved.row(0) = v.head(n).transpose();
    interleaved.row(1) = v.tail(n).transpose();
    return out;
  }

  /// Find every index i where pred(v[i]) is true. Stand-in for
  /// arma::find(some_predicate).
  template <class Vec, class Pred>
  IndexVector find_indices_where(const Vec & v, Pred pred) {
    std::vector<Index> hits;
    hits.reserve(v.size());
    for (Index i = 0; i < v.size(); ++i)
      if (pred(v[i]))
        hits.push_back(i);
    IndexVector out(hits.size());
    for (size_t k = 0; k < hits.size(); ++k)
      out[k] = hits[k];
    return out;
  }

  /// Return the indices that sort v in ascending order (stable). Stand-in for
  /// arma::sort_index.
  template <class T>
  IndexVector sort_index_ascending(const Vector<T> & v) {
    IndexVector idx(v.size());
    std::iota(idx.data(), idx.data() + idx.size(), Index{0});
    std::stable_sort(idx.data(), idx.data() + idx.size(),
                     [&](Index a, Index b) { return v[a] < v[b]; });
    return idx;
  }

  /// arma::linspace(a, b, n) replacement returning n equally-spaced points
  /// [a, b].
  template <class T>
  Vector<T> linspace(T a, T b, Index n) {
    Vector<T> out(n);
    if (n == 1) {
      out[0] = a;
      return out;
    }
    const T step = (b - a) / static_cast<T>(n - 1);
    for (Index i = 0; i < n; ++i)
      out[i] = a + step * static_cast<T>(i);
    return out;
  }

  /// Logarithmically-spaced points 10^a ... 10^b (n points). Mirrors
  /// arma::logspace.
  template <class T>
  Vector<T> logspace(T a, T b, Index n) {
    Vector<T> exponents = linspace(a, b, n);
    for (Index i = 0; i < n; ++i)
      exponents[i] = std::pow(static_cast<T>(10), exponents[i]);
    return exponents;
  }

  /// Index of the largest absolute value in v (matches arma's index_max for
  /// real and arma's index_max(abs(v)) for complex).
  template <class Vec>
  Index index_max_abs(const Vec & v) {
    using S = typename Vec::Scalar;
    using R = typename Eigen::NumTraits<S>::Real;
    Index best = 0;
    R bestVal = std::abs(v[0]);
    for (Index i = 1; i < v.size(); ++i) {
      R x = std::abs(v[i]);
      if (x > bestVal) { bestVal = x; best = i; }
    }
    return best;
  }

  /// Dump a dense matrix as ASCII (one row per line, space-separated).
  /// Stand-in for arma::Mat::save(name, arma::raw_ascii).
  template <class Mat>
  void save_raw_ascii(const Mat & M, const std::string & filename) {
    std::ofstream os(filename);
    if (!os) throw std::runtime_error("save_raw_ascii: cannot open " + filename);
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (Index r = 0; r < M.rows(); ++r) {
      for (Index c = 0; c < M.cols(); ++c) {
        if (c) os << ' ';
        os << M(r, c);
      }
      os << '\n';
    }
  }

  /// True iff M contains a NaN. Eigen has allFinite() but not has_nan().
  template <class Mat>
  bool has_nan(const Mat & M) {
    return M.array().isNaN().any();
  }

  /// True iff M contains an infinity.
  template <class Mat>
  bool has_inf(const Mat & M) {
    return (M.array().isInf()).any();
  }

  /// arma::dot for complex vectors is non-conjugating; Eigen's a.dot(b) is
  /// conjugating. Provide a non-conjugating dot for parity.
  template <class V1, class V2>
  auto dot_nonconj(const V1 & a, const V2 & b) {
    return (a.array() * b.array()).sum();
  }

  /// exp(K) for an anti-Hermitian K = -K^\dagger. Computed via the Hermitian
  /// eigendecomposition of iK: iK = U diag(w) U^\dagger with real w, so
  /// exp(K) = exp(-i * iK) = U diag(exp(-i w)) U^\dagger. Returns a matrix of
  /// the same scalar type as K (so complex-typed K produces a complex result;
  /// for a real K the caller is expected to have used the block trick or wrap
  /// in complex first).
  template <class T>
  Matrix<T> expm_antihermitian(const Matrix<T> & K) {
    using R = RealOf<T>;
    // Build iK. For complex T this is a rotation; for real T we need to widen.
    if constexpr (Eigen::NumTraits<T>::IsComplex) {
      Matrix<T> iK = T(R{0}, R{1}) * K; // multiply by i
      Eigen::SelfAdjointEigenSolver<Matrix<T>> es(iK);
      const auto & U = es.eigenvectors();
      const auto & w = es.eigenvalues();
      Vector<T> phase(w.size());
      for (Index i = 0; i < w.size(); ++i)
        phase[i] = std::exp(T(R{0}, -w[i]));
      return U * phase.asDiagonal() * U.adjoint();
    } else {
      // Real anti-symmetric K: promote to complex so that eigenvalues are real.
      Matrix<std::complex<R>> iK(K.rows(), K.cols());
      for (Index c = 0; c < K.cols(); ++c)
        for (Index r = 0; r < K.rows(); ++r)
          iK(r, c) = std::complex<R>(R{0}, R{1}) * K(r, c);
      Eigen::SelfAdjointEigenSolver<Matrix<std::complex<R>>> es(iK);
      const auto & U = es.eigenvectors();
      const auto & w = es.eigenvalues();
      Vector<std::complex<R>> phase(w.size());
      for (Index i = 0; i < w.size(); ++i)
        phase[i] = std::exp(std::complex<R>(R{0}, -w[i]));
      Matrix<std::complex<R>> C = U * phase.asDiagonal() * U.adjoint();
      // The result is real to within round-off for real K.
      Matrix<T> out(K.rows(), K.cols());
      for (Index c = 0; c < K.cols(); ++c)
        for (Index r = 0; r < K.rows(); ++r)
          out(r, c) = static_cast<T>(C(r, c).real());
      return out;
    }
  }

  /// Return a random permutation of {0, 1, ..., n-1}. Mirrors arma::randperm.
  /// Uses a Mersenne Twister seeded once per program.
  inline IndexVector randperm(Index n) {
    IndexVector out(n);
    std::iota(out.data(), out.data() + n, Index{0});
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::shuffle(out.data(), out.data() + n, rng);
    return out;
  }

} // namespace OpenOrbitalOptimizer

#endif
