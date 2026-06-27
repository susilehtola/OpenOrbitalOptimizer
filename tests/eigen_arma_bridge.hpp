/*
 *                This Source Code Form is subject to the
 *                terms of the Mozilla Public License, v. 2.0.
 *                If a copy of the MPL was not distributed
 *                with this file, You can obtain one at
 *                http://mozilla.org/MPL/2.0/.
 *
 *           Copyright (c) 2025 Susi Lehtola
 */
#ifndef EIGEN_ARMA_BRIDGE_HPP
#define EIGEN_ARMA_BRIDGE_HPP

#include <armadillo>
#include <openorbitaloptimizer/types.hpp>

#include <utility>
#include <vector>

/// Adapter helpers for atomtest, which still drives the SCFSolver using
/// Armadillo-shaped data (atomicsolver.hpp uses arma::mat throughout).
/// Both libraries store dense matrices in column-major order, so the
/// conversions below are straight memcpys.
namespace eaa {

  template <class T>
  OpenOrbitalOptimizer::Matrix<T> to_eigen(const arma::Mat<T> & A) {
    return Eigen::Map<const OpenOrbitalOptimizer::Matrix<T>>(
        A.memptr(), A.n_rows, A.n_cols);
  }

  template <class T>
  OpenOrbitalOptimizer::Vector<T> to_eigen(const arma::Col<T> & v) {
    return Eigen::Map<const OpenOrbitalOptimizer::Vector<T>>(
        v.memptr(), v.n_elem);
  }

  inline OpenOrbitalOptimizer::IndexVector to_eigen(const arma::uvec & v) {
    OpenOrbitalOptimizer::IndexVector out(v.n_elem);
    for (arma::uword i = 0; i < v.n_elem; ++i)
      out[static_cast<OpenOrbitalOptimizer::Index>(i)] =
          static_cast<OpenOrbitalOptimizer::Index>(v[i]);
    return out;
  }

  template <class T>
  arma::Mat<T> to_arma(const OpenOrbitalOptimizer::Matrix<T> & E) {
    return arma::Mat<T>(E.data(), E.rows(), E.cols());
  }

  template <class T>
  arma::Col<T> to_arma(const OpenOrbitalOptimizer::Vector<T> & v) {
    return arma::Col<T>(v.data(), v.size());
  }

  template <class T>
  std::vector<OpenOrbitalOptimizer::Matrix<T>>
  to_eigen(const std::vector<arma::Mat<T>> & v) {
    std::vector<OpenOrbitalOptimizer::Matrix<T>> out;
    out.reserve(v.size());
    for (const auto & m : v) out.push_back(to_eigen(m));
    return out;
  }

  template <class T>
  std::vector<arma::Mat<T>>
  to_arma(const std::vector<OpenOrbitalOptimizer::Matrix<T>> & v) {
    std::vector<arma::Mat<T>> out;
    out.reserve(v.size());
    for (const auto & e : v) out.push_back(to_arma(e));
    return out;
  }

  template <class T>
  std::vector<arma::Col<T>>
  to_arma(const std::vector<OpenOrbitalOptimizer::Vector<T>> & v) {
    std::vector<arma::Col<T>> out;
    out.reserve(v.size());
    for (const auto & e : v) out.push_back(to_arma(e));
    return out;
  }

} // namespace eaa

#endif
