/*
 *                This Source Code Form is subject to the
 *                terms of the Mozilla Public License, v. 2.0.
 *                If a copy of the MPL was not distributed
 *                with this file, You can obtain one at
 *                http://mozilla.org/MPL/2.0/.
 *
 *           Copyright (c) 2025 Susi Lehtola
 */
#ifndef OPENORBITALOPTIMIZER_TYPES_HPP
#define OPENORBITALOPTIMIZER_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <complex>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace OpenOrbitalOptimizer {

  /// Dense matrix alias.
  template <class T>
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  /// Dense column-vector alias.
  template <class T>
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  /// Index type.
  using Index = Eigen::Index;
  /// Index column vector (replacement for arma::uvec).
  using IndexVector = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

  /// Resolves the orbital scalar type from (Tbase, IsComplex):
  /// Tbase for IsComplex=false, std::complex<Tbase> for IsComplex=true.
  template <class Tbase, bool IsComplex>
  using OrbitalScalar = std::conditional_t<IsComplex, std::complex<Tbase>, Tbase>;

  /// Orbital coefficients in one symmetry block (rows = basis, cols = orbitals).
  template <class T> using OrbitalBlock = Matrix<T>;
  /// One OrbitalBlock per symmetry block, per particle type.
  template <class T> using Orbitals = std::vector<OrbitalBlock<T>>;

  /// Block-diagonal orbital gradient.
  template <class T> using OrbitalGradientBlock = Matrix<T>;
  template <class T> using OrbitalGradients = std::vector<OrbitalGradientBlock<T>>;

  /// Diagonal orbital Hessian (one column per orbital).
  template <class T> using DiagonalOrbitalHessianBlock = Matrix<T>;
  template <class T> using DiagonalOrbitalHessians = std::vector<DiagonalOrbitalHessianBlock<T>>;

  /// Real-valued occupations in one symmetry block.
  template <class T> using OrbitalBlockOccupations = Vector<T>;
  template <class T> using OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;

  /// Density matrix bundle: orbitals + occupations.
  template <class Torb, class Tbase>
  using DensityMatrix = std::pair<Orbitals<Torb>, OrbitalOccupations<Tbase>>;

  /// Real-valued orbital energies in one symmetry block.
  template <class T> using OrbitalEnergies = std::vector<Vector<T>>;

  /// Fock matrix in one symmetry block.
  template <class T> using FockMatrixBlock = Matrix<T>;
  template <class T> using FockMatrix = std::vector<FockMatrixBlock<T>>;

  /// Diagonalized Fock matrix: orbitals + energies.
  template <class Torb, class Tbase>
  using DiagonalizedFockMatrix = std::pair<Orbitals<Torb>, OrbitalEnergies<Tbase>>;

  /// Fock builder return value: (energy, Fock).
  template <class Torb, class Tbase>
  using FockBuilderReturn = std::pair<Tbase, FockMatrix<Torb>>;

  /// User-supplied Fock builder callback signature.
  template <class Torb, class Tbase>
  using FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;

  /// Optional batched Fock builder: given a list of densities, return the
  /// corresponding list of (energy, Fock) pairs. Enables per-vertex sweeps
  /// in the ODA polytope face-minimization step.
  template <class Torb, class Tbase>
  using BatchedFockBuilder = std::function<std::vector<FockBuilderReturn<Torb, Tbase>>(const std::vector<DensityMatrix<Torb, Tbase>> &)>;

  /// Single history entry: density, Fock-builder output, generation id.
  template <class Torb, class Tbase>
  using OrbitalHistoryEntry = std::tuple<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>, size_t>;
  template <class Torb, class Tbase>
  using OrbitalHistory = std::vector<OrbitalHistoryEntry<Torb, Tbase>>;

  /// (block index, orbital i, orbital j) describing a single orbital rotation.
  using OrbitalRotation = std::tuple<size_t, Index, Index>;

} // namespace OpenOrbitalOptimizer

#endif
