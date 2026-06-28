

# File types.hpp

[**File List**](files.md) **>** [**openorbitaloptimizer**](dir_3072c93c56dfbbd2cb4eee0809487533.md) **>** [**types.hpp**](types_8hpp.md)

[Go to the documentation of this file](types_8hpp.md)


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

  template <class T>
  using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  template <class T>
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using Index = Eigen::Index;
  using IndexVector = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

  template <class Tbase, bool IsComplex>
  using OrbitalScalar = std::conditional_t<IsComplex, std::complex<Tbase>, Tbase>;

  template <class T> using OrbitalBlock = Matrix<T>;
  template <class T> using Orbitals = std::vector<OrbitalBlock<T>>;

  template <class T> using OrbitalGradientBlock = Matrix<T>;
  template <class T> using OrbitalGradients = std::vector<OrbitalGradientBlock<T>>;

  template <class T> using DiagonalOrbitalHessianBlock = Matrix<T>;
  template <class T> using DiagonalOrbitalHessians = std::vector<DiagonalOrbitalHessianBlock<T>>;

  template <class T> using OrbitalBlockOccupations = Vector<T>;
  template <class T> using OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;

  template <class Torb, class Tbase>
  using DensityMatrix = std::pair<Orbitals<Torb>, OrbitalOccupations<Tbase>>;

  template <class T> using OrbitalEnergies = std::vector<Vector<T>>;

  template <class T> using FockMatrixBlock = Matrix<T>;
  template <class T> using FockMatrix = std::vector<FockMatrixBlock<T>>;

  template <class Torb, class Tbase>
  using DiagonalizedFockMatrix = std::pair<Orbitals<Torb>, OrbitalEnergies<Tbase>>;

  template <class Torb, class Tbase>
  using FockBuilderReturn = std::pair<Tbase, FockMatrix<Torb>>;

  template <class Torb, class Tbase>
  using FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;

  template <class Torb, class Tbase>
  using OrbitalHistoryEntry = std::tuple<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>, size_t>;
  template <class Torb, class Tbase>
  using OrbitalHistory = std::vector<OrbitalHistoryEntry<Torb, Tbase>>;

  using OrbitalRotation = std::tuple<size_t, Index, Index>;

} // namespace OpenOrbitalOptimizer

#endif
```


