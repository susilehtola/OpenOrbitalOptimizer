

# File armadillo\_compat.hpp

[**File List**](files.md) **>** [**openorbitaloptimizer**](dir_3072c93c56dfbbd2cb4eee0809487533.md) **>** [**armadillo\_compat.hpp**](armadillo__compat_8hpp.md)

[Go to the documentation of this file](armadillo__compat_8hpp.md)


```C++
/*
 *                This Source Code Form is subject to the
 *                terms of the Mozilla Public License, v. 2.0.
 *                If a copy of the MPL was not distributed
 *                with this file, You can obtain one at
 *                http://mozilla.org/MPL/2.0/.
 *
 *           Copyright (c) 2026 Susi Lehtola
 */


#ifndef OPENORBITALOPTIMIZER_ARMADILLO_COMPAT_HPP
#define OPENORBITALOPTIMIZER_ARMADILLO_COMPAT_HPP

#include "scfsolver.hpp"

#include <armadillo>

#include <complex>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace OpenOrbitalOptimizer {
namespace Armadillo {

  // ---- Armadillo <-> Eigen conversion helpers ------------------------

  template <class T>
  inline OpenOrbitalOptimizer::Matrix<T> to_eigen(const arma::Mat<T> & A) {
    return Eigen::Map<const OpenOrbitalOptimizer::Matrix<T>>(
        A.memptr(), A.n_rows, A.n_cols);
  }

  template <class T>
  inline OpenOrbitalOptimizer::Vector<T> to_eigen(const arma::Col<T> & v) {
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
  inline arma::Mat<T> to_arma(const OpenOrbitalOptimizer::Matrix<T> & E) {
    return arma::Mat<T>(E.data(), E.rows(), E.cols());
  }

  template <class T>
  inline arma::Col<T> to_arma(const OpenOrbitalOptimizer::Vector<T> & v) {
    return arma::Col<T>(v.data(), v.size());
  }

  template <class T>
  inline std::vector<OpenOrbitalOptimizer::Matrix<T>>
  to_eigen(const std::vector<arma::Mat<T>> & v) {
    std::vector<OpenOrbitalOptimizer::Matrix<T>> out;
    out.reserve(v.size());
    for (const auto & m : v) out.push_back(to_eigen(m));
    return out;
  }

  template <class T>
  inline std::vector<OpenOrbitalOptimizer::Vector<T>>
  to_eigen(const std::vector<arma::Col<T>> & v) {
    std::vector<OpenOrbitalOptimizer::Vector<T>> out;
    out.reserve(v.size());
    for (const auto & c : v) out.push_back(to_eigen(c));
    return out;
  }

  template <class T>
  inline std::vector<arma::Mat<T>>
  to_arma(const std::vector<OpenOrbitalOptimizer::Matrix<T>> & v) {
    std::vector<arma::Mat<T>> out;
    out.reserve(v.size());
    for (const auto & e : v) out.push_back(to_arma(e));
    return out;
  }

  template <class T>
  inline std::vector<arma::Col<T>>
  to_arma(const std::vector<OpenOrbitalOptimizer::Vector<T>> & v) {
    std::vector<arma::Col<T>> out;
    out.reserve(v.size());
    for (const auto & e : v) out.push_back(to_arma(e));
    return out;
  }

  // ---- Legacy container typedefs (Armadillo-shaped) ------------------

  template <class T> using OrbitalBlock = arma::Mat<T>;
  template <class T> using Orbitals = std::vector<OrbitalBlock<T>>;
  template <class T> using OrbitalGradientBlock = arma::Mat<T>;
  template <class T> using OrbitalGradients = std::vector<OrbitalGradientBlock<T>>;
  template <class T> using DiagonalOrbitalHessianBlock = arma::Mat<T>;
  template <class T> using DiagonalOrbitalHessians = std::vector<DiagonalOrbitalHessianBlock<T>>;
  template <class T> using OrbitalBlockOccupations = arma::Col<T>;
  template <class T> using OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;
  template <class Torb, class Tbase>
  using DensityMatrix = std::pair<Orbitals<Torb>, OrbitalOccupations<Tbase>>;
  template <class T> using OrbitalEnergies = std::vector<arma::Col<T>>;
  template <class T> using FockMatrixBlock = arma::Mat<T>;
  template <class T> using FockMatrix = std::vector<FockMatrixBlock<T>>;
  template <class Torb, class Tbase>
  using DiagonalizedFockMatrix = std::pair<Orbitals<Torb>, OrbitalEnergies<Tbase>>;
  template <class Torb, class Tbase>
  using FockBuilderReturn = std::pair<Tbase, FockMatrix<Torb>>;
  template <class Torb, class Tbase>
  using FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;
  using OrbitalRotation = std::tuple<size_t, arma::uword, arma::uword>;

  // ---- Legacy SCFSolver<Torb, Tbase> wrapper -------------------------

  template <class Torb, class Tbase>
  class SCFSolver {
    using EigenSolver = OpenOrbitalOptimizer::SCFSolver<Torb, Tbase>;
    using EigenDM = OpenOrbitalOptimizer::DensityMatrix<Torb, Tbase>;
    using EigenFR = OpenOrbitalOptimizer::FockBuilderReturn<Torb, Tbase>;

    FockBuilder<Torb, Tbase> arma_fock_builder_;
    EigenSolver impl_;

  public:
    SCFSolver(const arma::uvec & number_of_blocks_per_particle_type,
              const arma::Col<Tbase> & maximum_occupation,
              const arma::Col<Tbase> & number_of_particles,
              const FockBuilder<Torb, Tbase> & fock_builder,
              const std::vector<std::string> & block_descriptions)
      : arma_fock_builder_(fock_builder),
        impl_(to_eigen(number_of_blocks_per_particle_type),
              to_eigen(maximum_occupation),
              to_eigen(number_of_particles),
              [this](const EigenDM & dm) -> EigenFR {
                DensityMatrix<Torb, Tbase> arma_dm{
                    to_arma(dm.first),
                    to_arma(dm.second)};
                FockBuilderReturn<Torb, Tbase> arma_ret = arma_fock_builder_(arma_dm);
                return std::make_pair(arma_ret.first, to_eigen(arma_ret.second));
              },
              block_descriptions) {}

    // ---- Pass-through configuration ---------------------------------

    void verbosity(int v)                        { impl_.verbosity(v); }
    int  verbosity() const                       { return impl_.verbosity(); }
    void convergence_threshold(Tbase t)          { impl_.convergence_threshold(t); }
    Tbase convergence_threshold() const          { return impl_.convergence_threshold(); }
    void maximum_iterations(size_t n)            { impl_.maximum_iterations(n); }
    size_t maximum_iterations() const            { return impl_.maximum_iterations(); }
    bool frozen_occupations() const              { return impl_.frozen_occupations(); }
    void frozen_occupations(bool b)              { impl_.frozen_occupations(b); }
    void error_norm(const std::string & n)       { impl_.error_norm(n); }

    void diis_epsilon(Tbase e)                   { impl_.diis_epsilon(e); }
    Tbase diis_epsilon() const                   { return impl_.diis_epsilon(); }
    void diis_threshold(Tbase e)                 { impl_.diis_threshold(e); }
    Tbase diis_threshold() const                 { return impl_.diis_threshold(); }
    void diis_diagonal_damping(Tbase e)          { impl_.diis_diagonal_damping(e); }
    Tbase diis_diagonal_damping() const          { return impl_.diis_diagonal_damping(); }
    void diis_restart_factor(Tbase e)            { impl_.diis_restart_factor(e); }
    Tbase diis_restart_factor() const            { return impl_.diis_restart_factor(); }
    void optimal_damping_threshold(Tbase e)      { impl_.optimal_damping_threshold(e); }
    Tbase optimal_damping_threshold() const      { return impl_.optimal_damping_threshold(); }
    void optimal_damping_degeneracy_threshold(Tbase e) { impl_.optimal_damping_degeneracy_threshold(e); }
    Tbase optimal_damping_degeneracy_threshold() const { return impl_.optimal_damping_degeneracy_threshold(); }
    void maximum_history_length(int n)           { impl_.maximum_history_length(n); }
    int  maximum_history_length() const          { return impl_.maximum_history_length(); }
    void oda_restart_steps(int n)                { impl_.oda_restart_steps(n); }
    int  oda_restart_steps() const               { return impl_.oda_restart_steps(); }
    void orbital_rotation_steps_after_oda(size_t n) { impl_.orbital_rotation_steps_after_oda(n); }
    size_t orbital_rotation_steps_after_oda() const { return impl_.orbital_rotation_steps_after_oda(); }
    size_t last_polytope_dimension() const       { return impl_.last_polytope_dimension(); }
    size_t last_active_rotation_count() const    { return impl_.last_active_rotation_count(); }
    size_t number_of_fock_evaluations() const    { return impl_.number_of_fock_evaluations(); }

    void fixed_number_of_particles_per_block(const arma::Col<Tbase> & v) {
      impl_.fixed_number_of_particles_per_block(to_eigen(v));
    }

    // ---- Initialisation ---------------------------------------------

    void initialize_with_fock(const FockMatrix<Torb> & fock_guess) {
      impl_.initialize_with_fock(to_eigen(fock_guess));
    }
    void initialize_with_orbitals(const Orbitals<Torb> & orbitals,
                                   const OrbitalOccupations<Tbase> & occupations) {
      impl_.initialize_with_orbitals(to_eigen(orbitals), to_eigen(occupations));
    }

    // ---- Driving the SCF --------------------------------------------

    void run(const std::string & methods = "DIIS + ODA + CG") { impl_.run(methods); }
    void run_optimal_damping()                   { impl_.run("ODA + CG"); }
    bool converged() const                       { return impl_.converged(); }
    void brute_force_search_for_lowest_configuration() {
      impl_.brute_force_search_for_lowest_configuration();
    }

    // ---- Solution retrieval -----------------------------------------

    DensityMatrix<Torb, Tbase> get_solution(size_t ihist = 0) const {
      auto eigen_dm = impl_.get_solution(ihist);
      return {to_arma(eigen_dm.first), to_arma(eigen_dm.second)};
    }
    Orbitals<Torb> get_orbitals(size_t ihist = 0) const {
      return to_arma(impl_.get_orbitals(ihist));
    }
    OrbitalOccupations<Tbase> get_orbital_occupations(size_t ihist = 0) const {
      return to_arma(impl_.get_orbital_occupations(ihist));
    }
    FockBuilderReturn<Torb, Tbase> get_fock_build(size_t ihist = 0) const {
      auto eigen_fb = impl_.get_fock_build(ihist);
      return {eigen_fb.first, to_arma(eigen_fb.second)};
    }
    FockMatrix<Torb> get_fock_matrix(size_t ihist = 0) const {
      return to_arma(impl_.get_fock_matrix(ihist));
    }
    Tbase get_energy(size_t ihist = 0) const     { return impl_.get_energy(ihist); }

    // ---- Diagnostics & utilities ------------------------------------

    void print_history() const                   { impl_.print_history(); }
    void reset_history()                         { impl_.reset_history(); }

    DiagonalizedFockMatrix<Torb, Tbase> compute_orbitals(const FockMatrix<Torb> & fock) const {
      auto eigen_diag = impl_.compute_orbitals(to_eigen(fock));
      return {to_arma(eigen_diag.first), to_arma(eigen_diag.second)};
    }
    OrbitalOccupations<Tbase> update_occupations(const OrbitalEnergies<Tbase> & orbital_energies) const {
      return to_arma(impl_.update_occupations(to_eigen(orbital_energies)));
    }
  };

} // namespace Armadillo
} // namespace OpenOrbitalOptimizer

#endif
```


