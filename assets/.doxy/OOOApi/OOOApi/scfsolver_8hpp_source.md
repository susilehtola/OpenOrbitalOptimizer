

# File scfsolver.hpp

[**File List**](files.md) **>** [**openorbitaloptimizer**](dir_3072c93c56dfbbd2cb4eee0809487533.md) **>** [**scfsolver.hpp**](scfsolver_8hpp.md)

[Go to the documentation of this file](scfsolver_8hpp.md)


```C++
/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once
#include "types.hpp"
#include "eigen_compat.hpp"

#include <algorithm>
#include <any>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace OpenOrbitalOptimizer {

  namespace HelperRoutines {
    template<typename T>
    std::tuple<T,T,T,T> fit_cubic_polynomial_with_derivatives(T E0, T dE0, T x1, T E1, T dE1) {
      T a0 = E0;
      T a1 = dE0;
      T x1sq = x1*x1;
      T x1cu = x1sq*x1;
      T a3 = (dE1 + dE0)/x1sq - 2*(E1 - E0)/x1cu;
      T a2 = 3*(E1 - E0)/x1sq - (2*dE0 + dE1)/x1;
      return std::make_tuple(a0, a1, a2, a3);
    }

    template<typename T>
    std::pair<T,T> cubic_polynomial_zeros(T a0, T a1, T a2, T a3) {
      (void) a0;
      // Solve 3*a3*x^2 + 2*a2*x + a1 = 0
      T eps = std::numeric_limits<T>::epsilon();
      if(std::abs(a3) <= eps) {
        if(std::abs(a2) <= eps)
          throw std::logic_error("Cubic derivative is constant; no extrema");
        T x = -a1/(2*a2);
        return std::make_pair(x, x);
      }
      T disc = 4*a2*a2 - 12*a3*a1;
      if(disc < 0)
        throw std::logic_error("Cubic derivative has no real roots");
      T sq = std::sqrt(disc);
      T x1 = (-2*a2 - sq)/(6*a3);
      T x2 = (-2*a2 + sq)/(6*a3);
      return std::make_pair(x1, x2);
    }

  }

  template<typename Torb, typename Tbase> class SCFSolver {
    static_assert(std::is_same_v<Torb, Tbase> ||
                  std::is_same_v<Torb, std::complex<Tbase>>,
                  "SCFSolver<Torb, Tbase>: Torb must be either Tbase or "
                  "std::complex<Tbase>");
    /* Input data section */
    IndexVector number_of_blocks_per_particle_type_;
    Vector<Tbase> maximum_occupation_;
    Vector<Tbase> number_of_particles_;
    FockBuilder<Torb, Tbase> fock_builder_;
    BatchedFockBuilder<Torb, Tbase> batched_fock_builder_;
    std::vector<std::string> block_descriptions_;
    std::function<void(const std::map<std::string,std::any> & data)> callback_function_;
    std::function<bool(const std::map<std::string,std::any> & data)> callback_convergence_function_;

    Vector<Tbase> fixed_number_of_particles_per_block_;
    bool frozen_occupations_;

    int verbosity_;

    /* Internal data section */
    size_t number_of_blocks_;
    OrbitalHistory<Torb, Tbase> orbital_history_;
    OrbitalOccupations<Tbase> orbital_occupations_;

    size_t number_of_fock_evaluations_ = 0;

    size_t maximum_iterations_ = 128;
    Tbase diis_epsilon_ = 1e-1;
    Tbase diis_threshold_ = 1e-4;
    Tbase diis_diagonal_damping_ = 0.02;
    Tbase diis_restart_factor_ = 1e-4;

    Tbase optimal_damping_threshold_ = 1.0;

    Tbase density_restart_factor_ = 1e-4;
    int maximum_history_length_ = 10;
    int oda_restart_steps_ = 5;
    Tbase convergence_threshold_ = 1e-7;
    Tbase noise_safety_factor_ = 10;
    Tbase noise_floor_ = 0;
    std::string error_norm_ = "rms";

    std::string methods_ = "DIIS + ODA + CG";

    Tbase minimal_gradient_projection_ = 1e-4;
    Tbase occupied_threshold_ = 1e-6;
    Tbase initial_level_shift_ = 1e-3;
    Tbase level_shift_factor_ = 2.0;

    Tbase optimal_damping_degeneracy_threshold_ = 1e-2;
    Tbase occupation_change_threshold_ = 1e-6;
    size_t orbital_rotation_steps_after_oda_ = 0;
    size_t last_polytope_dimension_ = 0;
    size_t last_active_rotation_count_ = 0;

    Tbase old_energy_ = 0.0;

    Vector<Tbase> previous_orbital_gradient_;
    Vector<Tbase> previous_orbital_direction_;
    std::vector<OrbitalRotation> previous_orbital_dofs_;

    struct LBFGSState {
      std::deque<Vector<Tbase>> s;
      std::deque<Vector<Tbase>> y;
      std::deque<Tbase> rho;
      Vector<Tbase> pending_s;
      Vector<Tbase> pending_g;
      std::vector<OrbitalRotation> history_dofs;
    };
    std::unique_ptr<LBFGSState> lbfgs_;

    struct AllowedMethods {
      bool diis = false, oda = false, cg = false, lbfgs = false;
      bool orbital_rotation() const { return cg || lbfgs; }
      bool any() const { return diis || oda || orbital_rotation(); }
    };

    static AllowedMethods parse_method_string(const std::string & methods) {
      AllowedMethods allowed;
      std::string s = methods;
      std::transform(s.begin(), s.end(), s.begin(),
                     [](unsigned char c){ return std::tolower(c); });
      std::istringstream iss(s);
      std::string token;
      while(std::getline(iss, token, '+')) {
        while(!token.empty() && std::isspace((unsigned char)token.front()))
          token.erase(token.begin());
        while(!token.empty() && std::isspace((unsigned char)token.back()))
          token.pop_back();
        if(token.empty()) continue;
        if(token == "diis") allowed.diis = true;
        else if(token == "oda") allowed.oda = true;
        else if(token == "cg") allowed.cg = true;
        else if(token == "lbfgs") allowed.lbfgs = true;
        else throw std::logic_error("Unknown method '" + token
            + "' in methods string '" + methods
            + "' (allowed: DIIS, ODA, CG, LBFGS)");
      }
      if(!allowed.any())
        throw std::logic_error("No methods enabled in '" + methods + "'");
      return allowed;
    }

    static std::string to_upper_copy(const std::string & s) {
      std::string out = s;
      std::transform(out.begin(), out.end(), out.begin(),
                     [](unsigned char c){ return std::toupper(c); });
      return out;
    }

    /* Internal functions */
    bool empty_block(size_t iblock) const {
      // Check if Fock matrix has zero dimension
      if(iblock>=std::get<0>(orbital_history_[0]).first.size())
        throw std::logic_error("Trying to check empty block for nonexistent index!\n");
      return std::get<1>(orbital_history_[0]).second[iblock].size() == 0;
    }

    Matrix<Torb> get_density_matrix_block(size_t ihist, size_t iblock) const {
      const auto orbitals = get_orbital_block(ihist, iblock);
      const auto occupations = get_orbital_occupation_block(ihist, iblock);
      return orbitals * occupations.asDiagonal() * orbitals.adjoint();
    }

    OrbitalBlock<Torb> get_orbital_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbitals for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbitals for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).first[iblock];
    }

    OrbitalBlockOccupations<Tbase> get_orbital_occupation_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).second[iblock];
    }

    Tbase get_lowest_energy_after_index(size_t index=0) const {
      bool initialized = false;
      Tbase lowest_energy;
      for(size_t i=0;i<orbital_history_.size();i++) {
        if(get_index(i) > index) {
          if(not initialized) {
            initialized=true;
            lowest_energy = get_energy(i);
          } else {
            lowest_energy = std::min(lowest_energy, get_energy(i));
          }
        }
      }
      if(initialized)
        return lowest_energy;
      else {
        print_history();
        fflush(stdout);
        std::ostringstream oss;
        oss << "Did not find any entries with index greater than " << index << "!\n";
        throw std::logic_error(oss.str());
      }
    }

    size_t get_index(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access index for nonexistent history member!\n");
      return std::get<2>(orbital_history_[ihist]);
    }

    size_t largest_index() const {
      size_t index = get_index(0);
      for(size_t i=1;i<orbital_history_.size();i++) {
        index = std::max(index, get_index(i));
      }
      return index;
    }

    IndexVector matrix_dimension() const {
      const auto & fock = std::get<1>(orbital_history_[0]).second;
      IndexVector dim(fock.size());
      for(size_t i=0;i<fock.size();i++)
        dim(i) = fock[i].cols();
      return dim;
    }

    Matrix<Torb> get_fock_matrix_block(size_t ihist, size_t iblock) const {
      return std::get<1>(orbital_history_[ihist]).second[iblock];
    }

    Vector<Tbase> vectorise(const Matrix<Torb> & mat) const {
      return vectorise_real_imag(mat);
    }

    Vector<Tbase> vectorise(const std::vector<Matrix<Torb>> & mat) const {
      // Compute length of return vector
      size_t N=0;

      std::vector<Vector<Tbase>> vectors(mat.size());
      for(size_t iblock=0;iblock<mat.size();iblock++) {
        if(mat[iblock].size()==0)
          continue;
        vectors[iblock]=vectorise(mat[iblock]);
        N += vectors[iblock].size();
      }

      Vector<Tbase> v = Vector<Tbase>::Zero(N);
      size_t ioff=0;
      for(size_t iblock=0;iblock<vectors.size();iblock++) {
        if(mat[iblock].size()==0)
          continue;
        v.segment(ioff, vectors[iblock].size())=vectors[iblock];
        ioff += vectors[iblock].size();
      }

      return v;
    }

    Matrix<Torb> matricise(const Vector<Tbase> & vec, size_t nrows, size_t ncols) const {
      if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
        if(vec.size() != (Index)(nrows*ncols)) {
          std::ostringstream oss;
          oss << "Matricise error: expected " << nrows*ncols << " elements for " << nrows << " x " << ncols << " real matrix, but got " << vec.size() << " instead!\n";
          throw std::logic_error(oss.str());
        }
        return Eigen::Map<const Matrix<Torb>>(vec.data(), nrows, ncols);
      } else {
        if(vec.size() != (Index)(2*nrows*ncols)) {
          std::ostringstream oss;
          oss << "Matricise error: expected " << 2*nrows*ncols << " elements for " << nrows << " x " << ncols << " complex matrix, but got " << vec.size() << " instead!\n";
          throw std::logic_error(oss.str());
        }

        Eigen::Map<const Matrix<Tbase>> real_part(vec.data(), nrows, ncols);
        Eigen::Map<const Matrix<Tbase>> imag_part(vec.data()+nrows*ncols, nrows, ncols);
        Matrix<Torb> mat = real_part.template cast<Torb>()
                        + imag_part.template cast<Torb>() * std::complex<Tbase>(Tbase{0}, Tbase{1});
        return mat;
      }
    }

    std::vector<Matrix<Torb>> matricise(const Vector<Tbase> & vec, const IndexVector & dim) const {
      std::vector<Matrix<Torb>> mat(dim.size());
      size_t ioff = 0;
      for(Index iblock=0; iblock<dim.size(); iblock++) {
        if(dim(iblock)==0)
          continue;
        size_t sz = (size_t)dim(iblock) * (size_t)dim(iblock);
        if constexpr (Eigen::NumTraits<Torb>::IsComplex) {
          sz *= 2;
        }
        mat[iblock] = matricise(vec.segment(ioff, sz), dim(iblock), dim(iblock));
        ioff += sz;
      }
      return mat;
    }

    Matrix<Torb> diis_residual(size_t ihist, size_t iblock) const {
      // Error is measured by FPS-SPF = FP - PF, since we have a unit metric.
      auto F = get_fock_matrix_block(ihist, iblock);
      auto P = get_density_matrix_block(ihist, iblock);

      // Though F and P should be symmetric by construction, explicitly symmetrize
      // them and compute commutator to avoid symmetry-related numerical issues.
      Matrix<Torb> F_sym = 0.5 * (F + F.adjoint());
      Matrix<Torb> P_sym = 0.5 * (P + P.adjoint());
      Matrix<Torb> PF = P_sym * F_sym;
      Matrix<Torb> FP = F_sym * P_sym;
      Matrix<Torb> commutator = PF - FP;

      // To make the L^infty error independent of the underlying basis
      // set, we project the residual into the best orbitals we have
      auto C = get_orbital_block(0, iblock);
      commutator = C.adjoint() * commutator * C;
      return commutator;
    }

    std::vector<Matrix<Torb>> diis_residual(size_t ihist) const {
      std::vector<Matrix<Torb>> residuals(number_of_blocks_);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        residuals[iblock] = diis_residual(ihist, iblock);
      }
      return residuals;
    }

    Vector<Tbase> diis_error_vector(size_t ihist, size_t iblock) const {
      return vectorise(diis_residual(ihist, iblock));
    }

    Vector<Tbase> diis_error_vector(size_t ihist) const {
      // Form error vectors
      std::vector<Vector<Tbase>> error_vectors(number_of_blocks_);
      for(size_t iblock = 0; iblock<number_of_blocks_;iblock++) {
        error_vectors[iblock] = diis_error_vector(ihist, iblock);
        if(verbosity_>=20)
          printf("ihist %i block %i error vector norm %e\n", (int) ihist, (int) iblock, norm(error_vectors[iblock]));
        if(verbosity_>=30)
          std::cout << error_vectors[iblock] << std::endl;
      }

      // Compound error vector
      size_t nelem = 0;
      for(auto & block: error_vectors)
        nelem += block.size();

      Vector<Tbase> return_vector(nelem);
      size_t ioff=0;
      for(auto & block: error_vectors) {
        if(block.size()>0) {
          return_vector.segment(ioff, block.size()) = block;
          ioff += block.size();
        }
      }
      if(ioff!=nelem)
        throw std::logic_error("Indexing error!\n");

      return return_vector;
    }

    Tbase compute_noise_floor() const {
      const Tbase eps = std::numeric_limits<Tbase>::epsilon();
      std::vector<Matrix<Torb>> mock(number_of_blocks_);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        auto F = get_fock_matrix_block(0, iblock);
        Index n = F.rows();
        Tbase per_elem = eps * F.norm();
        Torb seed;
        if constexpr (Eigen::NumTraits<Torb>::IsComplex)
          // Load both real and imag halves of vectorise_real_imag so
          // the clamp is not looser for complex than for real.
          seed = Torb(per_elem, per_elem);
        else
          seed = Torb(per_elem);
        mock[iblock] = Matrix<Torb>::Constant(n, n, seed);
      }
      return norm(vectorise(mock));
    }

    Tbase diis_error_matrix_element(size_t ihist, size_t jhist) const {
      Tbase el=0.0;
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        Vector<Tbase> ei(diis_error_vector(ihist, iblock));
        Vector<Tbase> ej(diis_error_vector(jhist, iblock));
        el += ei.dot(ej);
      }
      return el;
    }

    Matrix<Tbase> diis_error_matrix(const std::vector<size_t> & mask) const {
      // The error matrix is given by the orbital gradient dot products
      const size_t N=mask.size();
      Matrix<Tbase> B = Matrix<Tbase>::Zero(N,N);

      for(size_t ihist=0; ihist<N; ihist++) {
        for(size_t jhist=0; jhist<=ihist; jhist++) {
          B(ihist, jhist) = B(jhist, ihist) = diis_error_matrix_element(mask[ihist], mask[jhist]);
        }
      }
      return B;
    }

    Vector<Tbase> diis_error_matrix_diagonal() const {
      Vector<Tbase> B = Vector<Tbase>::Zero(orbital_history_.size());
      for(Index ihist=0; ihist<B.size(); ihist++) {
        B(ihist) = diis_error_matrix_element(ihist, ihist);
      }
      return B;
    }

    Matrix<Tbase> diis_error_matrix() const {
      std::vector<size_t> mask(orbital_history_.size());
      for(size_t i=0;i<mask.size();i++)
        mask[i]=i;
      return diis_error_matrix(mask);
    }

    Vector<Tbase> diis_weights() const {
      // Only use reference points with error residuals that are sufficiently small
      std::vector<size_t> history_mask(orbital_history_.size());
      for(size_t i=0;i<history_mask.size();i++)
        history_mask[i]=i;
      Vector<Tbase> residuals(history_mask.size());
      for(Index i=0;i<residuals.size();i++)
        residuals(i) = diis_error_matrix_element(history_mask[i], history_mask[i]);
      Tbase min_residual = residuals.minCoeff();
      for(size_t i=history_mask.size()-1;i<history_mask.size();i--)
        // Criterion from Chupin et al, 2012
        if(residuals(i)*diis_restart_factor_ > min_residual)
          history_mask.erase(history_mask.begin()+i);
      size_t nrestart = orbital_history_.size()-history_mask.size();
      if(verbosity_>=10 and nrestart>0)
        printf("Removed %i entries corresponding to large DIIS errors\n", (int) nrestart);

      // Set up the DIIS error matrix
      const size_t N=history_mask.size();
      Matrix<Tbase> B = Matrix<Tbase>::Constant(N+1, N+1, Tbase(-1.0));
      B.block(0, 0, N, N) = diis_error_matrix(history_mask);
      B(N,N)=0.0;

      // Apply the diagonal damping
      B.block(0, 0, N, N).diagonal() *= 1.0+diis_diagonal_damping_;

      // To improve numerical conditioning, scale entries of error
      // matrix such that the last diagonal element is one; Eckert et
      // al, J. Comput. Chem 18. 1473-1483 (1997)
      Vector<Tbase> Bdiag(B.diagonal());
      Tbase diagmin = Bdiag.head(N).minCoeff();
      if(diagmin != 0.0)
        B.block(0, 0, N, N) /= diagmin;

      // Right-hand side of equation is
      Vector<Tbase> rh = Vector<Tbase>::Zero(N+1);
      rh(N)=-1.0;

      // Solve the equation
      Vector<Tbase> sol = B.colPivHouseholderQr().solve(rh);
      Vector<Tbase> diis_weights = sol.head(N);

      // Pad to full space
      Vector<Tbase> diis_weights_full = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t i=0;i<history_mask.size();i++)
        diis_weights_full[history_mask[i]] = diis_weights[i];

      return diis_weights_full;
    }
    Vector<Tbase> aediis_weights(const Vector<Tbase> & b, const Matrix<Tbase> & A) const {
      if(b.size()==1) {
        // Nothing to optimize
        return Vector<Tbase>::Ones(b.size());
      }

      // Parameters
      const size_t max_iter = 1000000;
      const Tbase df_tol = 1e-8;

      // Function to evaluate function value
      std::function<Tbase(const Vector<Tbase> & x)> fx = [b, A](const Vector<Tbase> & x) {
        return Tbase(0.5)*(x.transpose()*A*x).value() + b.dot(x);
      };

      // Function to determine optimal step
      std::function<Tbase(const Vector<Tbase> &, const Vector<Tbase> &)> optimal_step = [b, A](const Vector<Tbase> & current_direction, const Vector<Tbase> & x) {
        return -((current_direction.transpose()*A*x).value() + b.dot(current_direction))
               / (current_direction.transpose()*A*current_direction).value();
      };

      std::vector<Vector<Tbase>> xguess;
      // Center point
      xguess.push_back(Vector<Tbase>::Constant(b.size(), Tbase(1.0)/b.size()));
      // "Gauss" points
      for(Index i=0;i<b.size();i++) {
        Vector<Tbase> xtr = Vector<Tbase>::Constant(b.size(), Tbase(1.0)/(b.size()+2));
        xtr(i) *= 3;
        xguess.push_back(xtr);
      }
      // End points
      for(Index i=0;i<b.size();i++) {
        Vector<Tbase> xtr = Vector<Tbase>::Zero(b.size());
        xtr(i) = 1.0;
        xguess.push_back(xtr);
      }

      // Find minimum
      Vector<Tbase> yguess(xguess.size());
      for(size_t i=0;i<xguess.size();i++)
        yguess[i] = fx(xguess[i]);

      IndexVector idx(sort_index_ascending(yguess));
      Vector<Tbase> x = xguess[idx[0]];
      //std::cout << "Initial x: " << x.transpose() << std::endl;

      Matrix<Tbase> search_directions = Matrix<Tbase>::Identity(b.size(), b.size());

      auto current_point = fx(x);
      auto old_x = x;

      // Powell algorithm
      for(size_t imacro=0; imacro<max_iter; imacro++) {
        Tbase curval(current_point);

        for(Index i=0; i<b.size(); i++) {
          Vector<Tbase> c_i(search_directions.col(i));
          // x -> (1-step)*x + step*c_i = x + step*(c_i-x)
          Tbase step = optimal_step(c_i-x, x);
          if(!std::isnormal(step))
            continue;
          //printf("Direction %i: optimal step %e\n",i,step);
          if(step > 0.0 and step <= 1.0) {
            auto new_point = fx(x+step*(c_i-x));
            //printf("Direction %i: optimal step changes energy by %e\n",(int) i,new_point.first - current_point.first);
            if(new_point < current_point) {
              x += step*(c_i-x);
              current_point = new_point;
            }
          }
        }

        Tbase dE = current_point - curval;
        //printf("Macroiteration %i changed energy by %e\n", imacro, dE);

        // Update in x
        Vector<Tbase> dx = x - old_x;

        // Repeat line search along this direction
        Tbase step = optimal_step(dx, x);
        if(std::isnormal(step) and step > 0.0 and step <= 1.0) {
          auto new_point = fx(x+step*dx);
          if(new_point < current_point) {
            x += step*dx;
            //printf("Line search along dx changes energy by %e\n", new_point-current_point);
            current_point = new_point;
            dE = current_point - curval;
          }
        }
        old_x = x;

        //std::cout << "x: " << x.transpose() << std::endl;
        if(dE > -df_tol) {
          if(verbosity_ >= 10) {
            printf("A/EDIIS weights converged in %i macroiterations\n",(int) imacro);
            //std::cout << "xconv: " << x.transpose() << std::endl;
          }
          break;
        } else if(imacro==max_iter-1) {
          if(verbosity_ >= 10) {
            printf("A/EDIIS weights did not converge in %i macroiterations, dE=%e\n", (int) imacro, dE);
            //std::cout << "xfinal: " << x.transpose() << std::endl;
          }
        }

        /*
        // Rotate search directions. Generate a random ordering of the columns
        IndexVector rp(randperm(search_directions.cols()));
        {
          Matrix<Tbase> tmp(search_directions.rows(), search_directions.cols());
          for(Index c=0;c<search_directions.cols();c++)
            tmp.col(c) = search_directions.col(rp(c));
          search_directions = tmp;
        }
        // Mix the vectors together
        for(Index i=0;i<search_directions.cols();i++)
          for(Index j=0;j<i;j++) {
            Tbase r = 0;
            Vector<Tbase> newi = (1-r)*search_directions.col(i) + r*search_directions.col(j);
            Vector<Tbase> newj = (1-r)*search_directions.col(j) + r*search_directions.col(i);
            search_directions.col(i) = newi;
            search_directions.col(j) = newj;
          }
        */
      }

      // Handle the edge case where the last matrix has zero norm
      if(x(0)==0.0) {
        x.setZero();
        x(0)=1.0;
        // Reset search directions
        search_directions.setIdentity();
        for(Index i=0; i<b.size(); i++) {
          Vector<Tbase> c_i(search_directions.col(i));
          // x -> (1-step)*x + step*c_i = x + step*(c_i-x)
          Tbase step = optimal_step(c_i-x, x);
          if(!std::isnormal(step))
            continue;
          if(step > 0.0 and step < 1.0) {
            auto new_point = fx(x+step*(c_i-x));
            if(new_point < current_point) {
              x += step*(c_i-x);
              current_point = new_point;
            }
          }
        }
        //std::cout << "Using suboptimal solution instead: " << x.transpose() << std::endl;
      }

      //printf("Current energy %e\n",current_point);
      //throw std::logic_error("Stop");

      return x;
    }

    Vector<Tbase> adiis_linear_term() const {
      Vector<Tbase> ret = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        const auto & Dn = get_density_matrix_block(0, iblock);
        const auto & Fn = get_fock_matrix_block(0, iblock);
        for(Index ihist=0;ihist<ret.size();ihist++) {
          // D_i - D_n
          Matrix<Torb> dD(get_density_matrix_block(ihist, iblock) - Dn);
          ret(ihist) += 2.0*std::real((dD*Fn).trace());
        }
      }
      return ret;
    }

    Vector<Tbase> ediis_linear_term() const {
      Vector<Tbase> ret = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
        ret(ihist) = get_energy(ihist);
      }
      return ret;
    }

    Matrix<Tbase> adiis_quadratic_term() const {
      Matrix<Tbase> ret = Matrix<Tbase>::Zero(orbital_history_.size(), orbital_history_.size());
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        const auto & Dn = get_density_matrix_block(0, iblock);
        const auto & Fn = get_fock_matrix_block(0, iblock);
        for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
          for(size_t jhist=0;jhist<orbital_history_.size();jhist++) {
            // D_i - D_n
            Matrix<Torb> dD(get_density_matrix_block(ihist, iblock) - Dn);
            // F_j - F_n
            Matrix<Torb> dF(get_fock_matrix_block(jhist, iblock) - Fn);
            ret(ihist,jhist) += std::real((dD*dF).trace());
          }
        }
      }
      // Only the symmetric part matters; we also multiply by two
      // since we define the quadratic model as 0.5*x^T A x + b x
      return ret+ret.adjoint();
    }

    Matrix<Tbase> ediis_quadratic_term() const {
      Matrix<Tbase> ret = Matrix<Tbase>::Zero(orbital_history_.size(), orbital_history_.size());
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
          for(size_t jhist=0;jhist<orbital_history_.size();jhist++) {
            // D_i - D_j
            Matrix<Torb> dD(get_density_matrix_block(ihist, iblock) - get_density_matrix_block(jhist, iblock));
            // F_i - F_j
            Matrix<Torb> dF(get_fock_matrix_block(ihist, iblock) - get_fock_matrix_block(jhist, iblock));
            ret(ihist,jhist) -= std::real((dD*dF).trace());
          }
        }
      }
      // Only the symmetric part matters; the factor 0.5 already
      // exists in the base model
      return 0.5*(ret+ret.adjoint());
    }

    Vector<Tbase> adiis_weights() const {
      return aediis_weights(adiis_linear_term(), adiis_quadratic_term());
    }

    Vector<Tbase> ediis_weights() const {
      return aediis_weights(ediis_linear_term(), ediis_quadratic_term());
    }

    std::tuple<Vector<Tbase>,std::string> minimal_error_sampling_algorithm_weights(Tbase aediis_coeff) const {
      // Form DIIS and ADIIS weights
      Vector<Tbase> diis_w(diis_weights());
      if(verbosity_>=10) std::cout << "DIIS weights: " << diis_w.transpose() << std::endl;
      if(aediis_coeff == 0.0) {
        std::string step = "DIIS";
        return std::make_tuple(diis_w,step);
      }

      // Get various extrapolation weights
      const size_t N = orbital_history_.size();
      Vector<Tbase> adiis_w(adiis_weights());
      if(verbosity_>=10) std::cout << "ADIIS weights: " << adiis_w.transpose() << std::endl;
      Vector<Tbase> ediis_w(ediis_weights());
      if(verbosity_>=10) std::cout << "EDIIS weights: " << ediis_w.transpose() << std::endl;

      // Candidates
      Matrix<Tbase> candidate_w = Matrix<Tbase>::Zero(N, 2);
      size_t icol=0;
      candidate_w.col(icol++) = adiis_w;
      candidate_w.col(icol++) = ediis_w;
      const std::vector<std::string> weight_legend({"ADIIS", "EDIIS"});
      std::string step;

      Vector<Tbase> density_projections = Vector<Tbase>::Zero(candidate_w.cols());
      for(Index iw=0;iw<candidate_w.cols();iw++) {
        density_projections(iw) = density_projection(Vector<Tbase>(candidate_w.col(iw)));
      }
      if(verbosity_>=10)
        std::cout << "Density projections: " << density_projections.transpose() << std::endl;

      Index idx;
      density_projections.maxCoeff(&idx);
      if(verbosity_>=10)
        printf("Max density projection %e with %s weights\n",density_projections(idx),weight_legend[idx].c_str());

      Vector<Tbase> aediis_w = candidate_w.col(idx);
      Vector<Tbase> weights(aediis_coeff * aediis_w + (1.0 - aediis_coeff) * diis_w);
      if(aediis_coeff == 1.0) {
        step = weight_legend[idx];
      } else {
        step = weight_legend[idx] + "+DIIS";
      }

      return std::make_tuple(weights,step);
    }

    Tbase density_projection(const Vector<Tbase> & weights) const {
      // Get the extrapolated Fock matrix
      auto fock(extrapolate_fock(weights));

      // Reference calculation
      const auto reference_orbitals = get_orbitals();
      const auto reference_occupations = get_orbital_occupations();

      // Diagonalize the extrapolated Fock matrix
      auto diagonalized_fock = compute_orbitals(fock);
      auto & new_orbitals = diagonalized_fock.first;
      auto & new_orbital_energies = diagonalized_fock.second;

      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      return density_overlap(new_orbitals, new_occupations, reference_orbitals, reference_occupations);
    }

    Tbase occupation_difference(const OrbitalOccupations<Tbase> & old_occ, const OrbitalOccupations<Tbase> & new_occ) const {
      Tbase diff = 0.0;
      for(size_t iblock = 0; iblock<old_occ.size(); iblock++) {
        if(old_occ[iblock].size()==0)
          continue;
        Index n = std::min(new_occ[iblock].size(), old_occ[iblock].size());
        diff += (new_occ[iblock].head(n) - old_occ[iblock].head(n)).array().abs().sum();
        if(new_occ[iblock].size()>n)
          diff += new_occ[iblock].tail(new_occ[iblock].size()-n).array().abs().sum();
        else if(old_occ[iblock].size()>n)
          diff += old_occ[iblock].tail(old_occ[iblock].size()-n).array().abs().sum();
      }

      return diff;
    }

    FockMatrix<Torb> extrapolate_fock(const Vector<Tbase> & weights) const {
      if(weights.size() != (Index)orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.size() << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated Fock matrix
      FockMatrix<Torb> extrapolated_fock(number_of_blocks_);
      for(size_t iblock = 0; iblock < extrapolated_fock.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        // Apply the DIIS weight
        for(size_t ihist = 0; ihist < orbital_history_.size(); ihist++) {
          Matrix<Torb> block = weights(ihist) * get_fock_matrix_block(ihist, iblock);
          if(ihist==0) {
            extrapolated_fock[iblock] = block;
          } else {
            extrapolated_fock[iblock] += block;
          }
        }
      }

      return extrapolated_fock;
    }

    DensityMatrix<Torb, Tbase> extrapolate_density(const Vector<Tbase> & weights) const {
      if(weights.size() != (Index)orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.size() << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated density matrix
      std::vector<Matrix<Torb>> orbitals(number_of_blocks_);
      std::vector<Vector<Tbase>> occupations(number_of_blocks_);
      for(size_t iblock = 0; iblock < number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;

        Matrix<Torb> dm_block;
        for(size_t ihist = 0; ihist < orbital_history_.size(); ihist++) {
          Matrix<Torb> block = weights(ihist) * get_density_matrix_block(ihist, iblock);
          if(ihist==0) {
            dm_block = block;
          } else {
            dm_block += block;
          }
        }

        // Flip the sign so that the orbitals come in increasing occupation
        Matrix<Torb> neg_dm = -dm_block;
        Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(neg_dm);
        occupations[iblock] = es.eigenvalues();
        orbitals[iblock] = es.eigenvectors();
        occupations[iblock] *= Tbase{-1};
        // Zero out numerically zero occupations
        const Tbase zero_tol = 10*maximum_occupation_(iblock)*std::numeric_limits<Tbase>::epsilon();
        for(Index k=0; k<occupations[iblock].size(); k++) {
          if(std::abs(occupations[iblock][k]) <= zero_tol)
            occupations[iblock][k] = Tbase{0};
        }
      }

      return std::make_pair(orbitals,occupations);
    }

    OrbitalOccupations<Tbase> determine_maximum_overlap_occupations(const OrbitalOccupations<Tbase> & reference_occupations, const Orbitals<Torb> & C_reference, const Orbitals<Torb> & C_new) const {
      OrbitalOccupations<Tbase> new_occupations(reference_occupations);
      for(size_t iblock=0; iblock<new_occupations.size(); iblock++) {
        if(C_reference[iblock].size() == 0)
          continue;
        // Initialize
        new_occupations[iblock].setZero();

        // Magnitude of the overlap between the new orbitals and the reference ones
        Matrix<Tbase> orbital_projections = (C_new[iblock].adjoint()*C_reference[iblock]).array().abs().matrix();

        // Occupy the orbitals in ascending energy, especially if there are unoccupied orbitals in-between
        for(Index iorb=0; iorb<reference_occupations[iblock].size(); iorb++) {
          // Projections for this orbital
          Vector<Tbase> projection = orbital_projections.col(iorb);
          // Find the maximum index
          Index maximal_projection_index;
          Tbase maximal_projection = projection.maxCoeff(&maximal_projection_index);
          (void) maximal_projection;
          // Store projection
          new_occupations[iblock][maximal_projection_index] = reference_occupations[iblock](iorb);
          // and reset the corresponding row so that the orbital can't be reused
          orbital_projections.row(maximal_projection_index).setZero();

          //printf("Symmetry %i: reference orbital %i with occupation %.3f matches new orbital %i with projection %e\n",(int) iblock, (int) iorb, reference_occupations[iblock](iorb), (int) maximal_projection_index, maximal_projection);
        }
      }

      return new_occupations;
    }

    Tbase density_overlap(const Orbitals<Torb> & lorb, const OrbitalOccupations<Tbase> & locc, const Orbitals<Torb> & rorb, const OrbitalOccupations<Tbase> & rocc) const {
      if(lorb.size() != rorb.size() or lorb.size() != locc.size() or lorb.size() != rocc.size())
        throw std::logic_error("Inconsistent orbitals!\n");

      Tbase ovl=0.0;
      for(size_t iblock=0; iblock<lorb.size(); iblock++) {
        if(lorb[iblock].size()==0)
          continue;
        // Get orbital coefficients and occupations
        const auto & lC = lorb[iblock];
        const auto & lo = locc[iblock];
        const auto & rC = rorb[iblock];
        const auto & ro = rocc[iblock];
        // Compute projection
        Matrix<Torb> Pl(lC*lo.asDiagonal()*lC.adjoint());
        Matrix<Torb> Pr(rC*ro.asDiagonal()*rC.adjoint());
        ovl += std::real((Pl*Pr).trace());
      }
      return ovl;
    }

    bool attempt_extrapolation(const Vector<Tbase> & weights, bool density=false) {
      // Get the extrapolated Fock matrix
      if(not density) {
        auto fock(extrapolate_fock(weights));
        return attempt_fock(fock);
      } else {
        auto dm(extrapolate_density(weights));
        return add_entry(std::make_pair(dm.first, dm.second));
      }
    }

    bool attempt_fock(const FockMatrix<Torb> & fock) {
      // Diagonalize the Fock matrix
      auto diagonalized_fock = compute_orbitals(fock);
      auto new_orbitals = diagonalized_fock.first;
      auto new_orbital_energies = diagonalized_fock.second;

      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      // Try out the new occupations
      return add_entry(std::make_pair(new_orbitals, new_occupations));
    }

    std::pair<Vector<Tbase>, Tbase> solve_polytope_qp_(
        const Matrix<Tbase> & H,
        const Vector<Tbase> & g,
        Tbase E_orig,
        const std::vector<size_t> & particle_off,
        const std::vector<size_t> & particle_len) const {
      const size_t npars = g.size();
      const size_t nparts = particle_off.size();
      const Tbase eps = std::numeric_limits<Tbase>::epsilon();
      const Tbase tol = 100 * eps;

      auto model_value = [&](const Vector<Tbase> & lam) {
        return E_orig + g.dot(lam)
                      + Tbase(0.5) * (lam.transpose() * H * lam).value();
      };

      // Per-axis -> particle map.
      std::vector<size_t> particle_of(npars, nparts);
      for(size_t p = 0; p < nparts; p++)
        for(size_t i = particle_off[p]; i < particle_off[p] + particle_len[p]; i++)
          particle_of[i] = p;

      // Initial state: lam = 0, all non-neg constraints active, no
      // sum-caps active. Iteratively drop / add constraints until KKT.
      Vector<Tbase> lam = Vector<Tbase>::Zero(npars);
      std::vector<bool> at_zero(npars, true);
      std::vector<bool> sum_active(nparts, false);
      Vector<Tbase> nu = Vector<Tbase>::Zero(nparts);  // sum-cap multipliers

      const int max_iter = 8 * int(npars + nparts) + 16;
      for(int iter = 0; iter < max_iter; iter++) {
        std::vector<size_t> free_axes;
        for(size_t i = 0; i < npars; i++)
          if(!at_zero[i]) free_axes.push_back(i);
        const size_t n_free = free_axes.size();
        std::vector<size_t> active_sum_p;
        for(size_t p = 0; p < nparts; p++)
          if(sum_active[p]) active_sum_p.push_back(p);
        const size_t n_eq = active_sum_p.size();

        Vector<Tbase> p_step = Vector<Tbase>::Zero(npars);
        nu.setZero();
        bool solved = true;

        if(n_free > 0) {
          Matrix<Tbase> H_red(n_free, n_free);
          for(size_t r = 0; r < n_free; r++)
            for(size_t c = 0; c < n_free; c++)
              H_red(r, c) = H(free_axes[r], free_axes[c]);
          Vector<Tbase> g_red(n_free);
          Vector<Tbase> lam_red(n_free);
          for(size_t k = 0; k < n_free; k++) {
            g_red(k) = g(free_axes[k]);
            lam_red(k) = lam(free_axes[k]);
          }
          // Gradient of objective at lam, restricted to free axes
          // (at-zero axes contribute nothing since lam_i = 0 there).
          Vector<Tbase> grad_free = H_red * lam_red + g_red;

          Vector<Tbase> step_free;
          if(n_eq == 0) {
            try {
              step_free = H_red.colPivHouseholderQr().solve(-grad_free);
            } catch(...) { solved = false; }
            if(solved && !step_free.allFinite()) solved = false;
          } else {
            // Map each particle p with active sum-cap to its free-axis
            // local positions (rows of the constraint matrix).
            std::vector<std::vector<size_t>> particle_free(nparts);
            for(size_t k = 0; k < n_free; k++)
              particle_free[particle_of[free_axes[k]]].push_back(k);

            Matrix<Tbase> KKT = Matrix<Tbase>::Zero(n_free + n_eq, n_free + n_eq);
            Vector<Tbase> rhs = Vector<Tbase>::Zero(n_free + n_eq);
            KKT.block(0, 0, n_free, n_free) = H_red;
            rhs.segment(0, n_free) = -grad_free;
            for(size_t c = 0; c < n_eq; c++) {
              for(size_t local : particle_free[active_sum_p[c]]) {
                KKT(n_free + c, local) = 1;
                KKT(local, n_free + c) = 1;
              }
              // Step must preserve the active sum-cap equality, so
              // RHS for this constraint is 0 (not 1).
            }
            Vector<Tbase> sol;
            try {
              sol = KKT.colPivHouseholderQr().solve(rhs);
            } catch(...) { solved = false; }
            if(solved && !sol.allFinite()) solved = false;
            if(solved) {
              step_free = sol.segment(0, n_free);
              for(size_t c = 0; c < n_eq; c++)
                nu(active_sum_p[c]) = sol(n_free + c);
            }
          }
          if(solved)
            for(size_t k = 0; k < n_free; k++)
              p_step(free_axes[k]) = step_free(k);
        }

        if(!solved) break;  // singular KKT; keep best lam so far

        if(p_step.template lpNorm<Eigen::Infinity>() < tol) {
          // Step is zero -> we're at the QP optimum on the current
          // working set. Check Lagrange multipliers; drop the most-
          // negative active inequality, or stop if all multipliers are
          // non-negative.
          Vector<Tbase> grad_at_lam = g + H * lam;
          int    worst_kind = 0;   // 0 none, 1 non-neg axis, 2 sum-cap
          size_t worst_axis = npars;
          size_t worst_p    = nparts;
          Tbase  worst_val  = -tol;
          for(size_t i = 0; i < npars; i++) {
            if(!at_zero[i]) continue;
            Tbase mu = grad_at_lam(i)
                     + (sum_active[particle_of[i]]
                          ? nu(particle_of[i]) : Tbase(0));
            if(mu < worst_val) { worst_val = mu; worst_axis = i; worst_kind = 1; }
          }
          for(size_t p = 0; p < nparts; p++) {
            if(!sum_active[p]) continue;
            if(nu(p) < worst_val) { worst_val = nu(p); worst_p = p; worst_kind = 2; }
          }
          if(worst_kind == 0) break;  // KKT satisfied
          if(worst_kind == 1) at_zero[worst_axis] = false;
          else                sum_active[worst_p] = false;
          continue;
        }

        // Take the longest step alpha in [0, 1] that keeps lam feasible.
        Tbase alpha_max = std::numeric_limits<Tbase>::infinity();
        int    block_kind = 0;
        size_t block_axis = npars;
        size_t block_p    = nparts;
        for(size_t i = 0; i < npars; i++) {
          if(at_zero[i]) continue;
          if(p_step(i) < -tol) {
            Tbase a = -lam(i) / p_step(i);
            if(a < alpha_max) {
              alpha_max = a; block_axis = i; block_kind = 1;
            }
          }
        }
        for(size_t p = 0; p < nparts; p++) {
          if(sum_active[p]) continue;
          Tbase sum_lam = 0, sum_step = 0;
          for(size_t i = particle_off[p]; i < particle_off[p] + particle_len[p]; i++) {
            sum_lam += lam(i);
            sum_step += p_step(i);
          }
          if(sum_step > tol) {
            Tbase a = (Tbase(1) - sum_lam) / sum_step;
            if(a < alpha_max) {
              alpha_max = a; block_p = p; block_kind = 2;
            }
          }
        }

        Tbase alpha = std::min(alpha_max, Tbase(1));
        if(alpha < 0) alpha = 0;
        lam += alpha * p_step;
        for(size_t i = 0; i < npars; i++)
          if(lam(i) < 0 && lam(i) > -tol) lam(i) = 0;
        if(alpha < Tbase(1) - tol && block_kind != 0) {
          if(block_kind == 1) at_zero[block_axis] = true;
          else                sum_active[block_p] = true;
        }
      }
      return std::make_pair(lam, model_value(lam));
    }

    bool optimal_damping_step() {
      auto particles_left = [](Tbase n) {
        return n >= 10*std::numeric_limits<Tbase>::epsilon();
      };

      auto reference_orbitals = get_orbitals();
      auto reference_occupations = get_orbital_occupations();
      auto reference_fock = get_fock_matrix();
      auto reference_energy = get_energy();

      // Roothaan step: diagonalize current Fock matrix
      auto diagonalized_fock = compute_orbitals(reference_fock);
      auto new_orbitals = diagonalized_fock.first;
      auto new_orbital_energies = diagonalized_fock.second;

      // Skeleton occupations per particle type: [iparticle][itrial][iblock_within_particle]
      std::vector<std::vector<std::vector<Vector<Tbase>>>> trial_occupations_per_particle(number_of_blocks_per_particle_type_.size());

      for(Index iparticle=0; iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
        size_t iblock_start = particle_block_offset(iparticle);
        size_t nblocks_iparticle = number_of_blocks_per_particle_type_(iparticle);

        std::vector<Vector<Tbase>> particle_occupations(nblocks_iparticle);
        for(size_t iblock=0; iblock<nblocks_iparticle; iblock++)
          particle_occupations[iblock] = Vector<Tbase>::Zero(new_orbital_energies[iblock_start+iblock].size());
        IndexVector orbital_index = IndexVector::Zero(nblocks_iparticle);

        Tbase num_left = number_of_particles_(iparticle);
        auto all_energies = order_orbitals_by_energy(new_orbital_energies, iparticle);

        size_t ifill=0;
        while(particles_left(num_left)) {
          auto ienergy = std::get<0>(all_energies[ifill]);

          // Find end of this degenerate group
          size_t jfill;
          for(jfill=ifill+1; jfill < all_energies.size(); jfill++) {
            auto jenergy = std::get<0>(all_energies[jfill]);
            if(std::abs(ienergy-jenergy) > optimal_damping_degeneracy_threshold_)
              break;
          }

          // Total capacity of the degenerate group
          Tbase maximum_capacity = 0.0;
          for(size_t iorb=ifill; iorb<jfill; iorb++)
            maximum_capacity += maximum_occupation_(std::get<1>(all_energies[iorb]));

          if(num_left >= maximum_capacity or jfill-ifill==1) {
            // Group is fully filled or only one orbital — single skeleton
            for(size_t iorb=ifill; iorb<jfill; iorb++) {
              auto block_index = std::get<1>(all_energies[iorb]);
              auto capacity = maximum_occupation_(block_index);
              auto fill = std::min(capacity, num_left);
              particle_occupations[block_index-iblock_start](orbital_index(block_index-iblock_start)++) = fill;
              num_left -= fill;
              if(not particles_left(num_left))
                break;
            }
            if(not particles_left(num_left))
              trial_occupations_per_particle[iparticle].push_back(particle_occupations);
          } else {
            if(verbosity_>=5) {
              printf("Degenerate orbitals: iblock iorb E\n");
              for(size_t iorb=ifill; iorb<jfill; iorb++)
                printf("%s %3i % .9f\n",
                       block_descriptions_[std::get<1>(all_energies[iorb])].c_str(),
                       (int) std::get<2>(all_energies[iorb]),
                       std::get<0>(all_energies[iorb]));
            }

            // Enumerate the extremal vertices of the integer-filling
            // polytope
            //     V = { n in R^N : 0 <= n_k <= c_k, sum_k n_k = num_left }
            // where the N=jfill-ifill orbitals of the degenerate group
            // are indexed flat across whichever blocks they live in.
            // Each vertex of V has at most one fractional component:
            // a subset S of the N indices is fully filled (n_k = c_k)
            // and at most one residual index j carries num_left -
            // sum_{k in S} c_k. This enumerates every distinct integer
            // filling regardless of whether the group spans one block
            // (intra-block accidental degeneracy in a no-symmetry run)
            // or several (cross-block crossings such as 4s/3d in
            // atoms). The dedup check below collapses gauge-equivalent
            // skeletons in symmetric cases.
            struct GroupOrb {
              size_t local_block;     // block_index - iblock_start
              size_t slot_offset;     // position among this block's group orbitals
              Tbase capacity;
            };
            std::vector<GroupOrb> group_orbs;
            std::map<size_t, size_t> next_offset_per_block;
            for(size_t iorb_idx=ifill; iorb_idx<jfill; iorb_idx++) {
              auto block_index = std::get<1>(all_energies[iorb_idx]);
              size_t local_block = block_index - iblock_start;
              size_t offset = next_offset_per_block[local_block]++;
              group_orbs.push_back({local_block, offset,
                                    maximum_occupation_(block_index)});
            }
            const size_t N_group = group_orbs.size();
            if(N_group > 8 * sizeof(size_t) - 1)
              throw std::logic_error("Degenerate group too large for subset enumeration; raise the degeneracy threshold or split the group.\n");

            auto try_emit = [&](const std::vector<Tbase> & fills) {
              auto iter_occupations = particle_occupations;
              for(size_t k=0; k<N_group; k++) {
                const auto & g = group_orbs[k];
                iter_occupations[g.local_block](
                  orbital_index(g.local_block) + g.slot_offset) = fills[k];
              }
              auto match = [this, &iter_occupations](const std::vector<Vector<Tbase>> & list_occ) {
                Tbase sqdiff=0.0;
                for(size_t iblock=0; iblock<list_occ.size(); iblock++)
                  sqdiff += (list_occ[iblock]-iter_occupations[iblock]).norm();
                return sqdiff < occupation_change_threshold_;
              };
              auto idx = std::find_if(trial_occupations_per_particle[iparticle].begin(),
                                      trial_occupations_per_particle[iparticle].end(),
                                      match);
              if(idx == trial_occupations_per_particle[iparticle].end())
                trial_occupations_per_particle[iparticle].push_back(iter_occupations);
            };

            const Tbase tol = occupation_change_threshold_;
            const size_t n_subsets = size_t(1) << N_group;
            for(size_t mask=0; mask<n_subsets; mask++) {
              Tbase filled = 0;
              for(size_t k=0; k<N_group; k++)
                if(mask & (size_t(1) << k))
                  filled += group_orbs[k].capacity;
              Tbase residual = num_left - filled;
              if(residual < -tol)
                continue;  // S overfills

              std::vector<Tbase> fills(N_group, Tbase(0));
              for(size_t k=0; k<N_group; k++)
                if(mask & (size_t(1) << k))
                  fills[k] = group_orbs[k].capacity;

              if(std::abs(residual) < tol) {
                try_emit(fills);
              } else {
                // residual > 0: pick one orbital not in S to take it.
                // residual >= c_k is equivalent to flipping k into S,
                // which is enumerated by a different mask, so skip it
                // here to avoid trivial duplicates.
                for(size_t k=0; k<N_group; k++) {
                  if(mask & (size_t(1) << k))
                    continue;
                  Tbase cap_k = group_orbs[k].capacity;
                  if(residual < cap_k - tol) {
                    auto fills_with_frac = fills;
                    fills_with_frac[k] = residual;
                    try_emit(fills_with_frac);
                  }
                }
              }
            }

            num_left = 0.0;
          }

          ifill = jfill;
        }
      }

      size_t npars = 0;
      for(auto & trial: trial_occupations_per_particle)
        npars += trial.size();
      // Record the polytope dimension so the outer SCF state machine
      // can size its post-ODA orbital-rotation burst when the user has not overridden
      // orbital_rotation_steps_after_oda_.
      last_polytope_dimension_ = npars;
      if(verbosity_>=5) {
        printf("%i parameters in optimal damping\n", (int) npars);
        fflush(stdout);
      }
      if(npars==0)
        return false;

      // Build mixed density from a parameter vector lambda. Per
      // particle type, the lambda subvector has one entry per skeleton;
      // the residual (1 - sum) goes to the reference density.
      auto interpolate_density = [this, &reference_orbitals, &reference_occupations, &new_orbitals, &trial_occupations_per_particle](const Vector<Tbase> & lambda) {
        Orbitals<Torb> interp_orbs(reference_orbitals.size());
        OrbitalOccupations<Tbase> interp_occs(reference_orbitals.size());

        size_t iparam=0;
        for(Index iparticle=0; iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
          size_t ntrial = trial_occupations_per_particle[iparticle].size();
          if(ntrial==0)
            continue;
          Tbase lambda_sum = lambda.segment(iparam, ntrial).sum();

          for(size_t iblock_particle = 0; iblock_particle < (size_t)number_of_blocks_per_particle_type_(iparticle); iblock_particle++) {
            size_t iblock = iblock_particle + particle_block_offset(iparticle);
            if(empty_block(iblock))
              continue;

            Matrix<Torb> old_dm = reference_orbitals[iblock] * reference_occupations[iblock].asDiagonal() * reference_orbitals[iblock].adjoint();

            Vector<Tbase> new_occ = lambda(iparam)*trial_occupations_per_particle[iparticle][0][iblock_particle];
            for(size_t itrial=1; itrial<ntrial; itrial++)
              new_occ += lambda(iparam+itrial)*trial_occupations_per_particle[iparticle][itrial][iblock_particle];

            Matrix<Torb> new_dm = new_orbitals[iblock] * new_occ.asDiagonal() * new_orbitals[iblock].adjoint();
            Matrix<Torb> mix_dm = (1.0 - lambda_sum)*old_dm + new_dm;

            mix_dm *= -1.0;
            Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(mix_dm);
            interp_occs[iblock] = es.eigenvalues();
            interp_orbs[iblock] = es.eigenvectors();
            interp_occs[iblock] *= Tbase{-1};

            const Tbase zero_tol = 10*maximum_occupation_(iblock)*std::numeric_limits<Tbase>::epsilon();
            for(Index k=0; k<interp_occs[iblock].size(); k++) {
              if(std::abs(interp_occs[iblock](k)) <= zero_tol)
                interp_occs[iblock](k) = Tbase{0};
            }

            if(interp_occs[iblock].minCoeff() < -100*std::numeric_limits<Tbase>::epsilon()) {
              std::ostringstream oss;
              oss << "Negative natural occupation numbers in block " << iblock << "!\n" << interp_occs[iblock];
              throw std::logic_error(oss.str());
            }
          }
          iparam += ntrial;
        }
        if(iparam != (size_t)lambda.size())
          throw std::logic_error("Indexing inconsistency in optimal_damping_step\n");

        return std::make_pair(interp_orbs, interp_occs);
      };

      auto evaluate = [this, &interpolate_density](const Vector<Tbase> & lambda) {
        auto dm = interpolate_density(lambda);
        auto fock = fock_builder_(dm);
        return std::make_pair(dm, fock);
      };

      // Compute tr(fock * (P(dm1) - P(dm2))) where P is built from orbitals * diag(occ) * orbitals^H
      auto trace_diff = [this](const DensityMatrix<Torb,Tbase> & dm1, const DensityMatrix<Torb,Tbase> & dm2, const FockMatrix<Torb> & fock) {
        const auto & orbitals1 = dm1.first;
        const auto & occupations1 = dm1.second;
        const auto & orbitals2 = dm2.first;
        const auto & occupations2 = dm2.second;

        Tbase tr=0.0;
        for(size_t iblock=0; iblock<fock.size(); iblock++) {
          if(empty_block(iblock))
            continue;
          Matrix<Torb> dD = orbitals1[iblock] * occupations1[iblock].asDiagonal() * orbitals1[iblock].adjoint()
                          - orbitals2[iblock] * occupations2[iblock].asDiagonal() * orbitals2[iblock].adjoint();
          tr += std::real((dD*fock[iblock]).trace());
        }
        return tr;
      };

      Vector<Tbase> x0 = Vector<Tbase>::Zero(npars);
      const Tbase E_orig = reference_energy;
      const auto & F_orig = reference_fock;
      const DensityMatrix<Torb,Tbase> P_orig = std::make_pair(reference_orbitals, reference_occupations);

      // Evaluate each canonical vertex (one lambda_i = 1, others 0).
      // The axis-vertex densities all share the same orbitals
      // (new_orbitals) and differ only in their occupation vectors,
      // so they go through the batched Fock-builder helper which the
      // caller can override to amortise integral / grid setup.
      std::vector<DensityMatrix<Torb,Tbase>> axis_densities(npars);
      for(size_t idim=0; idim<npars; idim++) {
        x0.setZero();
        x0(idim) = 1.0;
        axis_densities[idim] = interpolate_density(x0);
      }
      auto axis_fock = evaluate_batch_(axis_densities);

      std::vector<std::pair<DensityMatrix<Torb,Tbase>,FockBuilderReturn<Torb,Tbase>>> evaluations(npars);
      for(size_t idim=0; idim<npars; idim++) {
        evaluations[idim] = std::make_pair(std::move(axis_densities[idim]),
                                           std::move(axis_fock[idim]));
        if(verbosity_>=5) {
          printf("Roothaan step in dimension %i yields energy % .10f change %e\n",
                 (int) idim, evaluations[idim].second.first, evaluations[idim].second.first - E_orig);
        }
      }

      // Build a second-order Taylor model of the energy on the product
      // of per-particle simplices around lambda = 0:
      //   E(lambda) ~= E_orig + g^T lambda + 0.5 lambda^T H lambda.
      // Gradient: g_i = tr(F_orig * (P_i - P_orig)).
      // Diagonal Hessian: H_ii = 2*(E_i - E_orig - g_i), the Hermite
      //   quadratic fit through (0, E_orig, g_i) and (1, E_i).
      // Off-diagonal Hessian: H_ij ~= tr((F_j - F_orig) * (P_i - P_orig)),
      //   exact when the energy is quadratic in P (Hartree-Fock) and a
      //   second-order finite difference otherwise; symmetrized over (i,j).
      // No additional Fock evaluations beyond the npars axis vertices.
      Vector<Tbase> grad(npars);
      for(size_t i=0; i<npars; i++)
        grad(i) = trace_diff(evaluations[i].first, P_orig, F_orig);
      Matrix<Tbase> hess(npars, npars);
      for(size_t i=0; i<npars; i++) {
        Tbase E_i = evaluations[i].second.first;
        hess(i, i) = 2*(E_i - E_orig - grad(i));
        for(size_t j=i+1; j<npars; j++) {
          const auto & F_j = evaluations[j].second.second;
          const auto & F_i = evaluations[i].second.second;
          Tbase from_j = trace_diff(evaluations[i].first, P_orig, F_j) - grad(i);
          Tbase from_i = trace_diff(evaluations[j].first, P_orig, F_i) - grad(j);
          hess(i, j) = 0.5*(from_j + from_i);
          hess(j, i) = hess(i, j);
        }
      }

      // Particle layout for the polytope: each active particle's lambda
      // sub-vector lives on its own simplex {lambda >= 0, sum(lambda) <= 1}.
      std::vector<size_t> particle_off, particle_len;
      {
        size_t off = 0;
        for(Index p=0; p<number_of_blocks_per_particle_type_.size(); p++) {
          size_t nt = trial_occupations_per_particle[p].size();
          if(nt > 0) {
            particle_off.push_back(off);
            particle_len.push_back(nt);
          }
          off += nt;
        }
      }
      // Minimise the quadratic model on the product-of-simplices
      // polytope via the active-set QP solver. The previous code
      // enumerated every face of the polytope (product over particles
      // of 2^(n_p+1)-1 faces), which is intractable when degenerate
      // groups span several blocks and produce npars in the tens.
      const Tbase eps = std::numeric_limits<Tbase>::epsilon();
      auto model_value = [&](const Vector<Tbase> & lam) {
        return E_orig + grad.dot(lam) + 0.5*(lam.transpose()*hess*lam).value();
      };
      (void) model_value;
      Vector<Tbase> lam_opt;
      Tbase model_min;
      std::tie(lam_opt, model_min) = solve_polytope_qp_(
          hess, grad, E_orig, particle_off, particle_len);
      if(verbosity_>=5) {
        printf("Quadratic model minimum at lambda = (");
        for(Index i=0; i<lam_opt.size(); i++)
          printf("%s%g", i ? "," : "", lam_opt(i));
        printf("), model energy change %e\n", model_min - E_orig);
      }

      // Per-axis cubic-fit secondary candidates: with E_orig, slope at
      // lambda=0, E_axis, and slope at lambda=1 we have four data points,
      // enough to fit a cubic along each axis. Roots of the cubic
      // derivative inside (0,1) capture 1D minima the quadratic Taylor
      // model misses when the energy is non-quadratic in lambda along
      // that direction (typical of DFT near convergence).
      std::vector<std::pair<Vector<Tbase>, std::string>> candidates;
      if(lam_opt.template lpNorm<Eigen::Infinity>() > 100*eps)
        candidates.emplace_back(lam_opt, "model min");
      for(size_t i=0; i<npars; i++) {
        Tbase E_i = evaluations[i].second.first;
        Tbase g_i = grad(i);
        Tbase slope_at_1 = trace_diff(evaluations[i].first, P_orig, evaluations[i].second.second);
        try {
          auto cubic = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(E_orig, g_i, Tbase(1), E_i, slope_at_1);
          auto zeros = std::apply(HelperRoutines::cubic_polynomial_zeros<Tbase>, cubic);
          for(Tbase z : {zeros.first, zeros.second}) {
            if(z > 100*eps && z < Tbase(1) - 100*eps) {
              Vector<Tbase> xc = Vector<Tbase>::Zero(npars);
              xc(i) = z;
              candidates.emplace_back(xc, std::string("cubic axis ") + std::to_string(i));
            }
          }
        } catch(std::logic_error &) {}
      }
      // Pair-diagonal cubics: along each edge from vertex e_i to vertex
      // e_j we have four data points (E_i, slope at lambda=e_i in
      // direction e_j-e_i, E_j, slope at lambda=e_j in same direction),
      // enough to fit a cubic. Roots of its derivative inside (0,1)
      // give 1D minima on the edge.
      for(size_t i=0; i<npars; i++) {
        for(size_t j=i+1; j<npars; j++) {
          const auto & P_i = evaluations[i].first;
          const auto & P_j = evaluations[j].first;
          const auto & F_i = evaluations[i].second.second;
          const auto & F_j = evaluations[j].second.second;
          Tbase E_i = evaluations[i].second.first;
          Tbase E_j = evaluations[j].second.first;
          Tbase slope_i = trace_diff(P_j, P_i, F_i);
          Tbase slope_j = trace_diff(P_j, P_i, F_j);
          try {
            auto cubic = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(E_i, slope_i, Tbase(1), E_j, slope_j);
            auto zeros = std::apply(HelperRoutines::cubic_polynomial_zeros<Tbase>, cubic);
            for(Tbase z : {zeros.first, zeros.second}) {
              if(z > 100*eps && z < Tbase(1) - 100*eps) {
                Vector<Tbase> xc = Vector<Tbase>::Zero(npars);
                xc(i) = Tbase(1) - z;
                xc(j) = z;
                candidates.emplace_back(xc, std::string("cubic edge ") + std::to_string(i) + "-" + std::to_string(j));
              }
            }
          } catch(std::logic_error &) {}
        }
      }

      if(verbosity_ >= 5) {
        size_t n_model = 0, n_axis = 0, n_edge = 0;
        for(const auto & cand : candidates) {
          if(cand.second == "model min") n_model++;
          else if(cand.second.rfind("cubic axis", 0) == 0) n_axis++;
          else if(cand.second.rfind("cubic edge", 0) == 0) n_edge++;
        }
        printf("Trial loop: %zu candidates (%zu quadratic-model + "
               "%zu cubic-axis + %zu cubic-edge); each evaluates one "
               "Fock build per scale until a descent step is found.\n",
               candidates.size(), n_model, n_axis, n_edge);
      }

      // Trial loop: at each backoff scale, evaluate candidates in order
      // and stop at the first one that strictly decreases the energy.
      // The quadratic-model / cubic-axis / cubic-edge candidates often
      // overlap (especially for npars == 1, where the quadratic and
      // cubic-axis candidates are competing 1D fits on the same axis),
      // so evaluating every candidate at every scale pays for
      // redundant Fock builds without changing the accepted descent
      // step. Densities from the rejected candidates evaluated before
      // the first success still enter the orbital history (via
      // add_entry) and seed DIIS in subsequent iterations; we just
      // stop adding more once we have a descent.
      bool succ = false;
      for(int scalefac=0; scalefac<=5; scalefac++) {
        Tbase scale = std::pow(Tbase(2), -scalefac);
        for(const auto & cand : candidates) {
          Vector<Tbase> x_scaled = scale * cand.first;
          auto eval = evaluate(x_scaled);
          number_of_fock_evaluations_++;
          bool ok = add_entry(eval.first, eval.second);
          if(ok) succ = true;
          if(verbosity_>=10) {
            printf("ODA %s at scale %g gives E = % .10f, change %e%s\n",
                   cand.second.c_str(), scale, eval.second.first, eval.second.first - E_orig,
                   ok ? " (accepted)" : "");
          }
          if(ok) break;
        }
        if(succ) break;
      }
      if(succ) {
        // ODA has globally rearranged the orbital basis (and possibly
        // the occupation pattern); the recorded PR+ CG state is no
        // longer tied to the current iterate. The lazy L-BFGS state
        // is also released if it was allocated; clear_lbfgs_state_()
        // is a no-op when L-BFGS has not been used.
        // The end-of-iteration cleanup() in run() runs the density-
        // matrix-difference pruning; doing it here too would print the
        // "Density matrix difference ..." line twice per ODA iteration.
        previous_orbital_gradient_.resize(0);
        previous_orbital_direction_.resize(0);
        previous_orbital_dofs_.clear();
        clear_lbfgs_state_();
      }
      // Update the active-rotation count seen at the new iterate so the
      // outer state machine can size its orbital-rotation burst from it.
      last_active_rotation_count_ = compute_active_rotation_count();
      return succ;
    }

    void cleanup() {
      Vector<Tbase> density_differences = Vector<Tbase>::Zero(orbital_history_.size()-1);
      for(size_t ihist=1;ihist<orbital_history_.size();ihist++) {
        density_differences(ihist-1)=density_matrix_difference(ihist, 0);
      }
      if(verbosity_ >= 10) {
        std::cout << "Density differences: " << density_differences.transpose() << std::endl;
      } else if(verbosity_>=5) {
        printf("Density matrix difference %e between lowest-energy and newest entry\n",density_differences(0));
      }

      // Sort the differences
      IndexVector idx(sort_index_ascending(density_differences));
      // Pick the indices that don't satisfy the criterion
      Tbase ref_diff = density_differences(idx(0));
      IndexVector sub_idx = find_indices_where(idx, [&](Index k){
        return density_restart_factor_*density_differences(k) > ref_diff;
      });
      if(sub_idx.size()) {
        IndexVector filtered_idx(sub_idx.size());
        for(Index k=0;k<sub_idx.size();k++)
          filtered_idx(k) = idx(sub_idx(k));
        // Sort descending
        std::sort(filtered_idx.data(), filtered_idx.data()+filtered_idx.size(), std::greater<Index>());
        if(verbosity_>=10)
          printf("Removing %i entries corresponding to large change in density matrix\n",(int) filtered_idx.size());
        for(Index k=0;k<filtered_idx.size();k++) {
          Index ihistm1 = filtered_idx(k);
          // Remember the off-by-one in the indices
          orbital_history_.erase(orbital_history_.begin()+ihistm1+1);
        }
      }
    }

    std::vector<OrbitalRotation> degrees_of_freedom() const {
      std::vector<OrbitalRotation> dofs;
      // Reference calculation
      const auto reference_occupations = get_orbital_occupations();

      // List occupied-occupied rotations, in case some orbitals are not fully occupied
      for(size_t iblock = 0; iblock < reference_occupations.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        IndexVector occupied_indices = find_indices_where(reference_occupations[iblock], [](Tbase v){ return v > 0.0; });
        for(Index io1 = 0; io1 < occupied_indices.size(); io1++)
          for(Index io2 = 0; io2 < io1; io2++) {
            auto o1 = occupied_indices[io1];
            auto o2 = occupied_indices[io2];
            if(reference_occupations[iblock][o1] != reference_occupations[iblock][o2])
              dofs.push_back(std::make_tuple(iblock, o1, o2));
          }
      }

      // List occupied-virtual rotations
      for(size_t iblock = 0; iblock < reference_occupations.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        // Find the occupied and virtual blocks
        IndexVector occupied_indices = find_indices_where(reference_occupations[iblock], [](Tbase v){ return v > 0.0; });
        IndexVector virtual_indices = find_indices_where(reference_occupations[iblock], [](Tbase v){ return v == 0.0; });
        for(Index oi=0; oi<occupied_indices.size(); oi++)
          for(Index vi=0; vi<virtual_indices.size(); vi++)
            dofs.push_back(std::make_tuple(iblock, occupied_indices[oi], virtual_indices[vi]));
      }

      return dofs;
    }

    Vector<Tbase> orbital_gradient_vector() const {
      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      Vector<Tbase> orb_grad;

      if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
        orb_grad = Vector<Tbase>::Zero(dof_list.size());
      } else {
        orb_grad = Vector<Tbase>::Zero(2*dof_list.size());
      }

      // Extract the orbital gradient
      for(size_t idof = 0; idof < dof_list.size(); idof++) {
        auto dof(dof_list[idof]);
        auto iblock = std::get<0>(dof);
        auto iorb = std::get<1>(dof);
        auto jorb = std::get<2>(dof);
        auto fock_block = get_fock_matrix_block(0, iblock);
        auto orbital_block = get_orbital_block(0, iblock);
        auto occ_block = get_orbital_occupation_block(0, iblock);

        Matrix<Torb> fock_mo = orbital_block.adjoint() * fock_block * orbital_block;
        orb_grad(idof) = 2*std::real(fock_mo(iorb,jorb))*(occ_block(jorb)-occ_block(iorb));
        if constexpr (Eigen::NumTraits<Torb>::IsComplex) {
          orb_grad(dof_list.size() + idof) = 2*std::imag(fock_mo(iorb,jorb))*(occ_block(jorb)-occ_block(iorb));
        }
      }

      if(has_nan(orb_grad))
        throw std::logic_error("Orbital gradient has NaNs");

      return orb_grad;
    }

    Vector<Tbase> diagonal_orbital_hessian() const {
      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      Vector<Tbase> orb_hess;

      if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
        orb_hess = Vector<Tbase>::Zero(dof_list.size());
      } else {
        orb_hess = Vector<Tbase>::Zero(2*dof_list.size());
      }

      // Extract the orbital hessient
      for(size_t idof = 0; idof < dof_list.size(); idof++) {
        auto dof(dof_list[idof]);
        auto iblock = std::get<0>(dof);
        auto iorb = std::get<1>(dof);
        auto jorb = std::get<2>(dof);
        auto fock_block = get_fock_matrix_block(0, iblock);
        auto orbital_block = get_orbital_block(0, iblock);
        auto occ_block = get_orbital_occupation_block(0, iblock);

        Matrix<Torb> fock_mo = orbital_block.adjoint() * fock_block * orbital_block;
        orb_hess(idof) = 2*std::real((fock_mo(iorb,iorb)-fock_mo(jorb,jorb))*(occ_block(jorb)-occ_block(iorb)));
        if constexpr (Eigen::NumTraits<Torb>::IsComplex) {
          orb_hess(dof_list.size() + idof) = orb_hess(idof);
        }
      }
      return orb_hess;
    }

    Vector<Tbase> precondition_search_direction(const Vector<Tbase> & gradient, const Vector<Tbase> & diagonal_hessian, Tbase shift=0.1) const {
      if(gradient.size() != diagonal_hessian.size())
        throw std::logic_error("precondition_search_direction: gradient and diagonal hessian have different size!\n");

      // Build positive definite diagonal Hessian
      Vector<Tbase> positive_hessian(diagonal_hessian);
      positive_hessian += (-diagonal_hessian.minCoeff()+shift)*Vector<Tbase>::Ones(positive_hessian.size());

      Tbase normalized_projection;
      Tbase maximum_spread = positive_hessian.maxCoeff();
      Vector<Tbase> preconditioned_direction;
      while(true) {
        // Normalize the largest values
        Vector<Tbase> normalized_hessian(positive_hessian);
        for(Index k=0;k<normalized_hessian.size();k++)
          if(normalized_hessian(k) > maximum_spread)
            normalized_hessian(k) = maximum_spread;

        // and divide the gradient by its square root
        preconditioned_direction = gradient.array()/normalized_hessian.array().sqrt();
        if(has_nan(preconditioned_direction))
          throw std::logic_error("Preconditioned search direction has NaNs");

        normalized_projection = preconditioned_direction.dot(gradient) / std::sqrt(preconditioned_direction.norm()*gradient.norm());
        if(normalized_projection >= minimal_gradient_projection_) {
          return preconditioned_direction;
        } else {
          if(verbosity_>=5) {
            printf("Warning - projection of preconditioned search direction on negative gradient %e is too small, decreasing spread of Hessian values from %e by factor 10\n",normalized_projection,maximum_spread);
          }
          maximum_spread /= 10;
        }
      }
    }

    Orbitals<Torb> form_rotation_matrices(const Vector<Tbase> & x) const {
      const Orbitals<Torb> reference_orbitals(get_orbitals());

      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      Vector<Tbase> orb_grad(dof_list.size());
      // Sort them by symmetry
      std::vector<std::vector<std::tuple<Index, Index, size_t>>> blocked_dof(reference_orbitals.size());
      for(size_t idof=0; idof<dof_list.size(); idof++) {
        auto dof = dof_list[idof];
        auto iblock = std::get<0>(dof);
        auto iorb = std::get<1>(dof);
        auto jorb = std::get<2>(dof);
        blocked_dof[iblock].push_back(std::make_tuple(iorb,jorb,idof));
      }

      // Form the rotation matrices
      Orbitals<Torb> kappa(reference_orbitals.size());
      for(size_t iblock=0; iblock < reference_orbitals.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        // Collect the rotation parameters
        kappa[iblock] = Matrix<Torb>::Zero(reference_orbitals[iblock].cols(), reference_orbitals[iblock].cols());
        for(auto dof: blocked_dof[iblock]) {
          auto iorb = std::get<0>(dof);
          auto jorb = std::get<1>(dof);
          auto idof = std::get<2>(dof);
          kappa[iblock](iorb,jorb) = x(idof);
        }
        // imaginary parameters
        if constexpr (Eigen::NumTraits<Torb>::IsComplex) {
          for(auto dof: blocked_dof[iblock]) {
            auto iorb = std::get<0>(dof);
            auto jorb = std::get<1>(dof);
            auto idof = std::get<2>(dof);
            kappa[iblock](iorb,jorb) += Torb(0.0,x(dof_list.size()+idof));
          }
        }
        // Antisymmetrize
        kappa[iblock] -= kappa[iblock].adjoint().eval();
      }

      return kappa;
    }

    Tbase maximum_rotation_step(const Vector<Tbase> & x) const {
      // Get the rotation matrices
      auto kappa(form_rotation_matrices(x));

      Tbase maximum_step = std::numeric_limits<Tbase>::max();
      for(size_t iblock=0; iblock < kappa.size(); iblock++) {
        if(kappa[iblock].size()==0)
          continue;
        Matrix<std::complex<Tbase>> kappa_imag =
            kappa[iblock].template cast<std::complex<Tbase>>() *
            std::complex<Tbase>(Tbase{0}, Tbase{-1});
        Eigen::SelfAdjointEigenSolver<Matrix<std::complex<Tbase>>> es(kappa_imag);
        Vector<Tbase> eval = es.eigenvalues();

        // Assume objective function is 4th order in orbitals
        Tbase block_maximum = Tbase(0.5*M_PI)/(eval.array().abs().maxCoeff());
        // The maximum allowed step is determined as the minimum of the block-wise steps
        maximum_step = std::min(maximum_step, block_maximum);
      }

      return maximum_step;
    }

    Orbitals<Torb> rotate_orbitals(const Vector<Tbase> & x) const {
      auto kappa(form_rotation_matrices(x));

      // Rotate the orbitals
      Orbitals<Torb> new_orbitals(get_orbitals());
      for(size_t iblock=0; iblock < new_orbitals.size(); iblock++) {
        if(empty_block(iblock))
          continue;

        // Exponentiated kappa
        Matrix<Torb> expkappa;

#if 0
        expkappa = expm_antihermitian(kappa[iblock]);
#else
        // Do eigendecomposition of -i*kappa (Hermitian) -> real evals,
        // complex evecs. Then exp(kappa) = evec * diag(exp(i*eval)) * evec^H.
        Matrix<std::complex<Tbase>> kappa_imag =
            kappa[iblock].template cast<std::complex<Tbase>>() *
            std::complex<Tbase>(Tbase{0}, Tbase{-1});
        Eigen::SelfAdjointEigenSolver<Matrix<std::complex<Tbase>>> es(kappa_imag);
        Vector<Tbase> eval = es.eigenvalues();
        Matrix<std::complex<Tbase>> evec = es.eigenvectors();
        Vector<std::complex<Tbase>> exp_diag(eval.size());
        for(Index k=0; k<eval.size(); ++k)
          exp_diag[k] = std::exp(std::complex<Tbase>(Tbase{0}, eval[k]));
        Matrix<std::complex<Tbase>> expkappa_imag = evec * exp_diag.asDiagonal() * evec.adjoint();
        if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
          expkappa = expkappa_imag.real();
        } else {
          expkappa = expkappa_imag;
        }
#endif

        // Do the rotation
        new_orbitals[iblock] = new_orbitals[iblock]*expkappa;
      }

      return new_orbitals;
    }
    OrbitalHistoryEntry<Torb, Tbase> make_history_entry(const DensityMatrix<Torb, Tbase> & density_matrix, const FockBuilderReturn<Torb, Tbase> & fock) const {
      static size_t index=0;
      return std::make_tuple(density_matrix, fock, index++);
    }
    OrbitalHistoryEntry<Torb, Tbase> evaluate_rotation(const Vector<Tbase> & x) {
      // Rotate orbitals
      auto new_orbitals(rotate_orbitals(x));
      // Compute the Fock matrix
      auto reference_occupations = get_orbital_occupations();

      auto density_matrix = std::make_pair(new_orbitals, reference_occupations);
      auto fock = fock_builder_(density_matrix);
      number_of_fock_evaluations_++;
      return make_history_entry(density_matrix, fock);
    }
    void level_shifting_step() {
      Tbase level_shift = initial_level_shift_;
      Tbase reference_energy = get_energy();
      size_t start_index = largest_index();

      if(verbosity_ >= 5)
        printf("Entering level shifting code, reference energy %e\n",reference_energy);

      // Get Fock matrix
      FockMatrix<Torb> fock = get_fock_matrix();
      // Form level shift matrix
      FockMatrix<Torb> shifted_fock;

      for(size_t ishift=0; ishift < 50; ishift++) {
        // Shift virtual orbitals up in energy. In practice, scale
        // the level shift by the fraction of unoccupied character,
        // so that SOMOs get half the shift
        shifted_fock = fock;
        for(size_t iblock=0; iblock<fock.size(); iblock++) {
          if(empty_block(iblock))
            continue;
          Vector<Tbase> fractional_occupations(get_orbital_occupation_block(0, iblock)/maximum_occupation_(iblock));
          fractional_occupations = Vector<Tbase>::Ones(fractional_occupations.size()) - fractional_occupations;
          Matrix<Torb> orbitals(get_orbital_block(0, iblock));

          shifted_fock[iblock] += level_shift *(orbitals * fractional_occupations.asDiagonal() * orbitals.adjoint());
        }

        // Add new Fock matrix
        attempt_fock(shifted_fock);
        Tbase best_energy = get_lowest_energy_after_index(start_index);
        if(verbosity_ >= 5)
          printf("Level shift iteration %i: shift %e energy change % e\n", ishift, level_shift, best_energy-reference_energy);

        if(best_energy > reference_energy) {
          // Energy did not decrease; increase level shift
          level_shift *= level_shift_factor_;
          continue;
        } else {
          return;
        }
      }
    }
    // --- Orbital-rotation step infrastructure ----------------------
    //
    // scaled_steepest_descent_step() (PR+ CG) and lbfgs_step()
    // (limited-memory BFGS, follow-up) both follow the same outline:
    //
    //   1. Pseudo-diagonalise F within equal-occupation sub-blocks to
    //      get a canonical orbital basis and orbital energies eps.
    //   2. Collect orbital-rotation degrees of freedom (b, i, j) with
    //      different occupations.
    //   3. Compute the gradient g_alpha and diagonal Hessian h_alpha.
    //   4. Build a descent direction d, possibly with curvature
    //      correction (PR+ CG or L-BFGS).
    //   5. Run a sigma-line-search along C(t) = C_pseudo * exp(t K).
    //
    // Steps 1-3 and the K, t_max, evaluate-at, slope-at-trial pieces
    // of step 5 are shared. The two algorithms differ only in how
    // step 4 builds the trial-1 direction (PR+ CG mix vs L-BFGS two-
    // loop recursion) and in what state they carry between calls.
    struct RotationStepContext {
      Orbitals<Torb> C_pseudo;
      OrbitalOccupations<Tbase> n_ref;
      std::vector<Vector<Tbase>> eps;
      std::vector<OrbitalRotation> dofs;
      Vector<Tbase> g;
      Vector<Tbase> h;
      Tbase E_ref = 0;
      size_t n_dof = 0;
      size_t n_par = 0;
      bool is_complex = false;
    };

    bool build_rotation_step_context_(RotationStepContext & ctx) const {
      if(orbital_history_.empty()) return false;

      ctx.n_ref = get_orbital_occupations();
      auto C_ref = get_orbitals();
      auto F_ref = get_fock_matrix();
      ctx.E_ref = get_energy();
      size_t nblocks = C_ref.size();
      ctx.is_complex = Eigen::NumTraits<Torb>::IsComplex;

      // Step 1: pseudo-diagonalise F within equal-occupation sub-blocks.
      pseudo_canonicalise_(C_ref, F_ref, ctx.n_ref, ctx.C_pseudo, ctx.eps);

      // Step 2: collect orbital-rotation degrees of freedom (i > j
      // pairs within the same block, with non-trivially different
      // occupations). Equal-occupation pairs are gauge directions
      // and excluded.
      std::vector<Matrix<Torb>> F_pseudo(nblocks);
      for(size_t b = 0; b < nblocks; b++) {
        if(empty_block(b) || ctx.C_pseudo[b].cols() == 0) {
          F_pseudo[b].resize(0, 0);
          continue;
        }
        F_pseudo[b] = ctx.C_pseudo[b].adjoint() * F_ref[b] * ctx.C_pseudo[b];
        Index n_b = ctx.C_pseudo[b].cols();
        for(Index i = 0; i < n_b; i++)
          for(Index j = 0; j < i; j++)
            if(std::abs(ctx.n_ref[b](i) - ctx.n_ref[b](j)) >= occupation_change_threshold_)
              ctx.dofs.emplace_back(b, i, j);
      }
      if(ctx.dofs.empty()) {
        if(verbosity_ >= 5)
          printf("Rotation step: no orbital rotation degrees of freedom.\n");
        return false;
      }

      // Step 3: gradient g_alpha = 2 Re(F_ij)(n_j - n_i) and diagonal
      // Hessian estimate h_alpha = 2 (eps_i - eps_j)(n_j - n_i). For
      // complex orbitals the real and imaginary parts of K_ij are
      // independent DOFs sharing h.
      ctx.n_dof = ctx.dofs.size();
      ctx.n_par = ctx.is_complex ? 2 * ctx.n_dof : ctx.n_dof;
      ctx.g = Vector<Tbase>::Zero(ctx.n_par);
      ctx.h = Vector<Tbase>::Zero(ctx.n_par);
      for(size_t a = 0; a < ctx.n_dof; a++) {
        const auto & dof = ctx.dofs[a];
        size_t b = std::get<0>(dof);
        Index i = std::get<1>(dof);
        Index j = std::get<2>(dof);
        Tbase dn = ctx.n_ref[b](j) - ctx.n_ref[b](i);
        Tbase de = ctx.eps[b](i) - ctx.eps[b](j);
        Torb Fij = F_pseudo[b](i, j);
        ctx.g(a) = 2 * std::real(Fij) * dn;
        ctx.h(a) = 2 * de * dn;
        if(ctx.is_complex) {
          ctx.g(ctx.n_dof + a) = 2 * std::imag(Fij) * dn;
          ctx.h(ctx.n_dof + a) = ctx.h(a);
        }
      }
      return true;
    }

    Vector<Tbase> preconditioned_sd_direction_(
        const RotationStepContext & ctx, Tbase sigma) const {
      Vector<Tbase> d(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        d(k) = -ctx.g(k) / (sigma + std::max(Tbase(0), ctx.h(k)));
      return d;
    }

    Orbitals<Torb> build_K_(const Vector<Tbase> & d,
                            const RotationStepContext & ctx) const {
      Orbitals<Torb> K(ctx.C_pseudo.size());
      for(size_t b = 0; b < ctx.C_pseudo.size(); b++)
        if(ctx.C_pseudo[b].cols() > 0)
          K[b] = Matrix<Torb>::Zero(ctx.C_pseudo[b].cols(), ctx.C_pseudo[b].cols());
      for(size_t a = 0; a < ctx.n_dof; a++) {
        const auto & dof = ctx.dofs[a];
        size_t b = std::get<0>(dof);
        Index i = std::get<1>(dof);
        Index j = std::get<2>(dof);
        if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
          K[b](i, j) = d(a);
          K[b](j, i) = -d(a);
        } else {
          Torb val(d(a), d(ctx.n_dof + a));
          K[b](i, j) = val;
          K[b](j, i) = -std::conj(val);
        }
      }
      return K;
    }

    Tbase t_max_for_K_(const Orbitals<Torb> & K) const {
      Tbase t_max = std::numeric_limits<Tbase>::max();
      for(size_t b = 0; b < K.size(); b++) {
        if(K[b].size() == 0) continue;
        Matrix<std::complex<Tbase>> KI = K[b].template cast<std::complex<Tbase>>() * std::complex<Tbase>(Tbase{0}, Tbase{-1});
        // Hermitize to suppress eig_sym roundoff warnings; -iK is
        // analytically Hermitian for anti-Hermitian K.
        KI = std::complex<Tbase>(Tbase{0.5}) * (KI + KI.adjoint().eval());
        Eigen::SelfAdjointEigenSolver<Matrix<std::complex<Tbase>>> es(KI);
        Vector<Tbase> ev = es.eigenvalues();
        if(ev.size() > 0) {
          Tbase max_abs = ev.array().abs().maxCoeff();
          if(max_abs > 0) t_max = std::min(t_max, Tbase(M_PI / 2) / max_abs);
        }
      }
      return t_max;
    }

    std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>
    evaluate_rotation_at_(const Orbitals<Torb> & K, Tbase t,
                          const RotationStepContext & ctx) {
      size_t nblocks = ctx.C_pseudo.size();
      Orbitals<Torb> C_new(nblocks);
      for(size_t b = 0; b < nblocks; b++) {
        if(ctx.C_pseudo[b].cols() == 0) {
          C_new[b] = ctx.C_pseudo[b];
          continue;
        }
        Matrix<Torb> tK = t * K[b];
        C_new[b] = ctx.C_pseudo[b] * expm_antihermitian(tK);
      }
      DensityMatrix<Torb, Tbase> dm = std::make_pair(C_new, ctx.n_ref);
      auto fock = fock_builder_(dm);
      number_of_fock_evaluations_++;
      return std::make_pair(dm, fock);
    }

    Vector<Tbase> directional_gradient_at_trial_(
        const RotationStepContext & ctx,
        const Orbitals<Torb> & C_trial,
        const FockMatrix<Torb> & F_trial) const {
      std::vector<Matrix<Torb>> F_MO(C_trial.size());
      for(size_t b = 0; b < C_trial.size(); b++) {
        if(C_trial[b].cols() == 0) continue;
        F_MO[b] = C_trial[b].adjoint() * F_trial[b] * C_trial[b];
      }
      Vector<Tbase> g_trial = Vector<Tbase>::Zero(ctx.n_par);
      for(size_t a = 0; a < ctx.n_dof; a++) {
        const auto & dof = ctx.dofs[a];
        size_t b = std::get<0>(dof);
        Index i = std::get<1>(dof);
        Index j = std::get<2>(dof);
        Torb Fij = F_MO[b](i, j);
        Tbase dn = ctx.n_ref[b](j) - ctx.n_ref[b](i);
        g_trial(a) = 2 * std::real(Fij) * dn;
        if(ctx.is_complex)
          g_trial(ctx.n_dof + a) = 2 * std::imag(Fij) * dn;
      }
      return g_trial;
    }

    template<typename Trial0DirectionFunc>
    bool sigma_line_search_(const RotationStepContext & ctx,
                            Trial0DirectionFunc trial_0_direction,
                            Vector<Tbase> & d_accepted,
                            const char * tag) {
      const Tbase sigma_0 = initial_level_shift_;
      const int max_sigma_trials = 3;
      const int max_t_trials = 8;
      const Tbase t_floor_ratio = Tbase(1e-6);
      const Tbase gnorm2 = ctx.g.dot(ctx.g);
      const Tbase slope_u_at_0 = -gnorm2 / sigma_0;

      Tbase sigma = sigma_0;
      bool first_sigma = true;
      bool have_sigma_cubic = false;
      Tbase E_first_sigma_trial = ctx.E_ref;
      Tbase slope_u_at_1 = 0;

      bool success = false;
      for(int sigma_trial = 0; sigma_trial < max_sigma_trials && !success; sigma_trial++) {
        Vector<Tbase> d = first_sigma
          ? trial_0_direction(sigma)
          : preconditioned_sd_direction_(ctx, sigma);

        Tbase slope_0 = d.dot(ctx.g);  // dE/dt at t = 0
        if(!std::isfinite(slope_0) || slope_0 >= 0) {
          if(verbosity_ >= 5)
            printf("%s: direction at sigma = %e is not descent (g.d = %e).\n",
                   tag, sigma, slope_0);
          sigma *= 2;
          first_sigma = false;
          continue;
        }

        Orbitals<Torb> K = build_K_(d, ctx);
        Tbase t_max = t_max_for_K_(K);
        if(!std::isfinite(t_max) || t_max <= 0) {
          if(verbosity_ >= 5)
            printf("%s: t_max not well-defined at sigma = %e.\n", tag, sigma);
          sigma *= 2;
          first_sigma = false;
          continue;
        }

        // Inner t-walk: cubic-Hermite-refined Armijo line search.
        Tbase t = std::min(Tbase(1), t_max);
        const Tbase t_floor = t_max * t_floor_ratio;
        bool first_t_trial_in_sigma = true;
        for(int t_trial = 0; t_trial < max_t_trials && !success; t_trial++) {
          auto trial_result = evaluate_rotation_at_(K, t, ctx);
          Tbase E_t = trial_result.second.first;
          if(verbosity_ >= 5)
            printf("%s: trial sigma %e t %e, energy % .10f, change %e\n",
                   tag, sigma, t, E_t, E_t - ctx.E_ref);

          if(E_t < ctx.E_ref) {
            add_entry(trial_result.first, trial_result.second);
            d_accepted = d;
            success = true;
            break;
          }

          // Failed t-trial: gather slope at the trial point so we can
          // predict either where the slope flips sign (overshoot) or
          // where the interior minimum of a non-monotonic profile sits.
          Vector<Tbase> g_t = directional_gradient_at_trial_(
              ctx, trial_result.first.first, trial_result.second.second);
          Tbase slope_t = g_t.dot(d);

          // Record info for the outer sigma cubic Hermite (only the
          // very first trial in this sigma-trial provides it).
          if(first_sigma && first_t_trial_in_sigma) {
            Tbase dEdsigma = 0;
            for(size_t k = 0; k < ctx.n_par; k++) {
              Tbase denom = sigma_0 + std::max(Tbase(0), ctx.h(k));
              dEdsigma += g_t(k) * ctx.g(k) / (denom * denom);
            }
            slope_u_at_1 = -sigma_0 * dEdsigma;
            E_first_sigma_trial = E_t;
            have_sigma_cubic = true;
          }
          first_t_trial_in_sigma = false;

          if(t <= t_floor) break;

          // Predict next t from the cubic Hermite fit on [0, t] with
          // E(0) = E_ref, E'(0) = slope_0, E(t) = E_t, E'(t) = slope_t.
          Tbase t_next = t * Tbase(0.5);
          try {
            auto cubic = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(
                ctx.E_ref, slope_0, t, E_t, slope_t);
            Tbase a2 = std::get<2>(cubic);
            Tbase a3 = std::get<3>(cubic);
            auto roots = HelperRoutines::cubic_polynomial_zeros<Tbase>(
                std::get<0>(cubic), std::get<1>(cubic), a2, a3);
            Tbase t_star = std::numeric_limits<Tbase>::quiet_NaN();
            for(Tbase r : {roots.first, roots.second}) {
              if(!(r > 0 && r < t)) continue;
              if(2*a2 + 6*a3*r > 0) { t_star = r; break; }
            }
            if(std::isfinite(t_star) && t_star > 0 && t_star < t) {
              t_next = t_star;
              if(verbosity_ >= 5)
                printf("%s: cubic Hermite predicts t = %e (in [0, %e]).\n",
                       tag, t_next, t);
            }
          } catch(const std::logic_error &) {
            // Cubic derivative has no real roots; fall through to halving.
          }
          if(t_next < t_floor) t_next = t_floor;
          if(t_next >= t)      t_next = t * Tbase(0.5);  // ensure progress
          t = t_next;
        }
        if(success) break;

        first_sigma = false;
        if(sigma_trial + 1 == max_sigma_trials) break;

        // Outer sigma fallback: predict the next sigma from a cubic
        // Hermite fit in u = sigma_0/sigma using slope_u data, or fall
        // back to geometric doubling.
        Tbase sigma_next = sigma * 2;
        if(have_sigma_cubic) {
          auto cubic = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(
              ctx.E_ref, slope_u_at_0, Tbase(1), E_first_sigma_trial, slope_u_at_1);
          Tbase a2 = std::get<2>(cubic);
          Tbase a3 = std::get<3>(cubic);
          try {
            auto roots = HelperRoutines::cubic_polynomial_zeros<Tbase>(
                std::get<0>(cubic), std::get<1>(cubic), a2, a3);
            Tbase u_star = std::numeric_limits<Tbase>::quiet_NaN();
            for(Tbase u : {roots.first, roots.second}) {
              if(!(u > 0 && u < 1)) continue;
              if(2*a2 + 6*a3*u > 0) { u_star = u; break; }
            }
            if(std::isfinite(u_star) && u_star > 0 && u_star < 1) {
              Tbase predicted = sigma_0 / u_star;
              if(std::isfinite(predicted) && predicted > sigma
                 && predicted < sigma * 100) {
                sigma_next = predicted;
                if(verbosity_ >= 5)
                  printf("%s: cubic Hermite predicts sigma = %e (u* = %e).\n",
                         tag, sigma_next, u_star);
              }
            }
          } catch(const std::logic_error &) {
            // Cubic derivative has no real roots; fall through to geometric.
          }
        }
        sigma = sigma_next;
      }
      return success;
    }

    void apply_pr_plus_cg_mix_(Vector<Tbase> & d,
                               const RotationStepContext & ctx) const {
      if(previous_orbital_gradient_.size() != ctx.g.size()
         || previous_orbital_direction_.size() != ctx.g.size()
         || previous_orbital_dofs_ != ctx.dofs)
        return;
      Tbase denom = previous_orbital_gradient_.dot(previous_orbital_gradient_);
      if(denom <= std::numeric_limits<Tbase>::min()) return;
      Tbase beta_PR = ctx.g.dot(ctx.g - previous_orbital_gradient_) / denom;
      Tbase beta = std::max(beta_PR, Tbase(0));
      Vector<Tbase> d_cg = d + beta * previous_orbital_direction_;
      if(d_cg.dot(ctx.g) < 0) {
        if(verbosity_ >= 5)
          printf("Scaled SD: CG update with beta = %e (PR = %e).\n", beta, beta_PR);
        d = d_cg;
      } else if(verbosity_ >= 5) {
        printf("Scaled SD: CG direction not descent, resetting to preconditioned SD.\n");
      }
    }

    bool scaled_steepest_descent_step() {
      RotationStepContext ctx;
      if(!build_rotation_step_context_(ctx)) return false;

      Vector<Tbase> d_accepted;
      bool success = sigma_line_search_(
          ctx,
          [&](Tbase sigma) {
            Vector<Tbase> d = preconditioned_sd_direction_(ctx, sigma);
            apply_pr_plus_cg_mix_(d, ctx);
            return d;
          },
          d_accepted,
          "Scaled SD");

      if(success) {
        previous_orbital_gradient_ = ctx.g;
        previous_orbital_direction_ = d_accepted;
        previous_orbital_dofs_ = ctx.dofs;
      } else {
        previous_orbital_gradient_.resize(0);
        previous_orbital_direction_.resize(0);
        previous_orbital_dofs_.clear();
      }
      return success;
    }

    void clear_lbfgs_state_() {
      lbfgs_.reset();
    }

    Vector<Tbase> lbfgs_direction_(
        const RotationStepContext & ctx, Tbase sigma) const {
      const auto & s = lbfgs_->s;
      const auto & y = lbfgs_->y;
      const auto & rho = lbfgs_->rho;
      Vector<Tbase> q = ctx.g;
      size_t m = s.size();
      std::vector<Tbase> alpha(m);
      for(size_t i = m; i-- > 0;) {
        alpha[i] = rho[i] * s[i].dot(q);
        q -= alpha[i] * y[i];
      }
      Vector<Tbase> r(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        r(k) = q(k) / (sigma + std::max(Tbase(0), ctx.h(k)));
      for(size_t i = 0; i < m; i++) {
        Tbase beta = rho[i] * y[i].dot(r);
        r += (alpha[i] - beta) * s[i];
      }
      return -r;
    }

    void apply_lbfgs_correction_(Vector<Tbase> & d,
                                 const RotationStepContext & ctx) const {
      if(!lbfgs_ || lbfgs_->s.empty()) return;
      if(lbfgs_->history_dofs != ctx.dofs) return;
      Vector<Tbase> d_lbfgs = lbfgs_direction_(ctx, initial_level_shift_);
      if(d_lbfgs.dot(ctx.g) < 0) {
        if(verbosity_ >= 5)
          printf("L-BFGS: applying two-loop direction (history size %zu).\n",
                 lbfgs_->s.size());
        d = d_lbfgs;
      } else if(verbosity_ >= 5) {
        printf("L-BFGS: two-loop direction not descent, resetting to preconditioned SD.\n");
      }
    }

    bool lbfgs_step() {
      RotationStepContext ctx;
      if(!build_rotation_step_context_(ctx)) return false;
      if(!lbfgs_) lbfgs_ = std::make_unique<LBFGSState>();
      LBFGSState & st = *lbfgs_;

      // Promote the pending (s, g_prev) into a full (s, y) history
      // pair using the current gradient, but only if the DOF set is
      // unchanged since the pair was recorded.
      if(st.pending_s.size() == (Index)ctx.n_par
         && st.pending_g.size() == (Index)ctx.n_par
         && st.history_dofs == ctx.dofs) {
        Vector<Tbase> y = ctx.g - st.pending_g;
        Tbase ys = y.dot(st.pending_s);
        if(ys > std::numeric_limits<Tbase>::min()) {
          st.s.push_back(st.pending_s);
          st.y.push_back(y);
          st.rho.push_back(Tbase(1) / ys);
          while(st.s.size() > (size_t) maximum_history_length_) {
            st.s.pop_front();
            st.y.pop_front();
            st.rho.pop_front();
          }
        } else if(verbosity_ >= 5) {
          printf("L-BFGS: curvature condition violated (y.s = %e), pair dropped.\n", ys);
        }
      } else if(!st.history_dofs.empty() && st.history_dofs != ctx.dofs) {
        clear_lbfgs_state_();
        lbfgs_ = std::make_unique<LBFGSState>();
        // st reference is now dangling -- rebind below if we keep using it.
      }
      // Pending pair has been consumed; clear it before this step.
      if(lbfgs_) {
        lbfgs_->pending_s.resize(0);
        lbfgs_->pending_g.resize(0);
      }

      Vector<Tbase> d_accepted;
      bool success = sigma_line_search_(
          ctx,
          [&](Tbase sigma) {
            Vector<Tbase> d = preconditioned_sd_direction_(ctx, sigma);
            apply_lbfgs_correction_(d, ctx);
            return d;
          },
          d_accepted,
          "L-BFGS");

      if(success) {
        if(!lbfgs_) lbfgs_ = std::make_unique<LBFGSState>();
        lbfgs_->pending_s = d_accepted;
        lbfgs_->pending_g = ctx.g;
        lbfgs_->history_dofs = ctx.dofs;
      } else {
        clear_lbfgs_state_();
      }
      return success;
    }

    std::vector<FockBuilderReturn<Torb, Tbase>>
    evaluate_batch_(const std::vector<DensityMatrix<Torb, Tbase>> & densities) {
      if(batched_fock_builder_) {
        auto results = batched_fock_builder_(densities);
        if(results.size() != densities.size()) {
          std::ostringstream oss;
          oss << "Batched Fock builder returned " << results.size()
              << " entries for " << densities.size() << " densities.\n";
          throw std::logic_error(oss.str());
        }
        number_of_fock_evaluations_ += densities.size();
        return results;
      }
      std::vector<FockBuilderReturn<Torb, Tbase>> results;
      results.reserve(densities.size());
      for(const auto & dm: densities) {
        results.push_back(fock_builder_(dm));
        number_of_fock_evaluations_++;
      }
      return results;
    }

    std::vector<IndexVector> occupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> occ_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        occ_idx[l]=find_indices_where(occupations[l], [this](Tbase v){ return v >= occupied_threshold_; });
      }
      return occ_idx;
    }

    std::vector<IndexVector> unoccupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> virt_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        virt_idx[l]=find_indices_where(occupations[l], [this](Tbase v){ return v < occupied_threshold_; });
      }
      return virt_idx;
    }

    void pseudo_canonicalise_(const Orbitals<Torb> & C_ref,
                              const FockMatrix<Torb> & F_ref,
                              const OrbitalOccupations<Tbase> & n_ref,
                              Orbitals<Torb> & C_pseudo,
                              OrbitalEnergies<Tbase> & eps_out) const {
      size_t nblocks = C_ref.size();
      C_pseudo.assign(nblocks, Matrix<Torb>());
      eps_out.assign(nblocks, Vector<Tbase>());
      for(size_t b = 0; b < nblocks; b++) {
        if(empty_block(b) || C_ref[b].cols() == 0) {
          C_pseudo[b] = C_ref[b];
          eps_out[b].resize(0);
          continue;
        }
        Index n_b = C_ref[b].cols();
        Matrix<Torb> F_MO = C_ref[b].adjoint() * F_ref[b] * C_ref[b];
        Matrix<Torb> U = Matrix<Torb>::Identity(n_b, n_b);
        eps_out[b] = Vector<Tbase>::Zero(n_b);
        std::vector<bool> used(n_b, false);
        for(Index i = 0; i < n_b; i++) {
          if(used[i]) continue;
          std::vector<Index> grp = {i};
          used[i] = true;
          for(Index j = i + 1; j < n_b; j++)
            if(!used[j] &&
               std::abs(n_ref[b](i) - n_ref[b](j)) < occupation_change_threshold_) {
              grp.push_back(j);
              used[j] = true;
            }
          IndexVector idx(grp.size());
          for(size_t k = 0; k < grp.size(); k++) idx(k) = grp[k];
          Matrix<Torb> F_sub(idx.size(), idx.size());
          for(Index r = 0; r < idx.size(); r++)
            for(Index c = 0; c < idx.size(); c++)
              F_sub(r, c) = F_MO(idx(r), idx(c));
          // Enforce exact Hermiticity to silence eig_sym roundoff
          // warnings; the unsymmetric residual is O(eps) for an
          // analytically Hermitian operator.
          F_sub = Tbase(0.5) * (F_sub + F_sub.adjoint().eval());
          Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(F_sub);
          Vector<Tbase> eps_sub = es.eigenvalues();
          Matrix<Torb> U_sub = es.eigenvectors();
          for(Index k = 0; k < idx.size(); k++) {
            eps_out[b](idx(k)) = eps_sub(k);
            for(Index l = 0; l < idx.size(); l++)
              U(idx(l), idx(k)) = U_sub(l, k);
          }
        }
        C_pseudo[b] = C_ref[b] * U;
      }
    }

    size_t compute_active_rotation_count() const {
      const auto C  = get_orbitals();
      const auto n  = get_orbital_occupations();
      const auto F  = get_fock_matrix();
      Orbitals<Torb> C_pseudo_unused;
      OrbitalEnergies<Tbase> eps;
      pseudo_canonicalise_(C, F, n, C_pseudo_unused, eps);

      // Aufbau-fill these energies and take the upper window edge per block.
      auto aufbau = update_occupations(eps);
      const Tbase inf = std::numeric_limits<Tbase>::infinity();
      Vector<Tbase> window_edge(C.size());
      window_edge.setConstant(-inf);
      for(size_t b = 0; b < C.size(); b++) {
        if(eps[b].size() == 0) continue;
        Tbase max_occ_eps = -inf;
        for(Index i = 0; i < eps[b].size(); i++)
          if(aufbau[b](i) > occupation_change_threshold_ && eps[b](i) > max_occ_eps)
            max_occ_eps = eps[b](i);
        if(std::isfinite(max_occ_eps))
          window_edge(b) = max_occ_eps + optimal_damping_degeneracy_threshold_;
      }

      size_t total = 0;
      for(size_t b = 0; b < C.size(); b++) {
        if(empty_block(b) || C[b].cols() == 0) continue;
        Index n_b = C[b].cols();
        Tbase edge = window_edge(b);

        // Sort orbital indices by energy (ascending).
        std::vector<size_t> order(n_b);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t bb) { return eps[b](a) < eps[b](bb); });

        // Walk sorted orbitals and identify ODA-style clusters.
        size_t start = 0;
        while(start < (size_t)n_b) {
          Tbase eps_start = eps[b](order[start]);
          size_t end = start;
          while(end + 1 < (size_t)n_b
                && eps[b](order[end + 1]) - eps_start
                    < optimal_damping_degeneracy_threshold_)
            end++;

          // Skip clusters whose lowest energy is above the window edge.
          if(eps_start <= edge && end > start) {
            Tbase min_n = n[b](order[start]);
            Tbase max_n = n[b](order[start]);
            for(size_t k = start + 1; k <= end; k++) {
              Tbase nk = n[b](order[k]);
              if(nk < min_n) min_n = nk;
              if(nk > max_n) max_n = nk;
            }
            if(max_n - min_n >= occupation_change_threshold_)
              total++;
          }
          start = end + 1;
        }
      }
      return total;
    }

    bool has_integer_occupations() const {
      const auto occupations = get_orbital_occupations();
      for(const auto & occ_block : occupations) {
        for(Index i = 0; i < occ_block.size(); i++) {
          Tbase n = occ_block(i);
          Tbase rounded = std::round(n);
          if(std::abs(n - rounded) >= occupation_change_threshold_)
            return false;
        }
      }
      return true;
    }

  public:
    SCFSolver(const IndexVector & number_of_blocks_per_particle_type, const Vector<Tbase> & maximum_occupation, const Vector<Tbase> & number_of_particles, const FockBuilder<Torb, Tbase> & fock_builder, const std::vector<std::string> & block_descriptions) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder), block_descriptions_(block_descriptions), frozen_occupations_(false), verbosity_(5) {
      // Run sanity checks
      number_of_blocks_ = number_of_blocks_per_particle_type_.sum();
      if((size_t)maximum_occupation_.size() != number_of_blocks_) {
        std::ostringstream oss;
        oss << "Vector of maximum occupation is not of expected length! Got " << maximum_occupation_.size() << " elements, expected " << number_of_blocks_ << "!\n";
        throw std::logic_error(oss.str());
      }
      if(number_of_particles_.size() != number_of_blocks_per_particle_type_.size()) {
        std::ostringstream oss;
        oss << "Vector of number of particles is not of expected length! Got " << number_of_particles_.size() << " elements, expected " << number_of_blocks_per_particle_type_.transpose() << "!\n";
        throw std::logic_error(oss.str());
      }
      if(block_descriptions_.size() != number_of_blocks_) {
        std::ostringstream oss;
        oss << "Vector of block descriptions is not of expected length! Got " << block_descriptions_.size() << " elements, expected " << number_of_blocks_ << "!\n";
        throw std::logic_error(oss.str());
      }
    }

    void initialize_with_fock(const FockMatrix<Torb> & fock_guess) {
      if(fock_guess.size() != number_of_blocks_)
        throw std::logic_error("Fed in Fock matrix does not have the required number of blocks!\n");

      // Compute orbitals
      auto diagonalized_fock = compute_orbitals(fock_guess);
      const auto & orbitals = diagonalized_fock.first;
      const auto & orbital_energies = diagonalized_fock.second;

      // Compute the occupations
      orbital_occupations_ = update_occupations(orbital_energies);
      // This routine handles the rest
      initialize_with_orbitals(orbitals, orbital_occupations_);
    }

    void initialize_with_orbitals(const Orbitals<Torb> & orbitals, const OrbitalOccupations<Tbase> & orbital_occupations) {
      if(orbitals.size() != orbital_occupations.size())
        throw std::logic_error("Fed in orbitals and orbital occupations are not consistent!\n");
      if(orbitals.size() != number_of_blocks_)
        throw std::logic_error("Fed in orbitals and orbital occupations do not have the required number of blocks!\n");
      orbital_history_.clear();

      // Reset number of evaluations
      number_of_fock_evaluations_ = 0;
      add_entry(std::make_pair(orbitals, orbital_occupations));

      // Check that dimensions are consistent
      bool consistent=true;
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        if(get_orbital_block(0,iblock).cols() != get_fock_matrix_block(0,iblock).cols()) {
          printf("get_orbital_block(0,iblock).cols()=%i != get_fock_matrix_block(0,iblock).cols())=%i\n",(int) get_orbital_block(0,iblock).cols(),(int) get_fock_matrix_block(0,iblock).cols());
          consistent=false;
        }
        if(get_orbital_occupation_block(0,iblock).size() != get_fock_matrix_block(0,iblock).cols()) {
          if(verbosity_>=10)
            printf("get_orbital_occupation_block(0,iblock).size()=%i != get_fock_matrix_block(0,iblock).cols()=%i\n",(int) get_orbital_occupation_block(0,iblock).size(),(int) get_fock_matrix_block(0,iblock).cols());
          consistent=false;
        }
      }
      // If they are not consistent (e.g. when a read-in guess has been used)
      if(not consistent) {
        if(verbosity_>=5)
          printf("Fed-in orbitals are not consistent with Fock matrix, recomputing orbitals\n");

        // Diagonalize the Fock matrix we just computed
        auto new_orbitals = compute_orbitals(get_fock_matrix());
        // Determine new occupations
        auto new_occupations = update_occupations(new_orbitals.second);

        // Clear out the old history
        orbital_history_.clear();
        // and add the new entry
        add_entry(std::make_pair(new_orbitals.first, new_occupations));
      }
    }

    void fixed_number_of_particles_per_block(const Vector<Tbase> & number_of_particles_per_block) {
      fixed_number_of_particles_per_block_ = number_of_particles_per_block;
    }

    // === Settings façade ==================================================
    //
    // Type-tagged set / get / options catalog. Every knob the solver
    // exposes is enumerated in options(); every writable knob is
    // reachable via set(key, value); every knob and read-only
    // diagnostic is reachable via get_real / get_int / get_string
    // according to its declared type. Unknown or wrong-type keys
    // throw std::invalid_argument.

    struct OptionInfo {
      const char * key;
      const char * type;      
      bool         writable;  
      const char * doc;       
    };

    static const std::vector<OptionInfo> & options() {
      static const std::vector<OptionInfo> catalog = {
        // -- Convergence -----------------------------------------------------
        {"convergence_threshold", "real", true,
         "DIIS-error convergence threshold"},
        {"noise_safety_factor",   "real", true,
         "K in effective threshold max(convergence_threshold, K * noise_floor)"},
        {"error_norm",            "string", true,
         "DIIS error norm; one of rms, fro, inf, 1, 2"},
        {"methods",               "string", true,
         "SCF method mix consumed by run(); e.g. \"DIIS + ODA + CG\", \"DIIS\", \"ODA + CG\", \"DIIS + ODA + LBFGS\""},
        // -- DIIS ------------------------------------------------------------
        {"diis_epsilon",          "real", true,
         "pure-DIIS blend cutoff"},
        {"diis_threshold",        "real", true,
         "A/EDIIS blend cutoff (Garza-Scuseria)"},
        {"diis_diagonal_damping", "real", true,
         "DIIS matrix diagonal damping"},
        {"diis_restart_factor",   "real", true,
         "DIIS history restart factor"},
        // -- Optimal damping (ODA) -------------------------------------------
        {"optimal_damping_threshold", "real", true,
         "DIIS error above which ODA takes over"},
        {"optimal_damping_degeneracy_threshold", "real", true,
         "ODA orbital-degeneracy window (Eh)"},
        // -- History / iteration ---------------------------------------------
        {"maximum_iterations",     "int", true,
         "outer SCF iteration cap"},
        {"maximum_history_length", "int", true,
         "DIIS and L-BFGS history depth"},
        {"oda_restart_steps",      "int", true,
         "steps of no DIIS progress before switching to ODA"},
        {"orbital_rotation_steps_after_oda", "int", true,
         "orbital-rotation steps after each ODA (0 = use last_active_rotation_count)"},
        // -- Orbital-rotation preconditioner ---------------------------------
        {"minimal_gradient_projection", "real", true,
         "minimum preconditioned-CG projection on gradient"},
        {"initial_level_shift",         "real", true,
         "orbital-rotation preconditioner floor"},
        {"level_shift_factor",          "real", true,
         "level-shift diminution factor"},
        // -- Occupations -----------------------------------------------------
        {"occupied_threshold",          "real", true,
         "occupied-orbital detection cutoff"},
        {"occupation_change_threshold", "real", true,
         "occupation-equality tolerance"},
        {"density_restart_factor",      "real", true,
         "history density-diff restart factor"},
        {"frozen_occupations",          "int",  true,
         "pin occupations across SCF (0 or 1)"},
        // -- Verbosity -------------------------------------------------------
        {"verbosity", "int", true, "0..30"},
        // -- Read-only diagnostics -------------------------------------------
        {"noise_floor",                "real", false,
         "frozen roundoff floor of DIIS error, populated by run()"},
        {"number_of_fock_evaluations", "int",  false,
         "Fock-evaluation counter (reset on initialize_with_*)"},
        {"last_polytope_dimension",    "int",  false,
         "ODA polytope dimension of the most recent optimal_damping_step"},
        {"last_active_rotation_count", "int",  false,
         "active rotations counted by the most recent ODA step"},
        {"converged",                  "int",  false,
         "0 or 1 -- re-evaluates the convergence rule now"},
      };
      return catalog;
    }

    void set(const std::string & key, Tbase v) {
      if      (key == "convergence_threshold")                 convergence_threshold_ = v;
      else if (key == "noise_safety_factor")                   noise_safety_factor_ = v;
      else if (key == "diis_epsilon")                          diis_epsilon_ = v;
      else if (key == "diis_threshold")                        diis_threshold_ = v;
      else if (key == "diis_diagonal_damping")                 diis_diagonal_damping_ = v;
      else if (key == "diis_restart_factor")                   diis_restart_factor_ = v;
      else if (key == "optimal_damping_threshold")             optimal_damping_threshold_ = v;
      else if (key == "optimal_damping_degeneracy_threshold")  optimal_damping_degeneracy_threshold_ = v;
      else if (key == "minimal_gradient_projection")           minimal_gradient_projection_ = v;
      else if (key == "initial_level_shift")                   initial_level_shift_ = v;
      else if (key == "level_shift_factor")                    level_shift_factor_ = v;
      else if (key == "occupied_threshold")                    occupied_threshold_ = v;
      else if (key == "occupation_change_threshold")           occupation_change_threshold_ = v;
      else if (key == "density_restart_factor")                density_restart_factor_ = v;
      else throw std::invalid_argument(
        "SCFSolver::set(real): unknown or non-real key '" + key + "'");
    }

    void set(const std::string & key, int v) {
      if      (key == "verbosity")                        verbosity_ = v;
      else if (key == "maximum_iterations")               maximum_iterations_ = (size_t) v;
      else if (key == "maximum_history_length")           maximum_history_length_ = v;
      else if (key == "oda_restart_steps")                oda_restart_steps_ = v;
      else if (key == "orbital_rotation_steps_after_oda") orbital_rotation_steps_after_oda_ = (size_t) v;
      else if (key == "frozen_occupations")               frozen_occupations_ = (v != 0);
      else throw std::invalid_argument(
        "SCFSolver::set(int): unknown or non-int key '" + key + "'");
    }

    void set(const std::string & key, const std::string & v) {
      if (key == "error_norm") {
        std::string prev = error_norm_;
        error_norm_ = v;
        try {
          Vector<Tbase> test = Vector<Tbase>::Ones(1);
          (void) norm(test);
        } catch (...) {
          error_norm_ = prev;
          throw;
        }
      } else if (key == "methods") {
        // Validate by parsing; store canonical uppercase.
        (void) parse_method_string(v);
        methods_ = to_upper_copy(v);
      } else {
        throw std::invalid_argument(
          "SCFSolver::set(string): unknown or non-string key '" + key + "'");
      }
    }

    Tbase get_real(const std::string & key) const {
      if      (key == "convergence_threshold")                 return convergence_threshold_;
      else if (key == "noise_safety_factor")                   return noise_safety_factor_;
      else if (key == "noise_floor")                           return noise_floor_;
      else if (key == "diis_epsilon")                          return diis_epsilon_;
      else if (key == "diis_threshold")                        return diis_threshold_;
      else if (key == "diis_diagonal_damping")                 return diis_diagonal_damping_;
      else if (key == "diis_restart_factor")                   return diis_restart_factor_;
      else if (key == "optimal_damping_threshold")             return optimal_damping_threshold_;
      else if (key == "optimal_damping_degeneracy_threshold")  return optimal_damping_degeneracy_threshold_;
      else if (key == "minimal_gradient_projection")           return minimal_gradient_projection_;
      else if (key == "initial_level_shift")                   return initial_level_shift_;
      else if (key == "level_shift_factor")                    return level_shift_factor_;
      else if (key == "occupied_threshold")                    return occupied_threshold_;
      else if (key == "occupation_change_threshold")           return occupation_change_threshold_;
      else if (key == "density_restart_factor")                return density_restart_factor_;
      else throw std::invalid_argument(
        "SCFSolver::get_real: unknown or non-real key '" + key + "'");
    }

    int get_int(const std::string & key) const {
      if      (key == "verbosity")                        return verbosity_;
      else if (key == "maximum_iterations")               return (int) maximum_iterations_;
      else if (key == "maximum_history_length")           return maximum_history_length_;
      else if (key == "oda_restart_steps")                return oda_restart_steps_;
      else if (key == "orbital_rotation_steps_after_oda") return (int) orbital_rotation_steps_after_oda_;
      else if (key == "frozen_occupations")               return frozen_occupations_ ? 1 : 0;
      else if (key == "number_of_fock_evaluations")       return (int) number_of_fock_evaluations_;
      else if (key == "last_polytope_dimension")          return (int) last_polytope_dimension_;
      else if (key == "last_active_rotation_count")       return (int) last_active_rotation_count_;
      else if (key == "converged")                        return converged() ? 1 : 0;
      else throw std::invalid_argument(
        "SCFSolver::get_int: unknown or non-int key '" + key + "'");
    }

    std::string get_string(const std::string & key) const {
      if      (key == "error_norm") return error_norm_;
      else if (key == "methods")    return methods_;
      else throw std::invalid_argument(
        "SCFSolver::get_string: unknown or non-string key '" + key + "'");
    }

    void print_settings(std::ostream & os = std::cout) const {
      const auto & catalog = options();
      size_t maxlen = 0;
      for (const auto & o : catalog)
        maxlen = std::max(maxlen, std::string(o.key).size());
      os << "OpenOrbitalOptimizer settings:\n";
      for (const auto & o : catalog) {
        os << "  " << std::left << std::setw((int)maxlen) << o.key << " = ";
        try {
          std::string t = o.type;
          if (t == "real") {
            os << std::scientific << std::setprecision(6) << get_real(o.key);
          } else if (t == "int") {
            os << get_int(o.key);
          } else if (t == "string") {
            os << "\"" << get_string(o.key) << "\"";
          } else {
            os << "?";
          }
        } catch (const std::exception &) {
          // Read-only diagnostic not yet available (e.g. converged
          // before initialize_with_*). Report as unavailable rather
          // than propagating -- print_settings shouldn't throw just
          // because history is empty.
          os << "n/a";
        }
        if (!o.writable) os << "  (read-only)";
        os << "\n";
      }
      os.flush();
    }

    static std::string citation() {
      return "Susi Lehtola and Lori A. Burns, "
             "\"OpenOrbitalOptimizer -- a reusable open source library "
             "for self-consistent field calculations\", "
             "J. Phys. Chem. A 129, 5651 (2025). "
             "doi:10.1021/acs.jpca.5c02110";
    }

    static void print_citation(std::ostream & os = std::cout) {
      os << "If you use OpenOrbitalOptimizer, please cite:\n"
         << "  " << citation() << "\n";
      os.flush();
    }

    // === End settings façade ==============================================

    void set_batched_fock_builder(BatchedFockBuilder<Torb, Tbase> builder) {
      batched_fock_builder_ = std::move(builder);
    }

    bool has_batched_fock_builder() const {
      return batched_fock_builder_ != nullptr;
    }

    Tbase get_energy(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Invalid entry!\n");
      return std::get<1>(orbital_history_[ihist]).first;
    }


    Tbase density_matrix_difference(size_t ihist, size_t jhist) {
      Tbase diff_norm = 0.0;
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        diff_norm += norm(vectorise(Matrix<Torb>(get_density_matrix_block(ihist, iblock)-get_density_matrix_block(jhist, iblock))));
      }
      return diff_norm;
    }

    Tbase norm(const Matrix<Tbase> & mat, std::string norm="") const {
      if(norm == "")
        norm=error_norm_;
      if(norm == "rms") {
        // rms isn't implemented in Armadillo for some reason
        if(mat.size() == 0)
          return 0;
        return mat.norm()/std::sqrt(1.0*mat.size());
      } else if(norm == "inf") {
        return mat.template lpNorm<Eigen::Infinity>();
      } else if(norm == "fro") {
        return mat.norm();
      } else if(norm == "1") {
        return mat.template lpNorm<1>();
      } else if(norm == "2") {
        return mat.norm();
      } else {
        throw std::logic_error("Unknown norm: " + norm);
      }
    }

    bool add_entry(const DensityMatrix<Torb, Tbase> & density) {
      // Compute the Fock matrix
      auto fock = fock_builder_(density);
      number_of_fock_evaluations_++;

      if(verbosity_>=5) {
        auto reference_energy = orbital_history_.size()>0 ? get_energy() : 0.0;
        printf("Evaluated energy % .10f (change from lowest %e)\n", fock.first, fock.first-reference_energy);
      }
      return add_entry(density, fock);
    }

    bool add_entry(const DensityMatrix<Torb, Tbase> & density, const FockBuilderReturn<Torb, Tbase> & fock) {
      // Make a pair
      orbital_history_.push_back(make_history_entry(density, fock));

      if(std::isnan(fock.first)) {
        throw std::logic_error("Got NaN total energy!\n");
      }
      if(std::isinf(fock.first)) {
        throw std::logic_error("Got +-infinite total energy!\n");
      }
      for(size_t iblock=0;iblock<fock.second.size();iblock++) {
        if(fock.second[iblock].rows()==0)
          continue;
        if(has_nan(fock.second[iblock])) {
          throw std::logic_error("Got NaN in Fock matrix!\n");
        }
        if(has_inf(fock.second[iblock])) {
          throw std::logic_error("Got +-infinity in Fock matrix!\n");
        }
      }

      if(orbital_history_.size()==1)
        // First try is a success by definition
        return true;
      else {
        // Otherwise we have to check if we lowered the energy
        Tbase new_energy = fock.first;
        Tbase old_energy = get_energy();
        bool return_value = new_energy < old_energy;

        // Now, we first sort the stack in increasing energy to get
        // the lowest energy solution at the beginning
        std::sort(orbital_history_.begin(), orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tbase> & a, const OrbitalHistoryEntry<Torb, Tbase> & b) {return std::get<1>(a).first < std::get<1>(b).first;});

        // and then the rest of the stack in decreasing iteration
        // number so that we always remove the oldest vector (lowest
        // index)
        std::sort(orbital_history_.begin()+1, orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tbase> & a, const OrbitalHistoryEntry<Torb, Tbase> & b) {return std::get<2>(a) > std::get<2>(b);});

        if(verbosity_>=20) {
          print_history();
        }

        // Drop last entry if we are over the history length limit
        if((int) orbital_history_.size() > maximum_history_length_)
          orbital_history_.pop_back();

        return return_value;
      }
    }

    void print_history() const {
      printf("Orbital history\n");
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++)
        printf("%2i % .9f % e % i\n",(int) ihist, get_energy(ihist), get_energy(ihist)-get_energy(), (int) get_index(ihist));
    }

    void reset_history() {
      while(orbital_history_.size()>1)
        orbital_history_.pop_back();
    }

    DiagonalizedFockMatrix<Torb,Tbase> compute_orbitals(const FockMatrix<Torb> & fock) const {
      DiagonalizedFockMatrix<Torb, Tbase> diagonalized_fock;
      // Allocate memory for orbitals and orbital energies
      diagonalized_fock.first.resize(fock.size());
      diagonalized_fock.second.resize(fock.size());

      // Diagonalize all blocks
      for(size_t iblock = 0; iblock < fock.size(); iblock++) {
        if(fock[iblock].size()==0)
          continue;
        // Symmetrize Fock matrix
        Matrix<Torb> fsymm(0.5*(fock[iblock]+fock[iblock].adjoint()));
        Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(fsymm);
        diagonalized_fock.second[iblock] = es.eigenvalues();
        diagonalized_fock.first[iblock] = es.eigenvectors();

        if(verbosity_>=10) {
          std::cout << block_descriptions_[iblock] + " orbital energies: " << diagonalized_fock.second[iblock].transpose() << std::endl;
        }
        fflush(stdout);
      }

      return diagonalized_fock;
    }

    Index particle_block_offset(size_t iparticle) const {
      return (iparticle>0) ? number_of_blocks_per_particle_type_.head(iparticle).sum() : 0;
    }

    std::vector<std::tuple<Tbase, size_t, size_t>> order_orbitals_by_energy(const OrbitalEnergies<Tbase> & orbital_energies, size_t iparticle) const {
      size_t block_offset = particle_block_offset(iparticle);
      std::vector<std::tuple<Tbase, size_t, size_t>> all_energies;
      for(size_t iblock = block_offset; iblock < block_offset + (size_t)number_of_blocks_per_particle_type_(iparticle); iblock++)
        for(Index iorb = 0; iorb < orbital_energies[iblock].size(); iorb++)
          all_energies.push_back(std::make_tuple(orbital_energies[iblock](iorb), iblock, iorb));
      std::stable_sort(all_energies.begin(), all_energies.end(), [](const std::tuple<Tbase, size_t, size_t> & a, const std::tuple<Tbase, size_t, size_t> & b) {return std::get<0>(a) < std::get<0>(b);});
      return all_energies;
    }

    Vector<Tbase> determine_number_of_particles_by_aufbau(const OrbitalEnergies<Tbase> & orbital_energies) const {
      Vector<Tbase> number_of_particles = Vector<Tbase>::Zero(number_of_blocks_);

      // Loop over particle types
      for(Index particle_type = 0; particle_type < number_of_blocks_per_particle_type_.size(); particle_type++) {
        auto all_energies = order_orbitals_by_energy(orbital_energies, particle_type);

        // Fill the orbitals in increasing energy. This is how many
        // particles we have to place
        Tbase num_left = number_of_particles_(particle_type);
        for(auto fill_orbital : all_energies) {
          // Increase number of occupied orbitals
          auto iblock = std::get<1>(fill_orbital);
          // Compute how many particles fit this orbital
          auto fill = std::min(maximum_occupation_(iblock), num_left);
          number_of_particles(iblock) += fill;
          num_left -= fill;
          // This should be sufficently tolerant to roundoff error
          if(num_left <= 10*std::numeric_limits<Tbase>::epsilon())
            break;
        }
      }

      return number_of_particles;
    }

    OrbitalOccupations<Tbase> update_occupations(const OrbitalEnergies<Tbase> & orbital_energies) const {
      if(frozen_occupations_)
        return get_orbital_occupations();

      // Number of particles per block
      Vector<Tbase> number_of_particles = ((size_t)fixed_number_of_particles_per_block_.size() == number_of_blocks_) ? fixed_number_of_particles_per_block_ : determine_number_of_particles_by_aufbau(orbital_energies);

      // Determine the number of occupied orbitals
      OrbitalOccupations<Tbase> occupations(orbital_energies.size());
      for(size_t iblock=0; iblock<orbital_energies.size(); iblock++) {
        if(orbital_energies[iblock].size()==0)
          continue;
        occupations[iblock] = Vector<Tbase>::Zero(orbital_energies[iblock].size());

        Tbase num_left = number_of_particles(iblock);
        for(Index iorb=0; iorb < occupations[iblock].size(); iorb++) {
          auto fill = std::min(maximum_occupation_(iblock), num_left);
          occupations[iblock](iorb) = fill;
          num_left -= fill;
          // This should be sufficently tolerant to roundoff error
          if(num_left <= 10*std::numeric_limits<Tbase>::epsilon())
            break;
        }
      }

      return occupations;
    }

    bool converged() const {
        // Nothing has been iterated yet, so trivially not converged.
        // Guarding here rather than at every call site (including
        // print_settings) keeps the diagnostic safe to query at any
        // point in the solver's lifetime.
        if(orbital_history_.empty())
            return false;
        if(callback_convergence_function_) {

            // Data to pass to callback function
            std::map<std::string, std::any> callback_data;
            callback_data["dE"] = get_energy() - old_energy_;
            callback_data["diis_error"] = norm(diis_error_vector(0));

            return callback_convergence_function_(callback_data);
        } else {
            Tbase effective = std::max(convergence_threshold_,
                                       noise_safety_factor_ * noise_floor_);
            return norm(diis_error_vector(0)) <= effective;
        }
    }

    void run() {
      AllowedMethods allowed = parse_method_string(methods_);
      if(frozen_occupations_)
        allowed.oda = false;  // occupations are pinned; ODA cannot move them

      // Freeze the roundoff noise floor of the DIIS residual from the
      // initial Fock. The basis conditioning is dominated by the
      // one-electron part so this barely moves during the run.
      noise_floor_ = compute_noise_floor();
      if(verbosity_ > 0 && noise_safety_factor_ > 0 &&
         convergence_threshold_ < noise_safety_factor_ * noise_floor_) {
        printf("Warning: convergence threshold %e is below %g x arithmetic "
               "noise floor %e Eh (epsilon=%e); clamping effective "
               "threshold to %e.\n",
               (double) convergence_threshold_,
               (double) noise_safety_factor_,
               (double) noise_floor_,
               (double) std::numeric_limits<Tbase>::epsilon(),
               (double) (noise_safety_factor_ * noise_floor_));
      }

      enum class StepKind { DIIS, ODA, OrbitalRotation };

      // Initial state: prefer DIIS, then ODA, then the orbital-rotation
      // step (CG or LBFGS). The chosen state is guaranteed to be
      // allowed by the parser above.
      StepKind state = allowed.diis ? StepKind::DIIS
                     : allowed.oda  ? StepKind::ODA
                                    : StepKind::OrbitalRotation;

      auto pick_next = [&allowed](
          std::initializer_list<StepKind> preferences, StepKind fallback) {
        for(auto k : preferences) {
          if((k == StepKind::DIIS && allowed.diis) ||
             (k == StepKind::ODA  && allowed.oda)  ||
             (k == StepKind::OrbitalRotation && allowed.orbital_rotation()))
            return k;
        }
        return fallback;
      };

      old_energy_ = 0.0;
      int failed_iterations = 0;
      // Number of orbital-rotation steps still owed by the current ODA -> orbital-rotation burst,
      // budgeted by orbital_rotation_steps_after_oda_ at the ODA transition.
      size_t orbital_rotation_steps_remaining = 0;

      // Burst-watch state: snapshot of the pseudo-canonical orbitals,
      // canonical-orbital energies, and equal-occupation sub-block
      // partition captured at the start of each ODA -> orbital-rotation
      // burst. After every rotation step the new pseudo-canonical
      // energies are compared against this snapshot to detect events
      // the rotation step itself cannot see: (i) a sign flip of
      // eps_i - eps_j for two differently-occupied orbitals (level
      // crossing across an occupation boundary), or (ii) an orbital
      // that has lost majority overlap with its initial equal-
      // occupation sub-block (qualitative change of orbital
      // character). Either trip ends the burst and hands control back
      // through the {DIIS, ODA} preference list so occupations can be
      // re-evaluated.
      Orbitals<Torb> burst_C_pseudo;
      OrbitalEnergies<Tbase> burst_eps;
      OrbitalOccupations<Tbase> burst_occ;
      // Per block: list of (i, j, eps_i - eps_j at burst start) for
      // every differently-occupied (i, j) pair, i < j. A pair trips
      // tripwire 1 when |delta_t - delta_0| exceeds
      // optimal_damping_degeneracy_threshold_, i.e. the gap has moved
      // by more than the energy window ODA uses to cluster orbitals.
      std::vector<std::vector<std::tuple<Index, Index, Tbase>>>
        burst_diff_occ_pairs;
      // Per block: index map orbital -> equal-occupation sub-block id
      // at burst start. Two orbitals share an id iff their occupations
      // differ by less than occupation_change_threshold_.
      std::vector<IndexVector> burst_subblock_id;
      // Minimum allowed sub-block-span overlap; falling below this for
      // any orbital ends the burst.
      const Tbase burst_subblock_overlap_floor = Tbase(0.8);

      auto init_burst_watch = [&]() {
        const auto C_now = get_orbitals();
        const auto F_now = get_fock_matrix();
        burst_occ = get_orbital_occupations();
        pseudo_canonicalise_(C_now, F_now, burst_occ, burst_C_pseudo, burst_eps);
        size_t nblk = burst_C_pseudo.size();
        burst_diff_occ_pairs.assign(nblk, {});
        burst_subblock_id.assign(nblk, IndexVector());
        for(size_t b = 0; b < nblk; b++) {
          if(burst_eps[b].size() == 0) continue;
          Index n_b = burst_eps[b].size();
          burst_subblock_id[b] = IndexVector::Zero(n_b);
          // Build sub-block ids by single-pass grouping on occupation.
          std::vector<bool> used(n_b, false);
          Index next_id = 0;
          for(Index i = 0; i < n_b; i++) {
            if(used[i]) continue;
            burst_subblock_id[b](i) = next_id;
            used[i] = true;
            for(Index j = i + 1; j < n_b; j++)
              if(!used[j] &&
                 std::abs(burst_occ[b](i) - burst_occ[b](j)) < occupation_change_threshold_) {
                burst_subblock_id[b](j) = next_id;
                used[j] = true;
              }
            next_id++;
          }
          // Differently-occupied pairs.
          for(Index i = 0; i < n_b; i++)
            for(Index j = i + 1; j < n_b; j++)
              if(burst_subblock_id[b](i) != burst_subblock_id[b](j)) {
                Tbase delta0 = burst_eps[b](i) - burst_eps[b](j);
                burst_diff_occ_pairs[b].emplace_back(i, j, delta0);
              }
        }
      };

      auto burst_watch_tripped = [&]() -> bool {
        if(burst_C_pseudo.empty()) return false;
        const auto C_now = get_orbitals();
        const auto F_now = get_fock_matrix();
        Orbitals<Torb> C_pseudo_now;
        OrbitalEnergies<Tbase> eps_now;
        pseudo_canonicalise_(C_now, F_now, burst_occ, C_pseudo_now, eps_now);

        // Tripwire 1: any differently-occupied pair's canonical-energy
        // gap has moved by more than ODA's degeneracy threshold.
        for(size_t b = 0; b < burst_diff_occ_pairs.size(); b++) {
          for(const auto & p : burst_diff_occ_pairs[b]) {
            Index i = std::get<0>(p);
            Index j = std::get<1>(p);
            Tbase delta0 = std::get<2>(p);
            Tbase deltat = eps_now[b](i) - eps_now[b](j);
            if(std::abs(deltat - delta0) > optimal_damping_degeneracy_threshold_) {
              if(verbosity_ >= 5)
                printf("Burst exit: block %zu orbitals %u, %u have a "
                       "canonical-energy gap shift %+e Eh (> threshold %e Eh).\n",
                       b, (unsigned) i, (unsigned) j,
                       deltat - delta0, optimal_damping_degeneracy_threshold_);
              return true;
            }
          }
        }

        // Tripwire 2: any orbital has lost majority overlap with its
        // initial equal-occupation sub-block.
        for(size_t b = 0; b < burst_C_pseudo.size(); b++) {
          if(burst_C_pseudo[b].cols() == 0) continue;
          Matrix<Torb> ovl = C_pseudo_now[b].adjoint() * burst_C_pseudo[b];
          Index n_b = burst_C_pseudo[b].cols();
          for(Index i = 0; i < n_b; i++) {
            Index sid = burst_subblock_id[b](i);
            Tbase span_w = 0;
            for(Index j = 0; j < n_b; j++)
              if(burst_subblock_id[b](j) == sid)
                span_w += std::norm(ovl(i, j));
            if(span_w < burst_subblock_overlap_floor) {
              if(verbosity_ >= 5)
                printf("Burst exit: block %zu orbital %u has sub-block "
                       "overlap %.3f < %.3f.\n",
                       b, (unsigned) i, span_w, burst_subblock_overlap_floor);
              return true;
            }
          }
        }
        return false;
      };

      auto clear_burst_watch = [&]() {
        burst_C_pseudo.clear();
        burst_eps.clear();
        burst_occ.clear();
        burst_diff_occ_pairs.clear();
        burst_subblock_id.clear();
      };
      // For termination when DIIS is not in the allowed set: track
      // whether ODA / CG have failed since the most recent successful
      // step. The loop exits when every allowed non-DIIS method has
      // failed in succession.
      bool oda_failed = false, rotation_failed = false;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        Tbase diis_error = norm(diis_error_vector(0));
        Tbase diis_max_error = diis_error_vector(0).template lpNorm<Eigen::Infinity>();
        Tbase dE = get_energy() - old_energy_;

        // Data to pass to callback function
        std::map<std::string, std::any> callback_data;
        callback_data["iter"] = iteration;
        callback_data["nfock"] = number_of_fock_evaluations_;
        callback_data["E"] = get_energy();
        callback_data["dE"] = get_energy() - old_energy_;
        callback_data["diis_error"] = diis_error;
        callback_data["diis_max_error"] = diis_max_error;

        if(verbosity_>=5) {
          printf("\n\n");
        }
        if(verbosity_>0) {
          printf("Iteration %i: %i Fock evaluations energy % .10f change % e DIIS error vector %s norm %e\n", (int) iteration, (int) number_of_fock_evaluations_, get_energy(), dE, error_norm_.c_str(), diis_error);
        }
        if(verbosity_>=5) {
          printf("History size %i\n",(int) orbital_history_.size());
        }
        if(converged()) {
          if(verbosity_)
            printf("Converged to energy % .10f!\n", get_energy());

          // Print out info
          callback_data["step"] = std::string("Converged");
          if(callback_function_)
            callback_function_(callback_data);
          break;
        }

        if(verbosity_>=5) {
          const auto occupations = get_orbital_occupations();
          auto occ_idx(occupied_orbitals(occupations));
          for(size_t l=0;l<occ_idx.size();l++) {
            if(occ_idx[l].size())
              std::cout << block_descriptions_[l] + " occupations: " << occupations[l].head(occ_idx[l].maxCoeff()+1).transpose() << std::endl;
          }
        }

        // Pre-step transition: bail out of DIIS if it is stalling or
        // the DIIS error is so large that A/EDIIS can't be trusted.
        // Only meaningful when at least one non-DIIS method is allowed.
        if(state == StepKind::DIIS && (allowed.oda || allowed.orbital_rotation())) {
          bool stalled =
            (diis_max_error >= optimal_damping_threshold_) ||
            (failed_iterations >= oda_restart_steps_);
          if(stalled) {
            StepKind next = pick_next({StepKind::ODA, StepKind::OrbitalRotation}, StepKind::DIIS);
            if(next != state) {
              if(verbosity_>=5) {
                const char * nname = next == StepKind::ODA ? "ODA"
                                   : (allowed.lbfgs ? "L-BFGS" : "CG");
                if(diis_max_error >= optimal_damping_threshold_)
                  printf("Switching DIIS -> %s: DIIS max error %e exceeds threshold %e\n",
                         nname, diis_max_error, optimal_damping_threshold_);
                else
                  printf("Switching DIIS -> %s: %i consecutive failed DIIS iterations\n",
                         nname, failed_iterations);
              }
              state = next;
            }
          }
        }

        old_energy_ = get_energy();

        if(state == StepKind::ODA) {
          if(verbosity_>=5) printf("Optimal damping step\n");
          callback_data["step"] = std::string("ODA");
          if(callback_function_)
            callback_function_(callback_data);
          bool oda_ok = optimal_damping_step();
          // Size the orbital-rotation burst: explicit orbital_rotation_steps_after_oda_ if the
          // user has set it, otherwise the count of orbital-rotation
          // DOFs at the new iterate that live inside a degenerate
          // group (sum_p sum_b sum_g N_g (K_g - N_g) at a polytope
          // vertex; up to K_g (K_g - 1)/2 at an interior point). One
          // orbital-rotation step is taken as a floor so a trivial polytope still
          // gets at least the Roothaan relaxation pass after each
          // ODA call.
          size_t orbital_rotation_burst = orbital_rotation_steps_after_oda_ > 0
                              ? orbital_rotation_steps_after_oda_
                              : std::max<size_t>(last_active_rotation_count_, 1);
          if(oda_ok) {
            failed_iterations = 0;
            oda_failed = rotation_failed = false;
            if(has_integer_occupations()) {
              // Polytope optimum sits on a vertex; hand back to DIIS
              // if available, else fall through the preference list.
              state = pick_next({StepKind::DIIS, StepKind::OrbitalRotation}, StepKind::ODA);
            } else {
              // Fractional polytope-interior optimum; relax the orbital
              // rotations (CG or L-BFGS) before DIIS gets its turn.
              state = pick_next({StepKind::OrbitalRotation, StepKind::DIIS}, StepKind::ODA);
            }
            orbital_rotation_steps_remaining = (state == StepKind::OrbitalRotation) ? orbital_rotation_burst : 0;
          } else {
            // Polytope minimum says we can't descend in occupation
            // space; try orbital rotations next, or DIIS if the
            // orbital-rotation branch isn't allowed.
            oda_failed = true;
            state = pick_next({StepKind::OrbitalRotation, StepKind::DIIS}, StepKind::ODA);
            orbital_rotation_steps_remaining = (state == StepKind::OrbitalRotation) ? orbital_rotation_burst : 0;
          }
          if(state == StepKind::OrbitalRotation && orbital_rotation_steps_remaining > 1)
            init_burst_watch();
          else
            clear_burst_watch();
        } else if(state == StepKind::OrbitalRotation) {
          // CG vs L-BFGS: prefer L-BFGS when it is allowed (limited-
          // memory quasi-Newton captures off-diagonal Hessian
          // information the diagonal preconditioner alone misses).
          bool use_lbfgs = allowed.lbfgs;
          if(verbosity_>=5) printf("%s step (%i remaining in burst)\n",
                                   use_lbfgs ? "L-BFGS" : "Scaled steepest descent",
                                   (int) orbital_rotation_steps_remaining);
          callback_data["step"] = std::string(use_lbfgs ? "LBFGS" : "CG");
          if(callback_function_)
            callback_function_(callback_data);
          bool rotation_ok = use_lbfgs ? lbfgs_step() : scaled_steepest_descent_step();
          if(rotation_ok) {
            failed_iterations = 0;
            oda_failed = rotation_failed = false;
          } else {
            rotation_failed = true;
          }
          if(orbital_rotation_steps_remaining > 0)
            orbital_rotation_steps_remaining--;
          // Stay in the orbital-rotation state only if we still owe
          // steps from the last ODA AND the line search just succeeded
          // AND the burst-watch has not flagged a level crossing or
          // qualitative orbital change. A failed line search means
          // there is no more descent in the orbital-rotation subspace
          // at the current occupations, and the next step would just
          // fail too; a tripped watch means the occupations themselves
          // are no longer the right Aufbau filling and ODA must be
          // re-consulted. In either case hand back through the
          // preference list (DIIS first, then ODA).
          bool burst_tripped =
            (orbital_rotation_steps_remaining > 0 && rotation_ok)
              ? burst_watch_tripped()
              : false;
          if(orbital_rotation_steps_remaining > 0 && rotation_ok && !burst_tripped) {
            state = StepKind::OrbitalRotation;
          } else {
            orbital_rotation_steps_remaining = 0;
            clear_burst_watch();
            state = pick_next({StepKind::DIIS, StepKind::ODA}, StepKind::OrbitalRotation);
          }
        } else {
          // DIIS step. Compute mixing factor (Garza and Scuseria, 2012).
          Tbase aediis_coeff;
          if(diis_error < diis_threshold_) {
            aediis_coeff = 0.0;
          } else if(diis_error < diis_epsilon_) {
            aediis_coeff = (diis_error-diis_threshold_)/(diis_epsilon_-diis_threshold_);
          } else {
            aediis_coeff = 1.0;
          }
          Vector<Tbase> weights;
          std::string step;
          std::tie(weights, step) = minimal_error_sampling_algorithm_weights(aediis_coeff);
          if(verbosity_>=5)
            printf("%s step\n",step.c_str());
          if(verbosity_>=10)
            std::cout << "Extrapolation weights: " << weights.transpose() << std::endl;

          callback_data["step"] = step;
          if(callback_function_)
            callback_function_(callback_data);

          if(!attempt_extrapolation(weights)) {
            if(verbosity_>=10) printf("Warning: did not go down in energy!\n");
            failed_iterations++;
          } else {
            failed_iterations=0;
            oda_failed = rotation_failed = false;
          }
          // Stay in DIIS; the pre-step check at the top of the next
          // iteration will move us to ODA / CG if DIIS keeps stalling.
          state = StepKind::DIIS;
        }

        // Early termination: if DIIS is not in the allowed set, exit
        // as soon as every allowed non-DIIS method has failed since
        // the last successful step. With DIIS available, DIIS keeps
        // retrying until ``maximum_iterations_`` and the loop never
        // exits early.
        if(!allowed.diis) {
          bool all_failed =
            (!allowed.oda || oda_failed) &&
            (!allowed.orbital_rotation() || rotation_failed);
          if(all_failed) {
            if(verbosity_>0) {
              printf("All allowed SCF methods failed at iteration %i; stopping with DIIS error vector %s norm %e.\n",
                     (int) iteration, error_norm_.c_str(), diis_error);
            }
            callback_data["step"] = std::string("Stalled");
            if(callback_function_)
              callback_function_(callback_data);
            break;
          }
        }
        // Do cleanup
        cleanup();
      }
    }

    DensityMatrix<Torb, Tbase> get_solution(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]);
    }

    Orbitals<Torb> get_orbitals(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).first;
    }

    OrbitalOccupations<Tbase> get_orbital_occupations(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).second;
    }

    FockBuilderReturn<Torb, Tbase> get_fock_build(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]);
    }

    FockMatrix<Torb> get_fock_matrix(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]).second;
    }


    void brute_force_search_for_lowest_configuration() {
      // Make sure we have a solution
      if(orbital_history_.size() == 0)
        run();
      else {
        Tbase diis_error = norm(diis_error_vector(0));
        if(diis_error >= diis_threshold_)
          run();
      }

      // Get the reference orbitals and orbital occupations
      auto reference_solution = orbital_history_[0];
      auto reference_orbitals = get_orbitals();
      auto reference_occupations = get_orbital_occupations();
      auto reference_energy = get_energy();
      auto reference_fock = get_fock_matrix();

      // We also need the orbital energies below
      auto diagonalized_fock = compute_orbitals(reference_fock);
      const auto & orbital_energies = diagonalized_fock.second;

      // The brute-force search sweeps occupations, so it must thaw
      // and silence for the duration -- save the caller's settings and
      // restore them on return through the RAII guard.
      struct SettingsGuard {
        int * verb; int old_verb;
        bool * frozen; bool old_frozen;
        ~SettingsGuard() { *verb = old_verb; *frozen = old_frozen; }
      };
      SettingsGuard guard{&verbosity_, verbosity_,
                          &frozen_occupations_, frozen_occupations_};
      verbosity_ = 0;
      frozen_occupations_ = false;
      while(true) {
        // Count the number of particles in each block
        Vector<Tbase> number_of_particles_per_block = Vector<Tbase>::Zero(number_of_blocks_);
        for(Index iblock=0; iblock<number_of_particles_per_block.size(); iblock++) {
          if(empty_block(iblock))
            continue;
          number_of_particles_per_block[iblock] = reference_occupations[iblock].sum();
        }
        std::cout << "Number of particles per block: " << number_of_particles_per_block.transpose() << std::endl;

        // List of occupations and resulting energies
        std::vector<std::pair<Vector<Tbase>,Tbase>> list_of_energies;

        // Loop over particle types. We have a double loop, since finding the lowest state in UHF probably requires this
        for(Index iparticle=0; iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
          size_t iblock_start = particle_block_offset(iparticle);
          size_t iblock_end = iblock_start + number_of_blocks_per_particle_type_(iparticle);

          // One-particle moves
          for(size_t iblock_source = iblock_start; iblock_source < iblock_end; iblock_source++)
            for(size_t iblock_target = iblock_start; iblock_target < iblock_end; iblock_target++) {
              if(iblock_source == iblock_target)
                continue;

              // Maximum number to move
              Tbase num_i_source = number_of_particles_per_block[iblock_source];
              Tbase i_target_capacity = reference_occupations[iblock_target].size()*maximum_occupation_[iblock_target];
              Tbase i_target_capacity_left = i_target_capacity - reference_occupations[iblock_target].sum();
              int num_i_max = std::ceil(std::min(num_i_source, i_target_capacity_left));
              num_i_max = std::min(num_i_max, (int) std::round(std::min(maximum_occupation_[iblock_source], maximum_occupation_[iblock_target])));

              // Generate trials by moving particles
              for(int imove=1; imove<=num_i_max; imove++) {
                // Modify the occupations
                auto trial_number(number_of_particles_per_block);
                Tbase i_moved = std::min((Tbase) imove, trial_number(iblock_source));
                trial_number(iblock_source) -= i_moved;
                trial_number(iblock_target) += i_moved;

                if(trial_number(iblock_source) < 0.0 or trial_number(iblock_target) > i_target_capacity)
                  continue;

                fixed_number_of_particles_per_block_ = trial_number;

                printf("isource = %i itarget = %i imoved = %f\n", (int)iblock_source, (int)iblock_target, i_moved);
                std::cout << "trial number of particles: " << trial_number.transpose() << std::endl;
                fflush(stdout);

                // Determine full orbital occupations from the specified data. Because we've fixed the number of particles in each block, it doesn't matter that the orbital energies aren't correct
                auto trial_occupations = update_occupations(orbital_energies);
                initialize_with_orbitals(reference_orbitals, trial_occupations);
                try {
                  run();
                } catch(...) {};
                // Add the result to the list
                list_of_energies.push_back(std::make_pair(trial_number, get_energy()));
                // Reset the restriction
                Vector<Tbase> dummy;
                fixed_number_of_particles_per_block_ = dummy;
              }
            }

          for(Index jparticle=0; jparticle<=iparticle; jparticle++) {
            size_t jblock_start = particle_block_offset(jparticle);
            size_t jblock_end = jblock_start + number_of_blocks_per_particle_type_(jparticle);

            // Loop over blocks of particles
            for(size_t iblock_source = iblock_start; iblock_source < iblock_end; iblock_source++)
              for(size_t iblock_target = iblock_start; iblock_target < iblock_end; iblock_target++) {

                bool same_particle = (iparticle == jparticle);
                size_t jblock_source_end = same_particle ? iblock_source+1 : jblock_end;
                size_t jblock_target_end = same_particle ? iblock_target+1 : jblock_end;
                printf("iparticle= %i jparticle= %i isource=%i itarget=%i\n",(int)iparticle,(int)jparticle,(int)iblock_source,(int)iblock_target);

                for(size_t jblock_source = jblock_start; jblock_source < jblock_source_end; jblock_source++)
                  for(size_t jblock_target = jblock_start; jblock_target < jblock_target_end; jblock_target++) {
                    // Skip trivial cases
                    if(iblock_source == iblock_target and jblock_source == jblock_target)
                      continue;
                    if(iblock_source == jblock_target and jblock_source == iblock_target)
                      continue;
                    // Skip one-particle cases
                    if(iblock_source == jblock_source and iblock_target == jblock_target)
                      continue;

                    // Maximum number to move
                    Tbase num_i_source = number_of_particles_per_block[iblock_source];
                    Tbase i_target_capacity = reference_occupations[iblock_target].size()*maximum_occupation_[iblock_target];
                    Tbase i_target_capacity_left = i_target_capacity - reference_occupations[iblock_target].sum();
                    int num_i_max = std::ceil(std::min(num_i_source, i_target_capacity_left));
                    num_i_max = std::min(num_i_max, (int) std::round(std::min(maximum_occupation_[iblock_source], maximum_occupation_[iblock_target])));

                    Tbase num_j_source = number_of_particles_per_block[jblock_source];
                    Tbase j_target_capacity = reference_occupations[jblock_target].size()*maximum_occupation_[jblock_target];
                    Tbase j_target_capacity_left = j_target_capacity - reference_occupations[jblock_target].sum();
                    int num_j_max = std::ceil(std::min(num_j_source, j_target_capacity_left));
                    num_j_max = std::min(num_j_max, (int) std::round(std::min(maximum_occupation_[jblock_source], maximum_occupation_[jblock_target])));

                    printf("i: source %f capacity left %f num max %i\n",num_i_source,i_target_capacity_left,num_i_max);
                    printf("j: source %f capacity left %f num max %i\n",num_j_source,j_target_capacity_left,num_j_max);
                    fflush(stdout);

                    // Generate trials by moving particles
                    for(int imove=1; imove<=num_i_max; imove++)
                      for(int jmove=1; jmove<=num_j_max; jmove++) {
                        // These also lead to degeneracies
                        if(iblock_source == iblock_target and imove > 0)
                          continue;
                        if(iblock_source == iblock_target and jmove == 0)
                          continue;
                        if(jblock_source == jblock_target and jmove > 0)
                          continue;
                        if(jblock_source == jblock_target and imove == 0)
                          continue;

                        // Modify the occupations
                        auto trial_number(number_of_particles_per_block);
                        Tbase i_moved = std::min((Tbase) imove, trial_number(iblock_source));
                        trial_number(iblock_source) -= i_moved;
                        trial_number(iblock_target) += i_moved;
                        Tbase j_moved = std::min((Tbase) jmove, trial_number(jblock_source));
                        trial_number(jblock_source) -= j_moved;
                        trial_number(jblock_target) += j_moved;

                        if(trial_number(iblock_source) < 0.0 or trial_number(jblock_source) < 0.0)
                          continue;
                        if(trial_number(iblock_target) > i_target_capacity)
                          continue;
                        if(trial_number(jblock_target) > j_target_capacity)
                          continue;

                        fixed_number_of_particles_per_block_ = trial_number;

                        printf("isource = %i itarget = %i imoved = %f\n", (int)iblock_source, (int)iblock_target, i_moved);
                        printf("jsource = %i jtarget = %i jmoved = %f\n", (int)jblock_source, (int)jblock_target, j_moved);
                        std::cout << "trial number of particles: " << trial_number.transpose() << std::endl;
                        fflush(stdout);

                        // Determine full orbital occupations from the specified data. Because we've fixed the number of particles in each block, it doesn't matter that the orbital energies aren't correct
                        auto trial_occupations = update_occupations(orbital_energies);
                        initialize_with_orbitals(reference_orbitals, trial_occupations);
                        try {
                          run();
                        } catch(...) {};
                        // Add the result to the list
                        list_of_energies.push_back(std::make_pair(trial_number, get_energy()));
                        // Reset the restriction
                        Vector<Tbase> dummy;
                        fixed_number_of_particles_per_block_ = dummy;
                      }
                  }
              }
          }
        }

        // Sort the list in ascending order
        std::sort(list_of_energies.begin(), list_of_energies.end(), [](const std::pair<Vector<Tbase>,Tbase> & a, const std::pair<Vector<Tbase>,Tbase> & b) {return a.second < b.second;});

        printf("Configurations\n");
        for(size_t iconf=0;iconf<list_of_energies.size();iconf++) {
          printf("%4i E= % .10f with occupations\n",(int) iconf, list_of_energies[iconf].second);
          std::cout << list_of_energies[iconf].first.transpose() << std::endl;
        }

        if(list_of_energies[0].second < reference_energy) {
          printf("Energy changed by %e by improved reference\n", list_of_energies[0].second - reference_energy);

          // Update the reference
          fixed_number_of_particles_per_block_ = list_of_energies[0].first;
          auto trial_occupations = update_occupations(orbital_energies);
          initialize_with_orbitals(reference_orbitals, trial_occupations);
          run();

          reference_solution = orbital_history_[0];
          reference_orbitals = get_orbitals();
          reference_occupations = get_orbital_occupations();
          reference_energy = get_energy();
          reference_fock = get_fock_matrix();
        } else {
          // Restore the reference calculation
          initialize_with_orbitals(reference_orbitals, reference_occupations);
          run();
          printf("Search converged!\n");
          break;
        }
      }
    }

    void callback_function(std::function<void(const std::map<std::string,std::any> &)> callback_function = nullptr) {
      callback_function_ = callback_function;
    }

    void callback_convergence_function(std::function<bool(const std::map<std::string,std::any> &)> callback_convergence_function = nullptr) {
      callback_convergence_function_ = callback_convergence_function;
    }
  };
}
```


