

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
#include "cg_optimizer.hpp"

#include <any>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace OpenOrbitalOptimizer {

  template<typename Tbase, bool IsComplex> class SCFSolver {
  public:
    using Torb = OrbitalScalar<Tbase, IsComplex>;
  private:
    /* Input data section */
    IndexVector number_of_blocks_per_particle_type_;
    Vector<Tbase> maximum_occupation_;
    Vector<Tbase> number_of_particles_;
    FockBuilder<Torb, Tbase> fock_builder_;
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
    std::string error_norm_ = "rms";

    Tbase minimal_gradient_projection_ = 1e-4;
    Tbase occupied_threshold_ = 1e-6;
    Tbase initial_level_shift_ = 1.0;
    Tbase level_shift_factor_ = 2.0;

    Tbase old_energy_ = 0.0;

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
      return orbitals * (occupations).asDiagonal() * (orbitals).adjoint();
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
        v.segment(ioff, vectors[iblock].size()) = vectors[iblock];
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
        if(dim[iblock]==0)
          continue;
        size_t sz = (size_t)dim[iblock] * (size_t)dim[iblock];
        if constexpr (Eigen::NumTraits<Torb>::IsComplex) {
          sz *= 2;
        }
        mat[iblock] = matricise(vec.segment(ioff, sz), dim[iblock], dim[iblock]);
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
      Index ioff = 0;
      for(auto & block: error_vectors) {
        if(block.size()>0) {
          return_vector.segment(ioff, block.size()) = block;
          ioff += block.size();
        }
      }
      if(ioff != static_cast<Index>(nelem))
        throw std::logic_error("Indexing error!\n");

      return return_vector;
    }

    Tbase diis_error_matrix_element(size_t ihist, size_t jhist) const {
      Tbase el=0.0;
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        Vector<Tbase> ei(diis_error_vector(ihist, iblock));
        Vector<Tbase> ej(diis_error_vector(jhist, iblock));
        el += (ei).dot(ej);
      }
      return el;
    }

    Matrix<Tbase> diis_error_matrix(const std::vector<size_t> & mask) const {
      // The error matrix is given by the orbital gradient dot products
      const size_t N=mask.size();
      Matrix<Tbase> B = Matrix<Tbase>::Zero(N, N);

      for(size_t ihist=0; ihist<N; ihist++) {
        for(size_t jhist=0; jhist<=ihist; jhist++) {
          B(ihist, jhist) = B(jhist, ihist) = diis_error_matrix_element(mask[ihist], mask[jhist]);
        }
      }
      return B;
    }

    Vector<Tbase> diis_error_matrix_diagonal() const {
      Vector<Tbase> B = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t ihist=0; ihist<(size_t)B.size(); ihist++) {
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
      for(size_t i=0;i<residuals.size();i++)
        residuals(i) = diis_error_matrix_element(history_mask[i], history_mask[i]);
      Tbase min_residual = (residuals).minCoeff();
      for(size_t i=history_mask.size()-1;i<history_mask.size();i--)
        // Criterion from Chupin et al, 2012
        if(residuals(i)*diis_restart_factor_ > min_residual)
          history_mask.erase(history_mask.begin()+i);
      size_t nrestart = orbital_history_.size()-history_mask.size();
      if(verbosity_>=10 and nrestart>0)
        printf("Removed %i entries corresponding to large DIIS errors\n", (int) nrestart);

      // Set up the DIIS error matrix
      const size_t N=history_mask.size();
      Matrix<Tbase> B = Matrix<Tbase>::Constant(N+1, N+1, -1.0);
      B.block(0, 0, (N-1)-(0)+1, (N-1)-(0)+1)=diis_error_matrix(history_mask);
      B(N,N)=0.0;

      // Apply the diagonal damping
      B.block(0, 0, (N-1)-(0)+1, (N-1)-(0)+1).diagonal() *= 1.0+diis_diagonal_damping_;

      // To improve numerical conditioning, scale entries of error
      // matrix such that the last diagonal element is one; Eckert et
      // al, J. Comput. Chem 18. 1473-1483 (1997)
      Vector<Tbase> Bdiag((B).diagonal());
      Tbase diagmin = (Bdiag.segment(0, (N-1)-(0)+1).minCoeff());
      if(diagmin != 0.0)
        B.block(0, 0, (N-1)-(0)+1, (N-1)-(0)+1) /= diagmin;

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
      xguess.push_back(Vector<Tbase>::Constant(b.size(), 1.0/b.size()));
      // "Gauss" points
      for(size_t i=0;i<b.size();i++) {
        Vector<Tbase> xtr = Vector<Tbase>::Constant(b.size(), Tbase(1)/(b.size()+2));
        xtr(i) *= 3;
        xguess.push_back(xtr);
      }
      // End points
      for(size_t i=0;i<b.size();i++) {
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
      //std::cout << "Initial x" << ": " << x.transpose() << std::endl;

      Matrix<Tbase> search_directions = Matrix<Tbase>::Identity(b.size(), b.size());

      auto current_point = fx(x);
      auto old_x = x;

      // Powell algorithm
      for(size_t imacro=0; imacro<max_iter; imacro++) {
        Tbase curval(current_point);

        for(size_t i=0; i<b.size(); i++) {
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

        //std::cout << "x" << ": " << x.transpose() << std::endl;
        if(dE > -df_tol) {
          if(verbosity_ >= 10) {
            printf("A/EDIIS weights converged in %i macroiterations\n",(int) imacro);
            //std::cout << "xconv" << ": " << x.transpose() << std::endl;
          }
          break;
        } else if(imacro==max_iter-1) {
          if(verbosity_ >= 10) {
            printf("A/EDIIS weights did not converge in %i macroiterations, dE=%e\n", (int) imacro, dE);
            //std::cout << "xfinal" << ": " << x.transpose() << std::endl;
          }
        }

        /*
        // Rotate search directions. Generate a random ordering of the columns
        IndexVector randperm(arma::randperm(search_directions.cols()));
        search_directions=search_directions.cols(randperm);
        // Mix the vectors together
        for(size_t i=0;i<search_directions.cols();i++)
          for(size_t j=0;j<i;j++) {
            Vector<Tbase> randu(1);
            randu.randu();

            Vector<Tbase> newi = (1-randu(0))*search_directions.col(i) + randu(0)*search_directions.col(j);
            Vector<Tbase> newj = (1-randu(0))*search_directions.col(j) + randu(0)*search_directions.col(i);
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
        for(size_t i=0; i<b.size(); i++) {
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
        //std::cout << "Using suboptimal solution instead" << ": " << x.transpose() << std::endl;
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
        for(size_t ihist=0;ihist<ret.size();ihist++) {
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
      if(verbosity_>=10) std::cout << "DIIS weights" << ": " << diis_w.transpose() << std::endl;
      if(aediis_coeff == 0.0) {
        std::string step = "DIIS";
        return std::make_tuple(diis_w,step);
      }

      // Get various extrapolation weights
      const size_t N = orbital_history_.size();
      Vector<Tbase> adiis_w(adiis_weights());
      if(verbosity_>=10) std::cout << "ADIIS weights" << ": " << adiis_w.transpose() << std::endl;
      Vector<Tbase> ediis_w(ediis_weights());
      if(verbosity_>=10) std::cout << "EDIIS weights" << ": " << ediis_w.transpose() << std::endl;

      // Candidates
      Matrix<Tbase> candidate_w = Matrix<Tbase>::Zero(N, 2);
      size_t icol=0;
      candidate_w.col(icol++) = adiis_w;
      candidate_w.col(icol++) = ediis_w;
      const std::vector<std::string> weight_legend({"ADIIS", "EDIIS"});
      std::string step;

      Vector<Tbase> density_projections = Vector<Tbase>::Zero(candidate_w.cols());
      for(size_t iw=0;iw<candidate_w.cols();iw++) {
        density_projections(iw) = density_projection(candidate_w.col(iw));
      }
      if(verbosity_>=10)
        std::cout << "Density projections" << ": " << density_projections.transpose() << std::endl;

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
      if(weights.size() != orbital_history_.size()) {
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
      if(weights.size() != orbital_history_.size()) {
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
        const Tbase zero_tol = 10*maximum_occupation_[iblock]*std::numeric_limits<Tbase>::epsilon();
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
        Matrix<Tbase> orbital_projections = (C_new[iblock].adjoint() * C_reference[iblock]).array().abs().matrix();

        // Occupy the orbitals in ascending energy, especially if there are unoccupied orbitals in-between
        for(size_t iorb=0; iorb<reference_occupations[iblock].size(); iorb++) {
          // Projections for this orbital
          Vector<Tbase> projection = orbital_projections.col(iorb);
          // Find the maximum index
          Index maximal_projection_index;
          Tbase maximal_projection = projection.maxCoeff(&maximal_projection_index);
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
        Matrix<Torb> Pl(lC*(lo).asDiagonal()*lC.adjoint());
        Matrix<Torb> Pr(rC*(ro).asDiagonal()*rC.adjoint());
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

    bool optimal_damping_step() {
      // Diagonalize the best Fock matrix
      auto diagonalized_fock = compute_orbitals(std::get<1>(orbital_history_[0]).second);
      auto new_orbitals = diagonalized_fock.first;
      auto new_orbital_energies = diagonalized_fock.second;
      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      // Form the new density matrix
      std::vector<Matrix<Torb>> dm_new(new_orbitals.size());
      for(size_t iblock=0; iblock<new_orbitals.size(); iblock++) {
        if(new_orbitals[iblock].cols() == 0)
          continue;
        dm_new[iblock] = new_orbitals[iblock] * (new_occupations[iblock]).asDiagonal() * (new_orbitals[iblock]).adjoint();
      }

      // Compute the energy gradient for each particle type for the density matrix mixing: P -> (1-lambda)*Pcurrent + lambda*Pnew
      size_t nparticles = number_of_blocks_per_particle_type_.size();
      Vector<Tbase> dEdlambda = Vector<Tbase>::Zero(nparticles);
      for(size_t iparticle=0;iparticle<nparticles;iparticle++) {
        size_t block_offset = particle_block_offset(iparticle);
        for(size_t iblock=block_offset;iblock<block_offset+number_of_blocks_per_particle_type_(iparticle);iblock++) {
          if(empty_block(iblock))
            continue;
          // Current density matrix
          Matrix<Torb> fock_current(get_fock_matrix_block(0, iblock));
          Matrix<Torb> dm_current(get_density_matrix_block(0, iblock));
          dEdlambda(iparticle) += std::real((fock_current*(dm_new[iblock] - dm_current)).trace());
        }
      }
      if(verbosity_>=10)
        std::cout << "Optimal damping: dE/dlambda" << ": " << dEdlambda.transpose() << std::endl;

      // Search direction is therefore
      Vector<Tbase> search_direction = -dEdlambda;
      // As we start the search from the current density matrix,
      // lambda=0 at the outset and we set any negative directions as
      // invalid
      IndexVector negative_indices = find_indices_where(search_direction,
        [](Tbase v){return v < Tbase{0};});
      for(Index k=0;k<negative_indices.size();k++)
        search_direction[negative_indices[k]] = Tbase{0};

      IndexVector valid_directions = find_indices_where(search_direction,
        [](Tbase v){return v != Tbase{0};});
      if(valid_directions.size()==0) {
        // No valid search directions!
        return false;
      }

      // The resulting trial is therefore the step that takes us to
      // the edge
      Vector<Tbase> lambda_trial = search_direction/(search_direction).maxCoeff();

      // Helper function
      std::function<DensityMatrix<Torb, Tbase>(const Vector<Tbase> &)> interpolate_dm = [&](const Vector<Tbase> & step) {
        Orbitals<Torb> new_orbs(number_of_blocks_);
        OrbitalOccupations<Tbase> new_occs(number_of_blocks_);
        for(size_t iparticle=0;iparticle<nparticles;iparticle++) {
          size_t block_offset = particle_block_offset(iparticle);
          for(size_t iblock=block_offset;iblock<block_offset+number_of_blocks_per_particle_type_(iparticle);iblock++) {
            if(empty_block(iblock))
              continue;
            Matrix<Torb> dm_block((1-step(iparticle))*get_density_matrix_block(0, iblock) + step(iparticle)*dm_new[iblock]);
            // Flip the sign so that the orbitals come in increasing occupation
            Matrix<Torb> neg_dm = -dm_block;
            Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(neg_dm);
            new_occs[iblock] = es.eigenvalues();
            new_orbs[iblock] = es.eigenvectors();
            new_occs[iblock] *= Tbase{-1};
            // Zero out numerically zero occupations
            const Tbase ztol = 10*maximum_occupation_[iblock]*std::numeric_limits<Tbase>::epsilon();
            for(Index k=0; k<new_occs[iblock].size(); k++) {
              if(std::abs(new_occs[iblock][k]) <= ztol)
                new_occs[iblock][k] = Tbase{0};
            }
          }
        }
        return std::make_pair(new_orbs, new_occs);
      };

      // Evaluate the energy with the trial density
      if(add_entry(interpolate_dm(lambda_trial)))
        // We already went down in energy, great!
        return true;

      // If we are here, we need to interpolate. Since we already
      // handled the case that the full step decreased the energy, we
      // know that the new step is the first in the stack since that
      // is how it is sorted. Energies are
      Tbase E0 = std::get<1>(orbital_history_[0]).first;
      Tbase E1 = std::get<1>(orbital_history_[1]).first;
      // and the gradients along the path are
      Tbase dE0 = (dEdlambda).dot( lambda_trial);

      Vector<Tbase> dEdlambda2 = Vector<Tbase>::Zero(nparticles);
      for(size_t iparticle=0;iparticle<nparticles;iparticle++) {
        size_t block_offset = particle_block_offset(iparticle);
        for(size_t iblock=block_offset;iblock<block_offset+number_of_blocks_per_particle_type_(iparticle);iblock++) {
          if(empty_block(iblock))
            continue;
          // Current density matrix
          Matrix<Torb> fock_new(get_fock_matrix_block(1, iblock));
          Matrix<Torb> dm_current(get_density_matrix_block(0, iblock));
          dEdlambda2(iparticle) += std::real((fock_new*(dm_new[iblock] - dm_current)).trace());
        }
      }
      Tbase dE1 = (dEdlambda2).dot( lambda_trial);

      // Fit cubic
      Tbase d = E0;
      Tbase c = dE0;
      Tbase b = -2*dE0 - 3*E0 + 3*E1 - dE1;
      Tbase a = dE0 + 2*E0 - 2*E1 + dE1;
      std::function<Tbase(Tbase)> eval_poly = [a,b,c,d](Tbase x) {
        return (((a*x+b)*x+c)*x)+d;
      };

      // Convert to derivative
      a *= 3;
      b *= 2;
      std::function<Tbase(Tbase)> eval_deriv = [a,b,c,d](Tbase x) {
        return (a*x+b)*x+c;
      };

      // Solve roots
      Tbase x1 = (-b - std::sqrt(b*b - 4*a*c))/(2*a);
      Tbase x2 = (-b + std::sqrt(b*b - 4*a*c))/(2*a);
      bool x1ok = x1 > 0.0 and x1<=1.0;
      bool x2ok = x2 > 0.0 and x2<=1.0;

      Tbase opt_step;
      if(x1ok and x2ok) {
        opt_step = eval_poly(x1) < eval_poly(x2) ? x1 : x2;
      } else if(x1ok) {
        opt_step = x1;
      } else if(x2ok) {
        opt_step = x2;
      } else {
        // No allowable solution!
        return false;
      }
      if(verbosity_>=10)
        printf("Optimal damping factor %e, predicted energy change %e\n",opt_step,eval_poly(opt_step)-eval_poly(0.0));

      // Mix the density matrices
      return add_entry(interpolate_dm(opt_step*lambda_trial));
    }

    void cleanup() {
      Vector<Tbase> density_differences = Vector<Tbase>::Zero(orbital_history_.size()-1);
      for(size_t ihist=1;ihist<orbital_history_.size();ihist++) {
        density_differences(ihist-1)=density_matrix_difference(ihist, 0);
      }
      if(verbosity_ >= 10) {
        std::cout << "Density differences" << ": " << density_differences.transpose() << std::endl;
      } else if(verbosity_>=5) {
        printf("Density matrix difference %e between lowest-energy and newest entry\n",density_differences(0));
      }

      // Sort the differences
      IndexVector idx = sort_index_ascending(density_differences);
      // Pick the indices that don't satisfy the criterion
      Tbase ref = density_differences[idx[0]];
      std::vector<Index> to_remove;
      for(Index k=0; k<idx.size(); k++) {
        if(density_restart_factor_*density_differences[idx[k]] > ref)
          to_remove.push_back(idx[k]);
      }
      if(!to_remove.empty()) {
        std::sort(to_remove.begin(), to_remove.end(), std::greater<Index>{});
        if(verbosity_>=10)
          printf("Removing %i entries corresponding to large change in density matrix\n",(int) to_remove.size());
        for(Index ihistm1: to_remove) {
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
        IndexVector occupied_indices = find_indices_where(reference_occupations[iblock], [](Tbase v){return v > Tbase{0};});
        for(size_t io1 = 0; io1 < occupied_indices.size(); io1++)
          for(size_t io2 = 0; io2 < io1; io2++) {
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
        IndexVector occupied_indices = find_indices_where(reference_occupations[iblock], [](Tbase v){return v > Tbase{0};});
        IndexVector virtual_indices = find_indices_where(reference_occupations[iblock], [](Tbase v){return v == Tbase{0};});
        for(auto o: occupied_indices)
          for(auto v: virtual_indices)
            dofs.push_back(std::make_tuple(iblock, o, v));
      }

      return dofs;
    }

    Vector<Tbase> orbital_gradient_vector() const {
      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      Vector<Tbase> orb_grad;

      if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
        orb_grad.setZero(dof_list.size());
      } else {
        orb_grad.setZero(2*dof_list.size());
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
        if constexpr (!!Eigen::NumTraits<Torb>::IsComplex) {
          orb_grad(dof_list.size() + idof) = 2*std::imag(fock_mo(iorb,jorb))*(occ_block(jorb)-occ_block(iorb));
        }
      }

      if(orb_grad.hasNaN())
        throw std::logic_error("Orbital gradient has NaNs");

      return orb_grad;
    }

    Vector<Tbase> diagonal_orbital_hessian() const {
      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      Vector<Tbase> orb_hess;

      if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
        orb_hess.setZero(dof_list.size());
      } else {
        orb_hess.setZero(2*dof_list.size());
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
        if constexpr (!!Eigen::NumTraits<Torb>::IsComplex) {
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
      positive_hessian += (-(diagonal_hessian).minCoeff()+shift)*Vector<Tbase>::Ones(positive_hessian.size());

      Tbase normalized_projection;
      Tbase maximum_spread = (positive_hessian).maxCoeff();
      Vector<Tbase> preconditioned_direction;
      while(true) {
        // Normalize the largest values
        Vector<Tbase> normalized_hessian(positive_hessian);
        IndexVector idx(find_indices_where(normalized_hessian, [&](Tbase v){return v > maximum_spread;}));
        normalized_hessian(idx) = maximum_spread*Vector<Tbase>::Ones(idx.size());

        // and divide the gradient by its square root
        preconditioned_direction = (gradient.array() / normalized_hessian.array().sqrt()).matrix();
        if(preconditioned_direction.hasNaN())
          throw std::logic_error("Preconditioned search direction has NaNs");

        normalized_projection = (preconditioned_direction).dot( gradient) / std::sqrt((preconditioned_direction).norm()*(gradient).norm());
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
        kappa[iblock].setZero(reference_orbitals[iblock].cols(), reference_orbitals[iblock].cols());
        for(auto dof: blocked_dof[iblock]) {
          auto iorb = std::get<0>(dof);
          auto jorb = std::get<1>(dof);
          auto idof = std::get<2>(dof);
          kappa[iblock](iorb,jorb) = x(idof);
        }
        // imaginary parameters
        if constexpr (!!Eigen::NumTraits<Torb>::IsComplex) {
          for(auto dof: blocked_dof[iblock]) {
            auto iorb = std::get<0>(dof);
            auto jorb = std::get<1>(dof);
            auto idof = std::get<2>(dof);
            kappa[iblock](iorb,jorb) += Torb(0.0,x(dof_list.size()+idof));
          }
        }
        // Antisymmetrize
        kappa[iblock] -= (kappa[iblock]).adjoint();
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

        // Do eigendecomposition of -i*kappa (Hermitian) -> real evals,
        // complex evecs. Then exp(kappa) = evec * diag(exp(i*eval)) * evec^H.
        Matrix<std::complex<Tbase>> kappa_imag =
            kappa[iblock].template cast<std::complex<Tbase>>() *
            std::complex<Tbase>(Tbase{0}, Tbase{-1});
        Eigen::SelfAdjointEigenSolver<Matrix<std::complex<Tbase>>> es(kappa_imag);
        Vector<Tbase> eval = es.eigenvalues();
        Matrix<std::complex<Tbase>> evec = es.eigenvectors();
        // Build exp(i*eval) as a complex diagonal
        Vector<std::complex<Tbase>> exp_diag(eval.size());
        for(Index k=0; k<eval.size(); ++k)
          exp_diag[k] = std::exp(std::complex<Tbase>(Tbase{0}, eval[k]));
        Matrix<std::complex<Tbase>> expkappa_imag = evec * exp_diag.asDiagonal() * evec.adjoint();
        if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
          expkappa = expkappa_imag.real();
        } else {
          expkappa = expkappa_imag;
        }

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

          shifted_fock[iblock] += level_shift *(orbitals * (fractional_occupations).asDiagonal() * orbitals.adjoint());
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
    void steepest_descent_step() {
      // Reference energy
      auto reference_energy = get_energy();

      // Get the orbital gradient
      auto gradient = orbital_gradient_vector();
      // and the diagonal Hessian
      auto diagonal_hessian = diagonal_orbital_hessian();

      // Precondition search direction
      auto search_direction = precondition_search_direction(-gradient, diagonal_hessian);

      // Ensure that the search direction is down-hill
      if((search_direction).dot( gradient) >= 0.0) {
        throw std::logic_error("Search direction is not down-hill?\n");
      }

      // Helper to evaluate steps
      std::function<Tbase(Tbase)> evaluate_step = [this, search_direction](Tbase length){
        Tbase reference_energy(get_energy());
        if(length==0.0)
          // We just get the reference energy
          return reference_energy;
        auto p(search_direction*length);
        auto entry = evaluate_rotation(p);
        if(length!=0.0)
          add_entry(std::get<0>(entry), std::get<1>(entry));
        if(verbosity_>=5)
          printf("Evaluated step %e with energy %.10f change from reference %e\n", length, std::get<1>(entry).first, std::get<1>(entry).first-reference_energy);
        return std::get<1>(entry).first;
      };
      std::function<Tbase(Tbase)> scan_step = [this, search_direction](Tbase length){
        auto p(search_direction*length);
        auto entry = evaluate_rotation(p);
        return std::get<1>(entry).first;
      };

      // Determine the maximal step size
      Tbase Tmu = maximum_rotation_step(search_direction);
      // This step is a whole quasiperiod. Since we are going downhill,
      // the minimum would be at Tmu/4. However, since the function is
      // nonlinear, the minimum is found at a shorter distance. We use
      // Tmu/5 as the trial step
      auto step = Tmu/5.0;

      // Current energy
      auto initial_energy(evaluate_step(0.0));

      static int iter=0;
      Vector<Tbase> ttest(linspace<Tbase>(0.0,1.0,51)*Tmu);

#if 0
      Matrix<Tbase> data(ttest.size(), 2);
      data.col(0)=ttest;
      for(size_t i=0;i<ttest.size();i++)
        data(i,1) = scan_step(ttest(i));
      std::ostringstream oss;
      oss << "scan_" << iter << ".dat";
      save_raw_ascii(data, oss.str());
      iter++;

      // Test the routines
      auto dof_list = degrees_of_freedom();
      auto g(search_direction);
      for(size_t i=0;i<g.size();i++) {
        auto dof(dof_list[i]);
        auto iblock = std::get<0>(dof);
        auto iorb = std::get<1>(dof);
        auto jorb = std::get<2>(dof);

        Tbase hh=cbrt(DBL_EPSILON);
        //Tbase hh=1e-10;

        std::function<Tbase(Tbase)> eval = [this, search_direction, i](Tbase xi){
          auto p(search_direction);
          p.setZero();
          p(i) = xi;
          auto entry = evaluate_rotation(p);
          return std::get<1>(entry).first;
        };

        auto E2mi = eval(-2*hh);
        auto Emi = eval(-hh);
        auto Ei = eval(hh);
        auto E2i = eval(2*hh);

        Tbase twop = (Ei-initial_energy)/hh;
        Tbase threep = (Ei-Emi)/(2*hh);
        printf("i=%i twop=%e threep=%e\n",i,twop,threep);

        Tbase h2diff = (Ei - 2*initial_energy + Emi)/(hh*hh);
        Tbase h4diff = (-1/12.0*E2mi +4.0/3.0*Emi - 5.0/2.0*initial_energy + 4.0/3.0*Ei -1./12.0*E2i)/(hh*hh);

        g(i) = threep;
        printf("g(%3i), block %i orbitals %i-%i, % e vs % e (two-point   % e) difference % e ratio % e\n",i,iblock, iorb, jorb, gradient(i),g(i),twop,gradient(i)-g(i),gradient(i)/g(i));
        printf("h(%3i), block %i orbitals %i-%i, % e vs % e (three-point % e) difference % e ratio % e\n",i,iblock, iorb, jorb, diagonal_hessian(i),h4diff,h2diff,diagonal_hessian(i)-h4diff,diagonal_hessian(i)/h4diff);
        fflush(stdout);
      }
      std::cout << "Analytic gradient:\n" << gradient << std::endl;
      std::cout << "Finite difference gradient:\n" << g << std::endl;
      std::cout << "Ratio:\n" << (gradient.array()/g.array()).matrix() << std::endl;
#endif

      // Line search
      bool search_success = false;
      for(size_t itrial=0; itrial<10; itrial++) {
        if(verbosity_>=5) {
          printf("Trial iteration %i\n",itrial);
          fflush(stdout);
        }

        // Evaluate the energy
        auto trial_energy = evaluate_step(step);
        if(trial_energy < initial_energy) {
          // We already decreased the energy! Don't do anything more,
          // because our expansion point has already changed and going
          // further would make no sense.
          search_success = true;
          break;
        }

        // Now we can fit a second order polynomial y = a x^2 + dE x +
        // initial_energy to our data: we know the initial value and the slope, and
        // the observed value.
        auto dE = (gradient).dot( search_direction);
        auto a = (trial_energy - dE*step - initial_energy)/(step*step);

        if(verbosity_>=10) {
          printf("a = %e\n",a);
          fflush(stdout);
        }

        // To be realistic, the parabola should open up
        auto fit_okay = std::isnormal(a) and a>0.0;
        if(fit_okay) {
          auto predicted_step = -dE/(2.0*a);
          auto predicted_energy = a * predicted_step*predicted_step + dE*predicted_step + initial_energy;

          // To be reliable, the predicted optimal step should also be
          // in [0.0, step]
          if(predicted_step < 0.0 or predicted_step > step)
            fit_okay = false;
          if(predicted_step == step)
            // Nothing to do since the step was already evaluated!
            break;

          if(fit_okay) {
            auto observed_energy = evaluate_step(predicted_step);
            if(verbosity_>=5) {
              printf("Predicted energy % .10f observed energy % .10f difference %e\n", predicted_energy, observed_energy,predicted_energy-observed_energy);
              fflush(stdout);
            }

            if(observed_energy < initial_energy) {
              search_success=true;
              break;
            } else {
              if(verbosity_>=5) {
                printf("Error: energy did not decrease in line search! Decreasing trial step size\n");
                fflush(stdout);
              }
              step = std::min(10.0*predicted_step, step/2.0);
            }
          }
        }
      }
      if(not search_success) {
        Vector<Tbase> ttest(logspace<Tbase>(-16,4,101)*Tmu);
        Matrix<Tbase> data(ttest.size(), 2);
        data.col(0)=ttest/Tmu;
        for(size_t i=0;i<ttest.size();i++) {
          data(i,1) = scan_step(ttest(i));
          printf("%e %e % e % e\n",data(i,0),data(i,0)*Tmu,data(i,1),data(i,1)-get_energy());
          fflush(stdout);
        }
        save_raw_ascii(data, "linesearch.dat");
        throw std::runtime_error("Failed to find suitable step size.\n");
      }
    }

    std::vector<IndexVector> occupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> occ_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        occ_idx[l]=find_indices_where(occupations[l], [&](Tbase v){return v >= occupied_threshold_;});
      }
      return occ_idx;
    }

    std::vector<IndexVector> unoccupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> virt_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        virt_idx[l]=find_indices_where(occupations[l], [&](Tbase v){return v < occupied_threshold_;});
      }
      return virt_idx;
    }

  public:
    SCFSolver(const IndexVector & number_of_blocks_per_particle_type, const Vector<Tbase> & maximum_occupation, const Vector<Tbase> & number_of_particles, const FockBuilder<Torb, Tbase> & fock_builder, const std::vector<std::string> & block_descriptions) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder), block_descriptions_(block_descriptions), frozen_occupations_(false), verbosity_(5) {
      // Run sanity checks
      number_of_blocks_ = (number_of_blocks_per_particle_type_).sum();
      if(maximum_occupation_.size() != number_of_blocks_) {
        std::ostringstream oss;
        oss << "Vector of maximum occupation is not of expected length! Got " << maximum_occupation_.size() << " elements, expected " << number_of_blocks_ << "!\n";
        throw std::logic_error(oss.str());
      }
      if(number_of_particles_.size() != number_of_blocks_per_particle_type_.size()) {
        std::ostringstream oss;
        oss << "Vector of number of particles is not of expected length! Got " << number_of_particles_.size() << " elements, expected " << number_of_blocks_per_particle_type_ << "!\n";
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

    bool frozen_occupations() const {
      return frozen_occupations_;
    }

    void frozen_occupations(bool frozen) {
      frozen_occupations_ = frozen;
    }

    int verbosity() const {
      return verbosity_;
    }

    void verbosity(int verbosity) {
      verbosity_ = verbosity;
    }

    Tbase convergence_threshold() const {
      return convergence_threshold_;
    }

    void convergence_threshold(Tbase convergence_threshold) {
      convergence_threshold_ = convergence_threshold;
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
        diff_norm += norm(vectorise(get_density_matrix_block(ihist, iblock)-get_density_matrix_block(jhist, iblock)));
      }
      return diff_norm;
    }

    std::string error_norm() const {
      return error_norm_;
    }

    template <typename Derived>
    Tbase norm(const Eigen::MatrixBase<Derived> & mat, std::string nrm = "") const {
      if(nrm == "")
        nrm = error_norm_;
      if(nrm == "rms") {
        if(mat.size() == 0)
          return Tbase{0};
        return mat.norm() / std::sqrt(Tbase(mat.size()));
      } else if(nrm == "fro" || nrm == "2") {
        return mat.norm();
      } else if(nrm == "inf") {
        return mat.template lpNorm<Eigen::Infinity>();
      } else if(nrm == "1") {
        return mat.template lpNorm<1>();
      } else {
        throw std::logic_error("Unknown norm: " + nrm);
      }
    }

    void error_norm(const std::string & error_norm) {
      // Set the norm
      error_norm_ = error_norm;
      // and check that it is a valid option
      Vector<Tbase> test = Vector<Tbase>::Ones(1);
      (void) norm(test);
    }

    size_t maximum_iterations() const {
      return maximum_iterations_;
    }

    void maximum_iterations(size_t maxit) {
      maximum_iterations_ = maxit;
    }

    Tbase diis_epsilon() const {
      return diis_epsilon_;
    }

    void diis_epsilon(Tbase eps) {
      diis_epsilon_ = eps;
    }

    Tbase diis_threshold() const {
      return diis_threshold_;
    }

    void diis_threshold(Tbase eps) {
      diis_threshold_ = eps;
    }

    Tbase diis_diagonal_damping() const {
      return diis_diagonal_damping_;
    }

    void diis_diagonal_damping(Tbase eps) {
      diis_diagonal_damping_ = eps;
    }

    Tbase diis_restart_factor() const {
      return diis_restart_factor_;
    }

    void diis_restart_factor(Tbase eps) {
      diis_restart_factor_ = eps;
    }

    Tbase optimal_damping_threshold() const {
      return optimal_damping_threshold_;
    }

    void optimal_damping_threshold(Tbase eps) {
      optimal_damping_threshold_ = eps;
    }

    int maximum_history_length() const {
      return maximum_history_length_;
    }

    void maximum_history_length(int maximum_history_length) {
      maximum_history_length_ = maximum_history_length;
    }

    int oda_restart_steps() const {
      return oda_restart_steps_;
    }

    void oda_restart_steps(int oda_restart_steps) {
      oda_restart_steps_ = oda_restart_steps;
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
        if(fock.second[iblock].hasNaN()) {
          throw std::logic_error("Got NaN in Fock matrix!\n");
        }
        if(fock.second[iblock].array().isInf().any()) {
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
        Matrix<Torb> fsymm = Tbase(0.5)*(fock[iblock] + fock[iblock].adjoint());
        Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(fsymm);
        diagonalized_fock.second[iblock] = es.eigenvalues();
        diagonalized_fock.first[iblock] = es.eigenvectors();

        if(verbosity_>=10) {
          std::cout << block_descriptions_[iblock] << " orbital energies: "
                    << diagonalized_fock.second[iblock].transpose() << std::endl;
        }
        fflush(stdout);
      }

      return diagonalized_fock;
    }

    Index particle_block_offset(size_t iparticle) const {
      return (iparticle>0) ? number_of_blocks_per_particle_type_.head(iparticle).sum() : Index{0};
    }

    Vector<Tbase> determine_number_of_particles_by_aufbau(const OrbitalEnergies<Tbase> & orbital_energies) const {
      Vector<Tbase> number_of_particles = Vector<Tbase>::Zero(number_of_blocks_);

      // Loop over particle types
      for(size_t particle_type = 0; particle_type < number_of_blocks_per_particle_type_.size(); particle_type++) {
        // Compute the offset in the block array
        size_t block_offset = particle_block_offset(particle_type);

        // Collect the orbital energies with the block index and the in-block index for this particle type
        std::vector<std::tuple<Tbase, size_t, size_t>> all_energies;
        for(size_t iblock = block_offset; iblock < block_offset + number_of_blocks_per_particle_type_(particle_type); iblock++)
          for(size_t iorb = 0; iorb < orbital_energies[iblock].size(); iorb++)
            all_energies.push_back(std::make_tuple(orbital_energies[iblock](iorb), iblock, iorb));

        // Sort the energies in increasing order
        std::stable_sort(all_energies.begin(), all_energies.end(), [](const std::tuple<Tbase, size_t, size_t> & a, const std::tuple<Tbase, size_t, size_t> & b) {return std::get<0>(a) < std::get<0>(b);});

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
      Vector<Tbase> number_of_particles = (fixed_number_of_particles_per_block_.size() == number_of_blocks_) ? fixed_number_of_particles_per_block_ : determine_number_of_particles_by_aufbau(orbital_energies);

      // Determine the number of occupied orbitals
      OrbitalOccupations<Tbase> occupations(orbital_energies.size());
      for(size_t iblock=0; iblock<orbital_energies.size(); iblock++) {
        if(orbital_energies[iblock].size()==0)
          continue;
        occupations[iblock].setZero(orbital_energies[iblock].size());

        Tbase num_left = number_of_particles(iblock);
        for(size_t iorb=0; iorb < occupations[iblock].size(); iorb++) {
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
        if(callback_convergence_function_) {

            // Data to pass to callback function
            std::map<std::string, std::any> callback_data;
            callback_data["dE"] = get_energy() - old_energy_;
            callback_data["diis_error"] = norm(diis_error_vector(0));

            return callback_convergence_function_(callback_data);
        } else {
            return norm(diis_error_vector(0)) <= convergence_threshold_;
        }
    }

    void run() {
      old_energy_ = 0.0;
      // Number of consecutive steps that the procedure failed to decrease the energy
      int failed_iterations = 0;
      size_t noda_steps = 0;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        Tbase diis_error = norm(diis_error_vector(0));
        Tbase diis_max_error = (diis_error_vector(0)).template lpNorm<Eigen::Infinity>();
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
              std::cout << block_descriptions_[l] << " occupations: "
                        << occupations[l].head(occ_idx[l].maxCoeff()+1).transpose() << std::endl;
          }
        }

        if(noda_steps == 0) {
          if(failed_iterations >= oda_restart_steps()) {
            // Run ODA for half the history length
            noda_steps = maximum_history_length_/2;
            if(verbosity_>=5) {
              printf("Switching to optimal damping for next iterations\n");
            }
          }
          if(diis_max_error >= optimal_damping_threshold_) {
            // The orbitals are so bad we can't trust A/EDIIS or DIIS
            noda_steps = 1;
          }
          if(frozen_occupations_) {
            // Don't let ODA overwrite frozen occs
            noda_steps = 0;
          }
        }

        // Do ODA if necessary
        if(noda_steps>0) {
          noda_steps--;
          old_energy_ = get_energy();
          if(verbosity_>=5) {
            if(diis_max_error >= optimal_damping_threshold_)
              printf("Optimal damping step due to large DIIS max error %e\n", diis_max_error);
            else
              printf("Optimal damping step\n");
          }
          callback_data["step"] = std::string("ODA");
          if(callback_function_)
            callback_function_(callback_data);
          if(optimal_damping_step())
            failed_iterations=0;

        } else {
          // Compute mixing factor (Garza and Scuseria, 2012)
          Tbase aediis_coeff;
          if(diis_error < diis_threshold_) {
            // If error is small, use pure DIIS
            aediis_coeff = 0.0;
          } else {
            if(diis_error < diis_epsilon_) {
              // Compute AEDIIS mixing coefficient
              aediis_coeff = (diis_error-diis_threshold_)/(diis_epsilon_-diis_threshold_);
            } else {
              // Error is large, use A/EDIIS
              aediis_coeff = 1.0;
            }
          }
          Vector<Tbase> weights;
          std::string step;
          std::tie(weights, step) = minimal_error_sampling_algorithm_weights(aediis_coeff);
          if(verbosity_>=5)
            printf("%s step\n",step.c_str());
          if(verbosity_>=10)
            std::cout << "Extrapolation weights" << ": " << weights.transpose() << std::endl;

          // Do the callback
          callback_data["step"] = step;
          if(callback_function_)
            callback_function_(callback_data);

          // Perform extrapolation.
          old_energy_ = get_energy();
          if(!attempt_extrapolation(weights)) {
            if(verbosity_>=10) printf("Warning: did not go down in energy!\n");
            // Increment number of consecutive failed iterations
            failed_iterations++;
          } else {
            // Step succeeded, reset counter
            failed_iterations=0;
          }
        }
        // Do cleanup
        cleanup();
      }
    }

    void run_optimal_damping() {
      old_energy_ = 0.0;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        Tbase diis_error = norm(diis_error_vector(0));
        Tbase diis_max_error = (diis_error_vector(0)).template lpNorm<Eigen::Infinity>();
        Tbase dE = get_energy() - old_energy_;

        if(verbosity_>=5) {
          printf("\n\n");
        }
        if(verbosity_>0) {
          printf("Iteration %i: %i Fock evaluations energy % .10f change % e DIIS error vector %s norm %e\n", (int) iteration, (int) number_of_fock_evaluations_, get_energy(), dE, error_norm_.c_str(), diis_error);
        }

        // Data to pass to callback function
        std::map<std::string, std::any> callback_data;
        callback_data["iter"] = iteration;
        callback_data["nfock"] = number_of_fock_evaluations_;
        callback_data["E"] = get_energy();
        callback_data["dE"] = get_energy() - old_energy_;
        callback_data["diis_error"] = diis_error;
        callback_data["diis_max_error"] = diis_max_error;
        callback_data["step"] = std::string("ODA");

        // Convergence check
        if(converged()) {
          if(verbosity_>0) {
            printf("Converged to energy % .10f\n", get_energy());
          }
          callback_data["step"] = std::string("Converged");
          if(callback_function_)
            callback_function_(callback_data);
          break;
        }

        // Printout
        if(callback_function_)
          callback_function_(callback_data);

        old_energy_ = get_energy();
        if(not optimal_damping_step())
          throw std::logic_error("Could not find descent step!\n");

        if(verbosity_>=5) {
          const auto occupations = get_orbital_occupations();
          auto occ_idx(occupied_orbitals(occupations));
          for(size_t l=0;l<occ_idx.size();l++) {
            if(occ_idx[l].size())
              std::cout << block_descriptions_[l] << " occupations: "
                        << occupations[l].head(occ_idx[l].maxCoeff()+1).transpose() << std::endl;
          }
        }
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

      verbosity_ = 0;
      frozen_occupations_ = false;
      while(true) {
        // Count the number of particles in each block
        Vector<Tbase> number_of_particles_per_block = Vector<Tbase>::Zero(number_of_blocks_);
        for(size_t iblock=0; iblock<number_of_particles_per_block.size(); iblock++) {
          if(empty_block(iblock))
            continue;
          number_of_particles_per_block[iblock] = (reference_occupations[iblock]).sum();
        }
        std::cout << "Number of particles per block" << ": " << number_of_particles_per_block.transpose() << std::endl;

        // List of occupations and resulting energies
        std::vector<std::pair<Vector<Tbase>,Tbase>> list_of_energies;

        // Loop over particle types. We have a double loop, since finding the lowest state in UHF probably requires this
        for(size_t iparticle=0; iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
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
              Tbase i_target_capacity_left = i_target_capacity - (reference_occupations[iblock_target]).sum();
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

                printf("isource = %i itarget = %i imoved = %f\n", iblock_source, iblock_target, i_moved);
                std::cout << "trial number of particles" << ": " << trial_number.transpose() << std::endl;
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

          for(size_t jparticle=0; jparticle<=iparticle; jparticle++) {
            size_t jblock_start = particle_block_offset(jparticle);
            size_t jblock_end = jblock_start + number_of_blocks_per_particle_type_(jparticle);

            // Loop over blocks of particles
            for(size_t iblock_source = iblock_start; iblock_source < iblock_end; iblock_source++)
              for(size_t iblock_target = iblock_start; iblock_target < iblock_end; iblock_target++) {

                bool same_particle = (iparticle == jparticle);
                size_t jblock_source_end = same_particle ? iblock_source+1 : jblock_end;
                size_t jblock_target_end = same_particle ? iblock_target+1 : jblock_end;
                printf("iparticle= %i jparticle= %i isource=%i itarget=%i\n",iparticle,jparticle,iblock_source,iblock_target);

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
                    Tbase i_target_capacity_left = i_target_capacity - (reference_occupations[iblock_target]).sum();
                    int num_i_max = std::ceil(std::min(num_i_source, i_target_capacity_left));
                    num_i_max = std::min(num_i_max, (int) std::round(std::min(maximum_occupation_[iblock_source], maximum_occupation_[iblock_target])));

                    Tbase num_j_source = number_of_particles_per_block[jblock_source];
                    Tbase j_target_capacity = reference_occupations[jblock_target].size()*maximum_occupation_[jblock_target];
                    Tbase j_target_capacity_left = j_target_capacity - (reference_occupations[jblock_target]).sum();
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

                        printf("isource = %i itarget = %i imoved = %f\n", iblock_source, iblock_target, i_moved);
                        printf("jsource = %i jtarget = %i jmoved = %f\n", jblock_source, jblock_target, j_moved);
                        std::cout << "trial number of particles" << ": " << trial_number.transpose() << std::endl;
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


