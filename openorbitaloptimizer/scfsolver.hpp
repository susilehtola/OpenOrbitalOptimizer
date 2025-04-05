/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once
#include <any>
#include <map>
#include <vector>
#include <armadillo>
#include "cg_optimizer.hpp"

namespace OpenOrbitalOptimizer {
  /// A symmetry block of orbitals is defined by the corresponding N x
  /// N matrix of orbital coefficients
  template<typename T> using OrbitalBlock = arma::Mat<T>;
  /// The set of orbitals is defined by a vector of orbital blocks,
  /// corresponding to each symmetry block of each particle type
  template<typename T> using Orbitals = std::vector<OrbitalBlock<T>>;

  /// A symmetry block of orbital gradients is defined by the
  /// corresponding N x N matrix
  template<typename T> using OrbitalGradientBlock = arma::Mat<T>;
  /// The set of orbital gradients is defined by a vector of orbital
  /// blocks, corresponding to each symmetry block of each particle
  /// type
  template<typename T> using OrbitalGradients = std::vector<OrbitalGradientBlock<T>>;

  /// A symmetry block of diagonal orbital Hessians is defined by the
  /// corresponding N x N matrix
  template<typename T> using DiagonalOrbitalHessianBlock = arma::Mat<T>;
  /// The set of diagonal orbital Hessians is defined by a vector of orbital
  /// blocks, corresponding to each symmetry block of each particle
  /// type
  template<typename T> using DiagonalOrbitalHessians = std::vector<DiagonalOrbitalHessianBlock<T>>;

  /// The occupations for each orbitals are floating point numbers
  template<typename T> using OrbitalBlockOccupations = arma::Col<T>;
  /// The occupations for the whole set of orbitals are again a
  /// vector
  template<typename T> using OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;

  /// The pair of orbitals and occupations defines the density matrix
  template<typename Torb, typename Tbase> using DensityMatrix = std::pair<Orbitals<Torb>,OrbitalOccupations<Tbase>>;

  /// Orbital energies are stored as a vector of vectors
  template<typename T> using OrbitalEnergies = std::vector<arma::Col<T>>;

  /// A symmetry block in a Fock matrix is likewise defined by a N x
  /// N matrix
  template<typename T> using FockMatrixBlock = arma::Mat<T>;
  /// The whole set of Fock matrices is a vector of blocks
  template<typename T> using FockMatrix = std::vector<FockMatrixBlock<T>>;
  /// The return of Fock matrix diagonalization is
  template<typename Torb, typename Tbase> using DiagonalizedFockMatrix = std::pair<Orbitals<Torb>,OrbitalEnergies<Tbase>>;

  /// The Fock matrix builder returns the energy and the Fock
  /// matrices for each orbital block
  template<typename Torb, typename Tbase> using FockBuilderReturn = std::pair<Tbase, FockMatrix<Torb>>;
  /// The Fock builder takes in the orbitals and orbital occupations,
  /// and returns the energy and Fock matrices
  template<typename Torb, typename Tbase> using FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;

  /// The history of orbital optimization is defined by the orbitals
  /// and their occupations - together the density matrix - and the
  /// resulting energy and Fock matrix
  template<typename Torb, typename Tbase> using OrbitalHistoryEntry = std::tuple<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>, size_t>;
  /// The history is then a vector
  template<typename Torb, typename Tbase> using OrbitalHistory = std::vector<OrbitalHistoryEntry<Torb, Tbase>>;

  /// List of orbital rotation angles: block index and orbital indices
  using OrbitalRotation = std::tuple<size_t, arma::uword, arma::uword>;

  /// SCF solver class
  template<typename Torb, typename Tbase> class SCFSolver {
    /* Input data section */
    /// The number of orbital blocks per particle type (length ntypes)
    arma::uvec number_of_blocks_per_particle_type_;
    /// The maximal capacity of each orbital block
    arma::Col<Tbase> maximum_occupation_;
    /// The number of particles of each class in total (length ntypes, used to determine Aufbau occupations)
    arma::Col<Tbase> number_of_particles_;
    /// The Fock builder used to evaluate energies and Fock matrices
    FockBuilder<Torb, Tbase> fock_builder_;
    /// Descriptions of the blocks
    std::vector<std::string> block_descriptions_;
    /// Callback function
    std::function<void(const std::map<std::string,std::any> & data)> callback_function_;

    /** (Optional) fixed number of particles in each symmetry, affects
        the way occupations are assigned in Aufbau. These are used if
        the array has the expected size.
    */
    arma::Col<Tbase> fixed_number_of_particles_per_block_;
    /// (Optional) freeze occupations altogether to their previous values
    bool frozen_occupations_;

    /// Verbosity level: 0 for silent, higher values for more info
    int verbosity_;

    /* Internal data section */
    /// The number of blocks
    size_t number_of_blocks_;
    /// The orbital history used for convergence acceleration
    OrbitalHistory<Torb, Tbase> orbital_history_;
    /// Orbital energies, updated each iteration from the lowest-energy solution
    OrbitalOccupations<Tbase> orbital_occupations_;

    /// Number of Fock matrix evaluations
    size_t number_of_fock_evaluations_ = 0;

    /// Maximum number of iterations
    size_t maximum_iterations_ = 128;
    /// Start to mix in DIIS at this error threshold (Garza and Scuseria, 2012)
    Tbase diis_epsilon_ = 1e-1;
    /// Threshold for pure DIIS (Garza and Scuseria, 2012)
    Tbase diis_threshold_ = 1e-4;
    /// Damping factor for DIIS diagonal (Hamilton and Pulay, 1986)
    Tbase diis_diagonal_damping_ = 0.02;
    /// DIIS restart criterion (Chupin et al, 2021)
    Tbase diis_restart_factor_ = 1e-4;

    /// Criterion for max error for which to use optimal damping
    Tbase optimal_damping_threshold_ = 1.0;

    /// History cleanup criterion: keep only those density matrices that satisfy delta ||P0-Pi|| < min_{j>0} ||P0-Pj||
    Tbase density_restart_factor_ = 1e-4;
    /// History length
    int maximum_history_length_ = 10;
    /// Convergence threshold for orbital gradient
    Tbase convergence_threshold_ = 1e-7;
    /// Norm to use by default: root-mean-square error
    std::string error_norm_ = "rms";

    /// Minimal normalized projection of preconditioned search direction onto gradient
    Tbase minimal_gradient_projection_ = 1e-4;
    /// Threshold for detection of occupied orbitals
    Tbase occupied_threshold_ = 1e-6;
    /// Initial level shift
    Tbase initial_level_shift_ = 1.0;
    /// Level shift diminution factor
    Tbase level_shift_factor_ = 2.0;

    /* Internal functions */
    /// Is the block empty?
    bool empty_block(size_t iblock) const {
      // Check if Fock matrix has zero dimension
      if(iblock>=std::get<0>(orbital_history_[0]).first.size())
        throw std::logic_error("Trying to check empty block for nonexistent index!\n");
      return std::get<1>(orbital_history_[0]).second[iblock].n_elem == 0;
    }

    /// Get a block of the density matrix for the ihist:th entry
    arma::Mat<Torb> get_density_matrix_block(size_t ihist, size_t iblock) const {
      const auto orbitals = get_orbital_block(ihist, iblock);
      const auto occupations = get_orbital_occupation_block(ihist, iblock);
      return orbitals * arma::diagmat(occupations) * arma::trans(orbitals);
    }

    /// Get a block of the orbital occupations for the ihist:th entry
    OrbitalBlock<Torb> get_orbital_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbitals for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbitals for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).first[iblock];
    }

    /// Get a block of the orbital occupations for the ihist:th entry
    OrbitalBlockOccupations<Tbase> get_orbital_occupation_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).second[iblock];
    }

    /// Get lowest energy after the given reference index
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

    /// Get the energy for the entry
    size_t get_index(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access index for nonexistent history member!\n");
      return std::get<2>(orbital_history_[ihist]);
    }

    /// Get largest index
    size_t largest_index() const {
      size_t index = get_index(0);
      for(size_t i=1;i<orbital_history_.size();i++) {
        index = std::max(index, get_index(i));
      }
      return index;
    }

    /// Matrix dimensions
    arma::uvec matrix_dimension() const {
      const auto & fock = std::get<1>(orbital_history_[0]).second;
      arma::uvec dim(fock.size());
      for(size_t i=0;i<fock.size();i++)
        dim(i) = fock[i].n_cols;
      return dim;
    }

    /// Get a block of the Fock matrix for the ihist:th entry
    arma::Mat<Torb> get_fock_matrix_block(size_t ihist, size_t iblock) const {
      return std::get<1>(orbital_history_[ihist]).second[iblock];
    }

    /// Vectorise
    arma::Col<Tbase> vectorise(const arma::Mat<Torb> & mat) const {
      if constexpr (arma::is_real<Torb>::value) {
        return arma::vectorise(mat);
      } else {
        return arma::join_cols(arma::vectorise(arma::real(mat)),arma::vectorise(arma::imag(mat)));
      }
    }

    /// Vectorise
    arma::Col<Tbase> vectorise(const std::vector<arma::Mat<Torb>> & mat) const {
      // Compute length of return vector
      size_t N=0;

      std::vector<arma::Col<Tbase>> vectors(mat.size());
      for(size_t iblock=0;iblock<mat.size();iblock++) {
        if(mat[iblock].n_elem==0)
          continue;
        vectors[iblock]=vectorise(mat[iblock]);
        N += vectors[iblock].n_elem;
      }

      arma::Col<Tbase> v(N,arma::fill::zeros);
      size_t ioff=0;
      for(size_t iblock=0;iblock<vectors.size();iblock++) {
        if(mat[iblock].n_elem==0)
          continue;
        v.subvec(ioff,ioff+vectors[iblock].n_elem-1)=vectors[iblock];
        ioff += vectors[iblock].n_elem;
      }

      return v;
    }

    arma::Mat<Torb> matricise(const arma::Col<Tbase> & vec, size_t nrows, size_t ncols) const {
      if constexpr (arma::is_real<Torb>::value) {
        if(vec.n_elem != nrows*ncols) {
          std::ostringstream oss;
          oss << "Matricise error: expected " << nrows*ncols << " elements for " << nrows << " x " << ncols << " real matrix, but got " << vec.n_elem << " instead!\n";
          throw std::logic_error(oss.str());
        }
        return arma::Mat<Torb>(vec.memptr(), nrows, ncols);
      } else {
        if(vec.n_elem != 2*nrows*ncols) {
          std::ostringstream oss;
          oss << "Matricise error: expected " << 2*nrows*ncols << " elements for " << nrows << " x " << ncols << " complex matrix, but got " << vec.n_elem << " instead!\n";
          throw std::logic_error(oss.str());
        }

        arma::Mat<Tbase> real(vec.memptr(), nrows, ncols);
        arma::Mat<Tbase> imag(vec.memptr()+nrows*ncols, nrows, ncols);
        arma::Mat<Torb> mat(real*std::complex<Tbase>(1.0,0.0) + imag*std::complex<Tbase>(0.0,1.0));
        return mat;
      }
    }

    std::vector<arma::Mat<Torb>> matricise(const arma::Col<Tbase> & vec, const arma::uvec & dim) const {
      std::vector<arma::Mat<Torb>> mat(dim.n_elem);
      size_t ioff = 0;
      for(size_t iblock=0; iblock<dim.n_elem; iblock++) {
        if(dim(iblock)==0)
          continue;
        size_t size = dim(iblock)*dim(iblock);
        if constexpr (not arma::is_real<Torb>::value) {
          size *= 2;
        }
        mat[iblock] = matricise(vec.subvec(ioff, ioff+size-1), dim(iblock), dim(iblock));
      }
      return mat;
    }

    /// Compute DIIS residual
    arma::Mat<Torb> diis_residual(size_t ihist, size_t iblock) const {
      // Error is measured by FPS-SPF = FP - PF, since we have a unit metric.
      auto F = get_fock_matrix_block(ihist, iblock);
      auto P = get_density_matrix_block(ihist, iblock);
      arma::Mat<Torb> PF = P*F;
      PF -= arma::trans(PF);

      // To make the L^infty error independent of the underlying basis
      // set, we project the residual into the best orbitals we have
      auto C = get_orbital_block(0, iblock);
      PF = C.t() * PF * C;
      return PF;
    }

    /// Compute DIIS residual
    std::vector<arma::Mat<Torb>> diis_residual(size_t ihist) const {
      std::vector<arma::Mat<Torb>> residuals(number_of_blocks_);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        residuals[iblock] = diis_residual(ihist, iblock);
      }
      return residuals;
    }

    /// Form DIIS error vector for ihist:th entry
    arma::Col<Tbase> diis_error_vector(size_t ihist, size_t iblock) const {
      return vectorise(diis_residual(ihist, iblock));
    }

    /// Form DIIS error vector for ihist:th entry
    arma::Col<Tbase> diis_error_vector(size_t ihist) const {
      // Form error vectors
      std::vector<arma::Col<Tbase>> error_vectors(number_of_blocks_);
      for(size_t iblock = 0; iblock<number_of_blocks_;iblock++) {
        error_vectors[iblock] = diis_error_vector(ihist, iblock);
        if(verbosity_>=20)
          printf("ihist %i block %i error vector norm %e\n",ihist,iblock,norm(error_vectors[iblock]));
        if(verbosity_>=30)
          error_vectors[iblock].print();
      }

      // Compound error vector
      size_t nelem = 0;
      for(auto & block: error_vectors)
        nelem += block.size();

      arma::Col<Tbase> return_vector(nelem);
      size_t ioff=0;
      for(auto & block: error_vectors) {
        if(block.size()>0) {
          return_vector.subvec(ioff,ioff+block.size()-1) = block;
          ioff += block.size();
        }
      }
      if(ioff!=nelem)
        throw std::logic_error("Indexing error!\n");

      return return_vector;
    }

    /// Compute element of DIIS error matrix
    Tbase diis_error_matrix_element(size_t ihist, size_t jhist) const {
      Tbase el=0.0;
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        arma::Col<Tbase> ei(diis_error_vector(ihist, iblock));
        arma::Col<Tbase> ej(diis_error_vector(jhist, iblock));
        el += arma::dot(ei,ej);
      }
      return el;
    }

    /// Form DIIS error matrix
    arma::Mat<Tbase> diis_error_matrix(const std::vector<size_t> & mask) const {
      // The error matrix is given by the orbital gradient dot products
      const size_t N=mask.size();
      arma::Mat<Tbase> B(N,N,arma::fill::zeros);

      for(size_t ihist=0; ihist<N; ihist++) {
        for(size_t jhist=0; jhist<=ihist; jhist++) {
          B(ihist, jhist) = B(jhist, ihist) = diis_error_matrix_element(mask[ihist], mask[jhist]);
        }
      }
      return B;
    }

    arma::Col<Tbase> diis_error_matrix_diagonal() const {
      arma::Col<Tbase> B(orbital_history_.size(),arma::fill::zeros);
      for(size_t ihist=0; ihist<B.n_elem; ihist++) {
        B(ihist) = diis_error_matrix_element(ihist, ihist);
      }
      return B;
    }

    arma::Mat<Tbase> diis_error_matrix() const {
      std::vector<size_t> mask(orbital_history_.size());
      for(size_t i=0;i<mask.size();i++)
        mask[i]=i;
      return diis_error_matrix(mask);
    }

    /// Calculate DIIS weights
    arma::Col<Tbase> diis_weights() const {
      // Only use reference points with error residuals that are sufficiently small
      std::vector<size_t> history_mask(orbital_history_.size());
      for(size_t i=0;i<history_mask.size();i++)
        history_mask[i]=i;
      arma::Col<Tbase> residuals(history_mask.size());
      for(size_t i=0;i<residuals.size();i++)
        residuals(i) = diis_error_matrix_element(history_mask[i], history_mask[i]);
      Tbase min_residual = arma::min(residuals);
      for(size_t i=history_mask.size()-1;i<history_mask.size();i--)
        // Criterion from Chupin et al, 2012
        if(residuals(i)*diis_restart_factor_ > min_residual)
          history_mask.erase(history_mask.begin()+i);
      size_t nrestart = orbital_history_.size()-history_mask.size();
      if(verbosity_>=10 and nrestart>0)
        printf("Removed %i entries corresponding to large DIIS errors\n", nrestart);

      // Set up the DIIS error matrix
      const size_t N=history_mask.size();
      arma::Mat<Tbase> B(N+1,N+1,arma::fill::value(-1.0));
      B.submat(0,0,N-1,N-1)=diis_error_matrix(history_mask);
      B(N,N)=0.0;

      // Apply the diagonal damping
      B.submat(0,0,N-1,N-1).diag() *= 1.0+diis_diagonal_damping_;

      // To improve numerical conditioning, scale entries of error
      // matrix such that the last diagonal element is one; Eckert et
      // al, J. Comput. Chem 18. 1473-1483 (1997)
      arma::Col<Tbase> Bdiag(arma::diagvec(B));
      B.submat(0,0,N-1,N-1) /= arma::min(Bdiag.subvec(0,N-1));

      // Right-hand side of equation is
      arma::Col<Tbase> rh(N+1, arma::fill::zeros);
      rh(N)=-1.0;

      // Solve the equation
      arma::Col<Tbase> diis_weights;
      arma::solve(diis_weights, B, rh);
      diis_weights=diis_weights.subvec(0,N-1);

      // Pad to full space
      arma::Col<Tbase> diis_weights_full(orbital_history_.size(),arma::fill::zeros);
      for(size_t i=0;i<history_mask.size();i++)
        diis_weights_full[history_mask[i]] = diis_weights[i];

      return diis_weights_full;
    }
    /// Calculate ADIIS weights by minimizing quadratic form
    arma::Col<Tbase> aediis_weights(const arma::Col<Tbase> & b, const arma::Mat<Tbase> & A) const {
      if(b.n_elem==1) {
        // Nothing to optimize
        return arma::ones<arma::Col<Tbase>>(b.n_elem);
      }

      // Parameters
      const size_t max_iter = 1000000;
      const Tbase df_tol = 1e-8;

      // Function to evaluate function value
      std::function<Tbase(const arma::Col<Tbase> & x)> fx = [b, A](const arma::Col<Tbase> & x) {
        return 0.5*arma::as_scalar(x.t()*A*x) + arma::dot(b,x);
      };

      // Function to determine optimal step
      std::function<Tbase(const arma::Col<Tbase> &, const arma::Col<Tbase> &)> optimal_step = [b, A](const arma::Col<Tbase> & current_direction, const arma::Col<Tbase> & x) {
        return -(arma::as_scalar(current_direction.t()*A*x) + arma::dot(b,current_direction)) / (arma::as_scalar(current_direction.t()*A*current_direction));
      };

      /// Make initial guesses for parameters
      std::vector<arma::Col<Tbase>> xguess;
      // Center point
      xguess.push_back(arma::Col<Tbase>(b.n_elem,arma::fill::value(1.0/b.n_elem)));
      // "Gauss" points
      for(size_t i=0;i<b.n_elem;i++) {
        arma::Col<Tbase> xtr(b.n_elem,arma::fill::value(1.0/(b.n_elem+2)));
        xtr(i) *= 3;
        xguess.push_back(xtr);
      }
      // End points
      for(size_t i=0;i<b.n_elem;i++) {
        arma::Col<Tbase> xtr(b.n_elem,arma::fill::zeros);
        xtr(i) = 1.0;
        xguess.push_back(xtr);
      }

      // Find minimum
      arma::vec yguess(xguess.size());
      for(size_t i=0;i<xguess.size();i++)
        yguess[i] = fx(xguess[i]);

      arma::uvec idx(arma::sort_index(yguess,"ascend"));
      arma::Col<Tbase> x = xguess[idx[0]];
      //x.t().print("Initial x");

      /// Matrix of search directions
      arma::Mat<Tbase> search_directions(b.n_elem,b.n_elem,arma::fill::eye);

      /// Evaluate initial point
      auto current_point = fx(x);
      auto old_point = current_point;
      auto old_x = x;

      // Powell algorithm
      for(size_t imacro=0; imacro<max_iter; imacro++) {
        Tbase curval(current_point);

        for(size_t i=0; i<b.n_elem; i++) {
          arma::Col<Tbase> c_i(search_directions.col(i));
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
        arma::Col<Tbase> dx = x - old_x;

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

        //x.t().print("x");
        if(dE > -df_tol) {
          if(verbosity_ >= 10) {
            printf("A/EDIIS weights converged in %i macroiterations\n",imacro);
            //x.t().print("xconv");
          }
          break;
        } else if(imacro==max_iter-1) {
          if(verbosity_ >= 10) {
            printf("A/EDIIS weights did not converge in %i macroiterations, dE=%e\n",imacro, dE);
            //x.t().print("xfinal");
          }
        }

        /*
        // Rotate search directions. Generate a random ordering of the columns
        arma::uvec randperm(arma::randperm(search_directions.n_cols));
        search_directions=search_directions.cols(randperm);
        // Mix the vectors together
        for(size_t i=0;i<search_directions.n_cols;i++)
          for(size_t j=0;j<i;j++) {
            arma::Col<Tbase> randu(1);
            randu.randu();

            arma::Col<Tbase> newi = (1-randu(0))*search_directions.col(i) + randu(0)*search_directions.col(j);
            arma::Col<Tbase> newj = (1-randu(0))*search_directions.col(j) + randu(0)*search_directions.col(i);
            search_directions.col(i) = newi;
            search_directions.col(j) = newj;
          }
        */
      }

      // Handle the edge case where the last matrix has zero norm
      if(x(0)==0.0) {
        x.zeros();
        x(0)=1.0;
        // Reset search directions
        search_directions.eye();
        for(size_t i=0; i<b.n_elem; i++) {
          arma::Col<Tbase> c_i(search_directions.col(i));
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
        //x.t().print("Using suboptimal solution instead");
      }

      //printf("Current energy %e\n",current_point);
      //throw std::logic_error("Stop");

      return x;
    }

    /// ADIIS linear term: <D_i - D_0 | F_i - F_0>
    arma::Col<Tbase> adiis_linear_term() const {
      arma::Col<Tbase> ret(orbital_history_.size(),arma::fill::zeros);
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        const auto & Dn = get_density_matrix_block(0, iblock);
        const auto & Fn = get_fock_matrix_block(0, iblock);
        for(size_t ihist=0;ihist<ret.size();ihist++) {
          // D_i - D_n
          arma::Mat<Torb> dD(get_density_matrix_block(ihist, iblock) - Dn);
          ret(ihist) += 2.0*std::real(arma::trace(dD*Fn));
        }
      }
      return ret;
    }

    /// EDIIS linear term: list of energies
    arma::Col<Tbase> ediis_linear_term() const {
      arma::Col<Tbase> ret(orbital_history_.size(),arma::fill::zeros);
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
        ret(ihist) = get_energy(ihist);
      }
      return ret;
    }

    /// ADIIS quadratic term: <D_i - D_n | F_j - F_n>
    arma::Mat<Tbase> adiis_quadratic_term() const {
      arma::Mat<Tbase> ret(orbital_history_.size(),orbital_history_.size(),arma::fill::zeros);
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        const auto & Dn = get_density_matrix_block(0, iblock);
        const auto & Fn = get_fock_matrix_block(0, iblock);
        for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
          for(size_t jhist=0;jhist<orbital_history_.size();jhist++) {
            // D_i - D_n
            arma::Mat<Torb> dD(get_density_matrix_block(ihist, iblock) - Dn);
            // F_j - F_n
            arma::Mat<Torb> dF(get_fock_matrix_block(jhist, iblock) - Fn);
            ret(ihist,jhist) += std::real(arma::trace(dD*dF));
          }
        }
      }
      // Only the symmetric part matters; we also multiply by two
      // since we define the quadratic model as 0.5*x^T A x + b x
      return ret+ret.t();
    }

    /// EDIIS quadratic term: -0.5*<D_i - D_j | F_i - F_j>
    arma::Mat<Tbase> ediis_quadratic_term() const {
      arma::Mat<Tbase> ret(orbital_history_.size(),orbital_history_.size(),arma::fill::zeros);
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
          for(size_t jhist=0;jhist<orbital_history_.size();jhist++) {
            // D_i - D_j
            arma::Mat<Torb> dD(get_density_matrix_block(ihist, iblock) - get_density_matrix_block(jhist, iblock));
            // F_i - F_j
            arma::Mat<Torb> dF(get_fock_matrix_block(ihist, iblock) - get_fock_matrix_block(jhist, iblock));
            ret(ihist,jhist) -= std::real(arma::trace(dD*dF));
          }
        }
      }
      // Only the symmetric part matters; the factor 0.5 already
      // exists in the base model
      return 0.5*(ret+ret.t());
    }

    /// Calculate ADIIS weights
    arma::Col<Tbase> adiis_weights() const {
      return aediis_weights(adiis_linear_term(), adiis_quadratic_term());
    }

    /// Calculate EDIIS weights
    arma::Col<Tbase> ediis_weights() const {
      return aediis_weights(ediis_linear_term(), ediis_quadratic_term());
    }

    /** Minimal Error Sampling Algorithm (MESA), doi:10.14288/1.0372885 */
    std::tuple<arma::Col<Tbase>,std::string> minimal_error_sampling_algorithm_weights(Tbase aediis_coeff) const {
      // Form DIIS and ADIIS weights
      arma::Col<Tbase> diis_w(diis_weights());
      if(verbosity_>=10) diis_w.t().print("DIIS weights");
      if(aediis_coeff == 0.0) {
        std::string step = "DIIS";
        return std::make_tuple(diis_w,step);
      }

      // Get various extrapolation weights
      const size_t N = orbital_history_.size();
      arma::Col<Tbase> adiis_w(adiis_weights());
      if(verbosity_>=10) adiis_w.t().print("ADIIS weights");
      arma::Col<Tbase> ediis_w(ediis_weights());
      if(verbosity_>=10) ediis_w.t().print("EDIIS weights");

      // Candidates
      arma::Mat<Tbase> candidate_w(N, 2, arma::fill::zeros);
      size_t icol=0;
      candidate_w.col(icol++) = adiis_w;
      candidate_w.col(icol++) = ediis_w;
      const std::vector<std::string> weight_legend({"ADIIS", "EDIIS"});
      std::string step;

      arma::Col<Tbase> density_projections(candidate_w.n_cols, arma::fill::zeros);
      for(size_t iw=0;iw<candidate_w.n_cols;iw++) {
        density_projections(iw) = density_projection(candidate_w.col(iw));
      }
      if(verbosity_>=10)
        density_projections.t().print("Density projections");

      arma::uword idx;
      density_projections.max(idx);
      if(verbosity_>=10)
        printf("Max density projection %e with %s weights\n",density_projections(idx),weight_legend[idx].c_str());

      arma::Col<Tbase> aediis_w = candidate_w.col(idx);
      arma::Col<Tbase> weights(aediis_coeff * aediis_w + (1.0 - aediis_coeff) * diis_w);
      if(aediis_coeff == 1.0) {
        step = weight_legend[idx];
      } else {
        step = weight_legend[idx] + "+DIIS";
      }

      return std::make_tuple(weights,step);
    }

    /// Compute density change with given weights
    Tbase density_projection(const arma::Col<Tbase> & weights) const {
      // Get the extrapolated Fock matrix
      auto fock(extrapolate_fock(weights));

      // Diagonalize the extrapolated Fock matrix
      auto diagonalized_fock = compute_orbitals(fock);
      auto & new_orbitals = diagonalized_fock.first;
      auto & new_orbital_energies = diagonalized_fock.second;

      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      // Reference calculation
      const auto reference_orbitals = get_orbitals();
      const auto reference_occupations = get_orbital_occupations();
      // Occupations corresponding to the reference orbitals
      auto maximum_overlap_occupations = determine_maximum_overlap_occupations(reference_occupations, reference_orbitals, new_orbitals);

      return density_overlap(new_orbitals, maximum_overlap_occupations, reference_orbitals, reference_occupations);
    }


    /// Computes the difference between orbital occupations
    Tbase occupation_difference(const OrbitalOccupations<Tbase> & old_occ, const OrbitalOccupations<Tbase> & new_occ) const {
      Tbase diff = 0.0;
      for(size_t iblock = 0; iblock<old_occ.size(); iblock++) {
        if(old_occ[iblock].n_elem==0)
          continue;
        size_t n = std::min(new_occ[iblock].n_elem, old_occ[iblock].n_elem);
        diff += arma::sum(arma::abs(new_occ[iblock].subvec(0,n-1)-old_occ[iblock].subvec(0,n-1)));
        if(new_occ[iblock].n_elem>n)
          diff += arma::sum(arma::abs(new_occ[iblock].subvec(n,new_occ[iblock].n_elem-1)));
        else if(old_occ[iblock].n_elem>n)
          diff += arma::sum(arma::abs(old_occ[iblock].subvec(n,old_occ[iblock].n_elem-1)));
      }

      return diff;
    }

    /// Perform DIIS extrapolation of Fock matrix
    FockMatrix<Torb> extrapolate_fock(const arma::Col<Tbase> & weights) const {
      if(weights.n_elem != orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.n_elem << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated Fock matrix
      FockMatrix<Torb> extrapolated_fock(number_of_blocks_);
      for(size_t iblock = 0; iblock < extrapolated_fock.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        // Apply the DIIS weight
        for(size_t ihist = 0; ihist < orbital_history_.size(); ihist++) {
          arma::Mat<Torb> block = weights(ihist) * get_fock_matrix_block(ihist, iblock);
          if(ihist==0) {
            extrapolated_fock[iblock] = block;
          } else {
            extrapolated_fock[iblock] += block;
          }
        }
      }

      return extrapolated_fock;
    }

    /// Perform DIIS extrapolation of density matrix
    DensityMatrix<Torb, Tbase> extrapolate_density(const arma::Col<Tbase> & weights) const {
      if(weights.n_elem != orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.n_elem << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated density matrix
      std::vector<arma::Mat<Torb>> orbitals(number_of_blocks_);
      std::vector<arma::Col<Tbase>> occupations(number_of_blocks_);
      for(size_t iblock = 0; iblock < number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;

        arma::Mat<Torb> dm_block;
        for(size_t ihist = 0; ihist < orbital_history_.size(); ihist++) {
          arma::Mat<Torb> block = weights(ihist) * get_density_matrix_block(ihist, iblock);
          if(ihist==0) {
            dm_block = block;
          } else {
            dm_block += block;
          }
        }

        // Flip the sign so that the orbitals come in increasing occupation
        arma::eig_sym(occupations[iblock], orbitals[iblock], -dm_block);
        occupations[iblock] *= -1;
        // Zero out numerically zero occupations
        arma::uvec zeroidx(arma::find(arma::abs(occupations[iblock])<=10*maximum_occupation_(iblock)*std::numeric_limits<Tbase>::epsilon()));
        occupations[iblock](zeroidx).zeros();
      }

      return std::make_pair(orbitals,occupations);
    }

    /// Compute maximum overlap orbital occupations
    OrbitalOccupations<Tbase> determine_maximum_overlap_occupations(const OrbitalOccupations<Tbase> & reference_occupations, const Orbitals<Torb> & C_reference, const Orbitals<Torb> & C_new) const {
      OrbitalOccupations<Tbase> new_occupations(reference_occupations);
      for(size_t iblock=0; iblock<new_occupations.size(); iblock++) {
        if(C_reference[iblock].n_elem == 0)
          continue;
        // Initialize
        new_occupations[iblock].zeros();

        // Magnitude of the overlap between the new orbitals and the reference ones
        arma::Mat<Tbase> orbital_projections(arma::abs(C_new[iblock].t()*C_reference[iblock]));

        // Occupy the orbitals in ascending energy, especially if there are unoccupied orbitals in-between
        for(size_t iorb=0; iorb<reference_occupations[iblock].n_elem; iorb++) {
          // Projections for this orbital
          auto projection = orbital_projections.col(iorb);
          // Find the maximum index
          auto maximal_projection_index = arma::index_max(projection);
          auto maximal_projection = projection(maximal_projection_index);
          // Store projection
          new_occupations[iblock][maximal_projection_index] = reference_occupations[iblock](iorb);
          // and reset the corresponding row so that the orbital can't be reused
          orbital_projections.row(maximal_projection_index).zeros();

          //printf("Symmetry %i: reference orbital %i with occupation %.3f matches new orbital %i with projection %e\n",(int) iblock, (int) iorb, reference_occupations[iblock](iorb), (int) maximal_projection_index, maximal_projection);
        }
      }

      return new_occupations;
    }

    /// Compute density overlap between two sets of orbitals and occupations
    Tbase density_overlap(const Orbitals<Torb> & lorb, const OrbitalOccupations<Tbase> & locc, const Orbitals<Torb> & rorb, const OrbitalOccupations<Tbase> & rocc) const {
      if(lorb.size() != rorb.size() or lorb.size() != locc.size() or lorb.size() != rocc.size())
        throw std::logic_error("Inconsistent orbitals!\n");

      Tbase ovl=0.0;
      for(size_t iblock=0; iblock<lorb.size(); iblock++) {
        if(lorb[iblock].n_elem==0)
          continue;
        // Get orbital coefficients and occupations
        const auto & lC = lorb[iblock];
        const auto & lo = locc[iblock];
        const auto & rC = rorb[iblock];
        const auto & ro = rocc[iblock];
        // Compute projection
        arma::Mat<Torb> Pl(lC*arma::diagmat(lo)*lC.t());
        arma::Mat<Torb> Pr(rC*arma::diagmat(ro)*rC.t());
        ovl += std::real(arma::trace(Pl*Pr));
      }
      return ovl;
    }

    /// Attempt extrapolation with given weights
    bool attempt_extrapolation(const arma::Col<Tbase> & weights, bool density=false) {
      // Get the extrapolated Fock matrix
      if(not density) {
        auto fock(extrapolate_fock(weights));
        return attempt_fock(fock);
      } else {
        auto dm(extrapolate_density(weights));
        return add_entry(std::make_pair(dm.first, dm.second));
      }
    }

    /// See if given Fock matrix reduces the energy
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

    /// Optimal damping step
    bool optimal_damping_step() {
      // Diagonalize the best Fock matrix
      auto diagonalized_fock = compute_orbitals(std::get<1>(orbital_history_[0]).second);
      auto new_orbitals = diagonalized_fock.first;
      auto new_orbital_energies = diagonalized_fock.second;
      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      // Form the new density matrix
      std::vector<arma::Mat<Torb>> dm_new(new_orbitals.size());
      for(size_t iblock=0; iblock<new_orbitals.size(); iblock++) {
        if(new_orbitals[iblock].n_cols == 0)
          continue;
        dm_new[iblock] = new_orbitals[iblock] * arma::diagmat(new_occupations[iblock]) * arma::trans(new_orbitals[iblock]);
      }

      // Compute the energy gradient for each particle type for the density matrix mixing: P -> (1-lambda)*Pcurrent + lambda*Pnew
      size_t nparticles = number_of_blocks_per_particle_type_.n_elem;
      arma::Col<Tbase> dEdlambda(nparticles, arma::fill::zeros);
      for(size_t iparticle=0;iparticle<nparticles;iparticle++) {
        size_t block_offset = particle_block_offset(iparticle);
        for(size_t iblock=block_offset;iblock<block_offset+number_of_blocks_per_particle_type_(iparticle);iblock++) {
          if(empty_block(iblock))
            continue;
          // Current density matrix
          arma::Mat<Torb> fock_current(get_fock_matrix_block(0, iblock));
          arma::Mat<Torb> dm_current(get_density_matrix_block(0, iblock));
          dEdlambda(iparticle) += std::real(arma::trace(fock_current*(dm_new[iblock] - dm_current)));
        }
      }
      if(verbosity_>=10)
        dEdlambda.t().print("Optimal damping: dE/dlambda");

      // Search direction is therefore
      arma::Col<Tbase> search_direction = -dEdlambda;
      // As we start the search from the current density matrix,
      // lambda=0 at the outset and we set any negative directions as
      // invalid
      arma::uvec negative_indices = arma::find(search_direction < 0.0);
      if(negative_indices.n_elem)
        search_direction(negative_indices).zeros();

      arma::uvec valid_directions = arma::find(search_direction != 0);
      if(valid_directions.n_elem==0) {
        // No valid search directions!
        return false;
      }

      // The resulting trial is therefore the step that takes us to
      // the edge
      arma::Col<Tbase> lambda_trial = search_direction/arma::max(search_direction);

      // Helper function
      std::function<DensityMatrix<Torb, Tbase>(const arma::Col<Tbase> &)> interpolate_dm = [&](const arma::Col<Tbase> & step) {
        Orbitals<Torb> new_orbs(number_of_blocks_);
        OrbitalOccupations<Tbase> new_occs(number_of_blocks_);
        for(size_t iparticle=0;iparticle<nparticles;iparticle++) {
          size_t block_offset = particle_block_offset(iparticle);
          for(size_t iblock=block_offset;iblock<block_offset+number_of_blocks_per_particle_type_(iparticle);iblock++) {
            if(empty_block(iblock))
              continue;
            arma::Mat<Torb> dm_block((1-step(iparticle))*get_density_matrix_block(0, iblock) + step(iparticle)*dm_new[iblock]);
            // Flip the sign so that the orbitals come in increasing occupation
            arma::eig_sym(new_occs[iblock], new_orbs[iblock], -dm_block);
            new_occs[iblock] *= -1;
            // Zero out numerically zero occupations
            arma::uvec zeroidx(arma::find(arma::abs(new_occs[iblock])<=10*maximum_occupation_(iblock)*std::numeric_limits<Tbase>::epsilon()));
            new_occs[iblock](zeroidx).zeros();
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
      Tbase dE0 = arma::dot(dEdlambda, lambda_trial);

      arma::Col<Tbase> dEdlambda2(nparticles, arma::fill::zeros);
      for(size_t iparticle=0;iparticle<nparticles;iparticle++) {
        size_t block_offset = particle_block_offset(iparticle);
        for(size_t iblock=block_offset;iblock<block_offset+number_of_blocks_per_particle_type_(iparticle);iblock++) {
          if(empty_block(iblock))
            continue;
          // Current density matrix
          arma::Mat<Torb> fock_new(get_fock_matrix_block(1, iblock));
          arma::Mat<Torb> dm_current(get_density_matrix_block(0, iblock));
          dEdlambda2(iparticle) += std::real(arma::trace(fock_new*(dm_new[iblock] - dm_current)));
        }
      }
      Tbase dE1 = arma::dot(dEdlambda2, lambda_trial);

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
      Tbase x1 = (-b - sqrt(b*b - 4*a*c))/(2*a);
      Tbase x2 = (-b + sqrt(b*b - 4*a*c))/(2*a);
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

    /// Clean up history from incorrect occupations
    void cleanup() {
      size_t nremoved=0;
      arma::Col<Tbase> density_differences(orbital_history_.size()-1,arma::fill::zeros);
      for(size_t ihist=1;ihist<orbital_history_.size();ihist++) {
        density_differences(ihist-1)=density_matrix_difference(ihist, 0);
      }
      if(verbosity_ >= 10) {
        density_differences.t().print("Density differences");
      } else if(verbosity_>=5) {
        printf("Density matrix difference %e between lowest-energy and newest entry\n",density_differences(0));
      }

      // Sort the differences
      arma::uvec idx(arma::sort_index(density_differences,"ascend"));
      // Pick the indices that don't satisfy the criterion
      arma::uvec sub_idx(arma::find(density_restart_factor_*density_differences(idx) > density_differences(idx(0))));
      if(sub_idx.n_elem) {
        idx=idx(sub_idx);
        idx=arma::sort(idx,"descend");
        if(verbosity_>=10)
          printf("Removing %i entries corresponding to large change in density matrix\n",(int) idx.n_elem);
        for(auto ihistm1: idx) {
          // Remember the off-by-one in the indices
          orbital_history_.erase(orbital_history_.begin()+ihistm1+1);
        }
      }
    }

    /// Form list of rotation angles
    std::vector<OrbitalRotation> degrees_of_freedom() const {
      std::vector<OrbitalRotation> dofs;
      // Reference calculation
      const auto reference_occupations = get_orbital_occupations();

      // List occupied-occupied rotations, in case some orbitals are not fully occupied
      for(size_t iblock = 0; iblock < reference_occupations.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        arma::uvec occupied_indices = arma::find(reference_occupations[iblock] > 0.0);
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
        arma::uvec occupied_indices = arma::find(reference_occupations[iblock] > 0.0);
        arma::uvec virtual_indices = arma::find(reference_occupations[iblock] == 0.0);
        for(auto o: occupied_indices)
          for(auto v: virtual_indices)
            dofs.push_back(std::make_tuple(iblock, o, v));
      }

      return dofs;
    }

    /// Formulate the orbital gradient vector
    arma::Col<Tbase> orbital_gradient_vector() const {
      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      arma::Col<Tbase> orb_grad;

      if constexpr (arma::is_real<Torb>::value) {
        orb_grad.zeros(dof_list.size());
      } else {
        orb_grad.zeros(2*dof_list.size());
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

        arma::Mat<Torb> fock_mo = orbital_block.t() * fock_block * orbital_block;
        orb_grad(idof) = 2*std::real(fock_mo(iorb,jorb))*(occ_block(jorb)-occ_block(iorb));
        if constexpr (!arma::is_real<Torb>::value) {
          orb_grad(dof_list.size() + idof) = 2*std::imag(fock_mo(iorb,jorb))*(occ_block(jorb)-occ_block(iorb));
        }
      }

      if(orb_grad.has_nan())
        throw std::logic_error("Orbital gradient has NaNs");

      return orb_grad;
    }

    /// Formulate the diagonal orbital Hessian
    arma::Col<Tbase> diagonal_orbital_hessian() const {
      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      arma::Col<Tbase> orb_hess;

      if constexpr (arma::is_real<Torb>::value) {
        orb_hess.zeros(dof_list.size());
      } else {
        orb_hess.zeros(2*dof_list.size());
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

        arma::Mat<Torb> fock_mo = orbital_block.t() * fock_block * orbital_block;
        orb_hess(idof) = 2*std::real((fock_mo(iorb,iorb)-fock_mo(jorb,jorb))*(occ_block(jorb)-occ_block(iorb)));
        if constexpr (!arma::is_real<Torb>::value) {
          orb_hess(dof_list.size() + idof) = orb_hess(idof);
        }
      }
      return orb_hess;
    }

    /// Formulate the diagonal orbital Hessian
    arma::Col<Tbase> precondition_search_direction(const arma::Col<Tbase> & gradient, const arma::Col<Tbase> & diagonal_hessian, Tbase shift=0.1) const {
      if(gradient.n_elem != diagonal_hessian.n_elem)
        throw std::logic_error("precondition_search_direction: gradient and diagonal hessian have different size!\n");

      // Build positive definite diagonal Hessian
      arma::Col<Tbase> positive_hessian(diagonal_hessian);
      positive_hessian += (-arma::min(diagonal_hessian)+shift)*arma::ones<arma::Col<Tbase>>(positive_hessian.n_elem);

      Tbase normalized_projection;
      Tbase maximum_spread = arma::max(positive_hessian);
      arma::Col<Tbase> preconditioned_direction;
      while(true) {
        // Normalize the largest values
        arma::Col<Tbase> normalized_hessian(positive_hessian);
        arma::uvec idx(arma::find(normalized_hessian>maximum_spread));
        normalized_hessian(idx) = maximum_spread*arma::ones<arma::Col<Tbase>>(idx.n_elem);

        // and divide the gradient by its square root
        preconditioned_direction = gradient/arma::sqrt(normalized_hessian);
        if(preconditioned_direction.has_nan())
          throw std::logic_error("Preconditioned search direction has NaNs");

        normalized_projection = arma::dot(preconditioned_direction, gradient) / std::sqrt(arma::norm(preconditioned_direction,2)*arma::norm(gradient, 2));
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

    /// Rotation matrices
    Orbitals<Torb> form_rotation_matrices(const arma::Col<Tbase> & x) const {
      const Orbitals<Torb> reference_orbitals(get_orbitals());

      // Get the degrees of freedom
      auto dof_list = degrees_of_freedom();
      arma::Col<Tbase> orb_grad(dof_list.size());
      // Sort them by symmetry
      std::vector<std::vector<std::tuple<arma::uword, arma::uword, size_t>>> blocked_dof(reference_orbitals.size());
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
        kappa[iblock].zeros(reference_orbitals[iblock].n_cols, reference_orbitals[iblock].n_cols);
        for(auto dof: blocked_dof[iblock]) {
          auto iorb = std::get<0>(dof);
          auto jorb = std::get<1>(dof);
          auto idof = std::get<2>(dof);
          kappa[iblock](iorb,jorb) = x(idof);
        }
        // imaginary parameters
        if constexpr (!arma::is_real<Torb>::value) {
          for(auto dof: blocked_dof[iblock]) {
            auto iorb = std::get<0>(dof);
            auto jorb = std::get<1>(dof);
            auto idof = std::get<2>(dof);
            kappa[iblock](iorb,jorb) += Torb(0.0,x(dof_list.size()+idof));
          }
        }
        // Antisymmetrize
        kappa[iblock] -= arma::trans(kappa[iblock]);
      }

      return kappa;
    }

    /// Determine maximum step size; doi:10.1016/j.sigpro.2009.03.015
    Tbase maximum_rotation_step(const arma::Col<Tbase> & x) const {
      // Get the rotation matrices
      auto kappa(form_rotation_matrices(x));

      Tbase maximum_step = std::numeric_limits<Tbase>::max();
      for(size_t iblock=0; iblock < kappa.size(); iblock++) {
        if(kappa[iblock].n_elem==0)
          continue;
        arma::Col<Tbase> eval;
        arma::Mat<std::complex<Tbase>> evec;
        arma::Mat<std::complex<Tbase>> kappa_imag(kappa[iblock]*std::complex<Tbase>(0.0,-1.0));
        arma::eig_sym(eval, evec, kappa_imag);

        // Assume objective function is 4th order in orbitals
        Tbase block_maximum = 0.5*M_PI/arma::max(arma::abs(eval));
        // The maximum allowed step is determined as the minimum of the block-wise steps
        maximum_step = std::min(maximum_step, block_maximum);
      }

      return maximum_step;
    }

    /// Rotate the orbitals through the given parameters
    Orbitals<Torb> rotate_orbitals(const arma::Col<Tbase> & x) const {
      auto kappa(form_rotation_matrices(x));

      // Rotate the orbitals
      Orbitals<Torb> new_orbitals(get_orbitals());
      for(size_t iblock=0; iblock < new_orbitals.size(); iblock++) {
        if(empty_block(iblock))
          continue;

        // Exponentiated kappa
        arma::Mat<Torb> expkappa;

#if 0
        expkappa = arma::expmat(kappa[iblock]);
#else
        // Do eigendecomposition
        arma::Col<Tbase> eval;
        arma::Mat<std::complex<Tbase>> evec;
        arma::Mat<std::complex<Tbase>> kappa_imag(kappa[iblock]*std::complex<Tbase>(0.0,-1.0));
        arma::eig_sym(eval, evec, kappa_imag);
        // Exponentiate
        arma::Mat<std::complex<Tbase>> expkappa_imag(evec*arma::diagmat(arma::exp(eval*std::complex<Tbase>(0.0,1.0)))*evec.t());
        if constexpr (arma::is_real<Torb>::value) {
          expkappa = arma::real(expkappa_imag);
        } else {
          expkappa = expkappa_imag;
        }
#endif

        // Do the rotation
        new_orbitals[iblock] = new_orbitals[iblock]*expkappa;
      }

      return new_orbitals;
    }
    /// Make an orbital history entry
    OrbitalHistoryEntry<Torb, Tbase> make_history_entry(const DensityMatrix<Torb, Tbase> & density_matrix, const FockBuilderReturn<Torb, Tbase> & fock) const {
      static size_t index=0;
      return std::make_tuple(density_matrix, fock, index++);
    }
    /// Evaluate the energy with a given orbital rotation vector
    OrbitalHistoryEntry<Torb, Tbase> evaluate_rotation(const arma::Col<Tbase> & x) {
      // Rotate orbitals
      auto new_orbitals(rotate_orbitals(x));
      // Compute the Fock matrix
      auto reference_occupations = get_orbital_occupations();

      auto density_matrix = std::make_pair(new_orbitals, reference_occupations);
      auto fock = fock_builder_(density_matrix);
      number_of_fock_evaluations_++;
      return make_history_entry(density_matrix, fock);
    }
    /// Level shift step
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
          arma::Col<Tbase> fractional_occupations(get_orbital_occupation_block(0, iblock)/maximum_occupation_(iblock));
          fractional_occupations = arma::ones<arma::Col<Tbase>>(fractional_occupations.n_elem) - fractional_occupations;
          arma::Mat<Torb> orbitals(get_orbital_block(0, iblock));

          shifted_fock[iblock] += level_shift *(orbitals * arma::diagmat(fractional_occupations) * orbitals.t());
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
    /// Take a steepest descent step
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
      if(arma::dot(search_direction, gradient) >= 0.0) {
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
      arma::Col<Tbase> ttest(arma::linspace<arma::Col<Tbase>>(0.0,1.0,51)*Tmu);

#if 0
      arma::Mat<Tbase> data(ttest.n_elem, 2);
      data.col(0)=ttest;
      for(size_t i=0;i<ttest.n_elem;i++)
        data(i,1) = scan_step(ttest(i));
      std::ostringstream oss;
      oss << "scan_" << iter << ".dat";
      data.save(oss.str(),arma::raw_ascii);
      iter++;

      // Test the routines
      auto dof_list = degrees_of_freedom();
      auto g(search_direction);
      for(size_t i=0;i<g.n_elem;i++) {
        auto dof(dof_list[i]);
        auto iblock = std::get<0>(dof);
        auto iorb = std::get<1>(dof);
        auto jorb = std::get<2>(dof);

        Tbase hh=cbrt(DBL_EPSILON);
        //Tbase hh=1e-10;

        std::function<Tbase(Tbase)> eval = [this, search_direction, i](Tbase xi){
          auto p(search_direction);
          p.zeros();
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
      gradient.print("Analytic gradient");
      g.print("Finite difference gradient");
      (gradient/g).print("Ratio");
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
        auto dE = arma::dot(gradient, search_direction);
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
              step = std::max(10.0*predicted_step, step/2.0);
            }
          }
        }
      }
      if(not search_success) {
        arma::Col<Tbase> ttest(arma::logspace<arma::Col<Tbase>>(-16,4,101)*Tmu);
        arma::Mat<Tbase> data(ttest.n_elem, 2);
        data.col(0)=ttest/Tmu;
        for(size_t i=0;i<ttest.n_elem;i++) {
          data(i,1) = scan_step(ttest(i));
          printf("%e %e % e % e\n",data(i,0),data(i,0)*Tmu,data(i,1),data(i,1)-get_energy());
          fflush(stdout);
        }
        data.save("linesearch.dat",arma::raw_ascii);
        throw std::runtime_error("Failed to find suitable step size.\n");
      }
    }

    /// List of occupied orbitals
    std::vector<arma::uvec> occupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<arma::uvec> occ_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        occ_idx[l]=arma::find(occupations[l]>=occupied_threshold_);
      }
      return occ_idx;
    }

    /// List of occupied orbitals
    std::vector<arma::uvec> unoccupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<arma::uvec> virt_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        virt_idx[l]=arma::find(occupations[l]<occupied_threshold_);
      }
      return virt_idx;
    }

  public:
    /// Constructor
    SCFSolver(const arma::uvec & number_of_blocks_per_particle_type, const arma::Col<Tbase> & maximum_occupation, const arma::Col<Tbase> & number_of_particles, const FockBuilder<Torb, Tbase> & fock_builder, const std::vector<std::string> & block_descriptions) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder), block_descriptions_(block_descriptions), verbosity_(5), frozen_occupations_(false) {
      // Run sanity checks
      number_of_blocks_ = arma::sum(number_of_blocks_per_particle_type_);
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

    /// Initialize the solver with a guess Fock matrix
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

    /// Initialize with precomputed orbitals and occupations
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
        if(get_orbital_block(0,iblock).n_cols != get_fock_matrix_block(0,iblock).n_cols) {
          printf("get_orbital_block(0,iblock).n_cols=%i != get_fock_matrix_block(0,iblock).n_cols)=%i\n",get_orbital_block(0,iblock).n_cols,get_fock_matrix_block(0,iblock).n_cols);
          consistent=false;
        }
        if(get_orbital_occupation_block(0,iblock).n_elem != get_fock_matrix_block(0,iblock).n_cols) {
          if(verbosity_>=10)
            printf("get_orbital_occupation_block(0,iblock).n_elem=%i != get_fock_matrix_block(0,iblock).n_cols=%i\n",get_orbital_occupation_block(0,iblock).n_elem,get_fock_matrix_block(0,iblock).n_cols);
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

    /// Fix the number of occupied orbitals per block
    void fixed_number_of_particles_per_block(const arma::Col<Tbase> & number_of_particles_per_block) {
      fixed_number_of_particles_per_block_ = number_of_particles_per_block;
    }

    /// Get frozen occupations
    bool frozen_occupations() const {
      return frozen_occupations_;
    }

    /// Set frozen occupations
    void frozen_occupations(bool frozen) {
      frozen_occupations_ = frozen;
    }

    /// Get verbosity
    int verbosity() const {
      return verbosity_;
    }

    /// Set verbosity
    void verbosity(int verbosity) {
      verbosity_ = verbosity;
    }

    /// Get convergence threshold
    Tbase convergence_threshold() const {
      return convergence_threshold_;
    }

    /// Set verbosity
    void convergence_threshold(Tbase convergence_threshold) {
      convergence_threshold_ = convergence_threshold;
    }

    /// Get the energy for the n:th entry
    Tbase get_energy(size_t ihist=0) const {
      if(ihist>orbital_history_.size())
        throw std::logic_error("Invalid entry!\n");
      return std::get<1>(orbital_history_[ihist]).first;
    }


    /// Density matrix difference norm
    Tbase density_matrix_difference(size_t ihist, size_t jhist) {
      Tbase diff_norm = 0.0;
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        diff_norm += norm(get_density_matrix_block(ihist, iblock)-get_density_matrix_block(jhist, iblock));
      }
      return diff_norm;
    }

    /// Get the used error norm
    std::string error_norm() const {
      return error_norm_;
    }

    /// Evaluate the norm
    Tbase norm(const arma::Mat<Tbase> & mat, std::string norm="") const {
      if(norm == "")
        norm=error_norm_;
      if(norm == "rms") {
        // rms isn't implemented in Armadillo for some reason
        return arma::norm(mat,"fro")/std::sqrt(1.0*mat.n_elem);
      } else {
        return arma::norm(mat, norm.c_str());
      }
    }

    /// Set the used error norm
    void error_norm(const std::string & error_norm) {
      // Set the norm
      error_norm_ = error_norm;
      // and check that it is a valid option
      arma::Col<Tbase> test(1,arma::fill::ones);
      (void) norm(test);
    }

    /// Get the maximum number of iterations
    size_t maximum_iterations() const {
      return maximum_iterations_;
    }

    /// Set the maximum number of iterations
    void maximum_iterations(size_t maxit) {
      maximum_iterations_ = maxit;
    }

    /// When to start mixing in DIIS
    Tbase diis_epsilon() const {
      return diis_epsilon_;
    }

    /// When to start mixing in DIIS
    void diis_epsilon(Tbase eps) {
      diis_epsilon_ = eps;
    }

    /// When to switch over to DIIS
    Tbase diis_threshold() const {
      return diis_threshold_;
    }

    /// When to switch over to DIIS
    void diis_threshold(Tbase eps) {
      diis_threshold_ = eps;
    }

    /// Damping factor for DIIS diagonal
    Tbase diis_diagonal_damping() const {
      return diis_diagonal_damping_;
    }

    /// Damping factor for DIIS diagonal
    void diis_diagonal_damping(Tbase eps) {
      diis_diagonal_damping_ = eps;
    }

    /// DIIS restart criterion
    Tbase diis_restart_factor() const {
      return diis_restart_factor_;
    }

    /// DIIS restart criterion
    void diis_restart_factor(Tbase eps) {
      diis_restart_factor_ = eps;
    }

    /// Use optimal damping when max error bigger than this
    Tbase optimal_damping_threshold() const {
      return optimal_damping_threshold_;
    }

    /// Use optimal damping when max error bigger than this
    void optimal_damping_threshold(Tbase eps) {
      optimal_damping_threshold_ = eps;
    }

    /// Get maximum_history_length
    int maximum_history_length() const {
      return maximum_history_length_;
    }

    /// Set maximum_history_length
    void maximum_history_length(int maximum_history_length) {
      maximum_history_length_ = maximum_history_length;
    }

    /// Add entry to history, return value is True if energy was lowered
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

    /// Add entry to history, return value is True if energy was lowered
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
        if(fock.second[iblock].n_rows==0)
          continue;
        if(fock.second[iblock].has_nan()) {
          throw std::logic_error("Got NaN in Fock matrix!\n");
        }
        if(fock.second[iblock].has_inf()) {
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
        if(orbital_history_.size() > maximum_history_length_)
          orbital_history_.pop_back();

        return return_value;
      }
    }

    /// Print the DIIS history
    void print_history() const {
      printf("Orbital history\n");
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++)
        printf("%2i % .9f % e % i\n",ihist,get_energy(ihist),get_energy(ihist)-get_energy(),get_index(ihist));
    }

    /// Reset the DIIS history
    void reset_history() {
      while(orbital_history_.size()>1)
        orbital_history_.pop_back();
    }

    /// Computes orbitals and orbital energies by diagonalizing the Fock matrix
    DiagonalizedFockMatrix<Torb,Tbase> compute_orbitals(const FockMatrix<Torb> & fock) const {
      DiagonalizedFockMatrix<Torb, Tbase> diagonalized_fock;
      // Allocate memory for orbitals and orbital energies
      diagonalized_fock.first.resize(fock.size());
      diagonalized_fock.second.resize(fock.size());

      // Diagonalize all blocks
      for(size_t iblock = 0; iblock < fock.size(); iblock++) {
        if(fock[iblock].n_elem==0)
          continue;
        // Symmetrize Fock matrix
        arma::Mat<Torb> fsymm(0.5*(fock[iblock]+fock[iblock].t()));
        arma::eig_sym(diagonalized_fock.second[iblock], diagonalized_fock.first[iblock], fsymm);

        if(verbosity_>=10) {
          diagonalized_fock.second[iblock].t().print(block_descriptions_[iblock] + " orbital energies");
        }
        fflush(stdout);
      }

      return diagonalized_fock;
    }

    /// Determines the offset for the blocks of the iparticle:th particle
    arma::uword particle_block_offset(size_t iparticle) const {
      return (iparticle>0) ? arma::sum(number_of_blocks_per_particle_type_.subvec(0,iparticle-1)) : 0;
    }

    /// Determine number of particles in each block
    arma::Col<Tbase> determine_number_of_particles_by_aufbau(const OrbitalEnergies<Tbase> & orbital_energies) const {
      arma::Col<Tbase> number_of_particles(number_of_blocks_, arma::fill::zeros);

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
          auto iorb = std::get<2>(fill_orbital);
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

    /// Determines occupations based on the current orbital energies
    OrbitalOccupations<Tbase> update_occupations(const OrbitalEnergies<Tbase> & orbital_energies) const {
      if(frozen_occupations_)
        return get_orbital_occupations();

      // Number of particles per block
      arma::Col<Tbase> number_of_particles = (fixed_number_of_particles_per_block_.n_elem == number_of_blocks_) ? fixed_number_of_particles_per_block_ : determine_number_of_particles_by_aufbau(orbital_energies);

      // Determine the number of occupied orbitals
      OrbitalOccupations<Tbase> occupations(orbital_energies.size());
      for(size_t iblock=0; iblock<orbital_energies.size(); iblock++) {
        if(orbital_energies[iblock].n_elem==0)
          continue;
        occupations[iblock].zeros(orbital_energies[iblock].size());

        Tbase num_left = number_of_particles(iblock);
        for(size_t iorb=0; iorb < occupations[iblock].n_elem; iorb++) {
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

    /// Check if we are converged
    bool converged() const {
      return norm(diis_error_vector(0)) <= convergence_threshold_;
    }

    /// Run the SCF
    void run() {
      Tbase old_energy = 0.0;
      // Number of consecutive steps that the procedure failed to decrease the energy
      size_t failed_iterations = 0;
      size_t noda_steps = 0;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        Tbase diis_error = norm(diis_error_vector(0));
        Tbase diis_max_error = arma::norm(diis_error_vector(0),"inf");
        Tbase dE = get_energy() - old_energy;

        // Data to pass to callback function
        std::map<std::string, std::any> callback_data;
        callback_data["iter"] = iteration;
        callback_data["nfock"] = number_of_fock_evaluations_;
        callback_data["E"] = get_energy();
        callback_data["dE"] = get_energy() - old_energy;
        callback_data["diis_error"] = diis_error;
        callback_data["diis_max_error"] = diis_max_error;

        if(verbosity_>=5) {
          printf("\n\n");
        }
        if(verbosity_>0) {
          printf("Iteration %i: %i Fock evaluations energy % .10f change % e DIIS error vector %s norm %e\n", iteration, number_of_fock_evaluations_, get_energy(), dE, error_norm_.c_str(), diis_error);
        }
        if(verbosity_>=5) {
          printf("History size %i\n",orbital_history_.size());
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
            if(occ_idx[l].n_elem)
              occupations[l].subvec(0,arma::max(occ_idx[l])).t().print(block_descriptions_[l] + " occupations");
          }
        }

        if(noda_steps == 0) {
          if(failed_iterations >= maximum_history_length_/2) {
            // Run the same number of steps using ODA
            noda_steps = maximum_history_length_/2;
            if(verbosity_>=5) {
              printf("Switching to optimal damping for next iterations\n");
            }
          }
          if(diis_max_error >= optimal_damping_threshold_) {
            // The orbitals are so bad we can't trust A/EDIIS or DIIS
            noda_steps = 1;
          }
        }

        // Do ODA if necessary
        if(noda_steps>0) {
          noda_steps--;
          old_energy = get_energy();
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
          arma::Col<Tbase> weights;
          std::string step;
          std::tie(weights, step) = minimal_error_sampling_algorithm_weights(aediis_coeff);
          if(verbosity_>=5)
            printf("%s step\n",step.c_str());
          if(verbosity_>=10)
            weights.t().print("Extrapolation weights");

          // Do the callback
          callback_data["step"] = step;
          if(callback_function_)
            callback_function_(callback_data);

          // Perform extrapolation.
          old_energy = get_energy();
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

    /// Run optimal damping
    void run_optimal_damping() {
      Tbase old_energy = 0.0;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        Tbase diis_error = norm(diis_error_vector(0));
        Tbase diis_max_error = arma::norm(diis_error_vector(0),"inf");
        Tbase dE = get_energy() - old_energy;

        if(verbosity_>=5) {
          printf("\n\n");
        }
        if(verbosity_>0) {
          printf("Iteration %i: %i Fock evaluations energy % .10f change % e DIIS error vector %s norm %e\n", iteration, number_of_fock_evaluations_, get_energy(), dE, error_norm_.c_str(), diis_error);
        }

        // Data to pass to callback function
        std::map<std::string, std::any> callback_data;
        callback_data["iter"] = iteration;
        callback_data["nfock"] = number_of_fock_evaluations_;
        callback_data["E"] = get_energy();
        callback_data["dE"] = get_energy() - old_energy;
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

        old_energy = get_energy();
        if(not optimal_damping_step())
          throw std::logic_error("Could not find descent step!\n");

        if(verbosity_>=5) {
          const auto occupations = get_orbital_occupations();
          auto occ_idx(occupied_orbitals(occupations));
          for(size_t l=0;l<occ_idx.size();l++) {
            if(occ_idx[l].n_elem)
              occupations[l].subvec(0,arma::max(occ_idx[l])).t().print(block_descriptions_[l] + " occupations");
          }
        }
      }
    }

    /// Get the SCF solution
    DensityMatrix<Torb, Tbase> get_solution(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]);
    }

    /// Get the orbitals
    Orbitals<Torb> get_orbitals(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).first;
    }

    /// Get the orbital occupations
    OrbitalOccupations<Tbase> get_orbital_occupations(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).second;
    }

    /// Get the Fock matrix builder return
    FockBuilderReturn<Torb, Tbase> get_fock_build(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]);
    }

    /// Get the Fock matrix for the ihist:th entry
    FockMatrix<Torb> get_fock_matrix(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]).second;
    }


    /// Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search
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
        arma::Col<Tbase> number_of_particles_per_block(number_of_blocks_,arma::fill::zeros);
        for(size_t iblock=0; iblock<number_of_particles_per_block.size(); iblock++) {
          if(empty_block(iblock))
            continue;
          number_of_particles_per_block[iblock] = arma::sum(reference_occupations[iblock]);
        }
        number_of_particles_per_block.t().print("Number of particles per block");

        // List of occupations and resulting energies
        std::vector<std::pair<arma::Col<Tbase>,Tbase>> list_of_energies;

        // Loop over particle types. We have a double loop, since finding the lowest state in UHF probably requires this
        for(size_t iparticle=0; iparticle<number_of_blocks_per_particle_type_.n_elem; iparticle++) {
          size_t iblock_start = particle_block_offset(iparticle);
          size_t iblock_end = iblock_start + number_of_blocks_per_particle_type_(iparticle);

          // One-particle moves
          for(size_t iblock_source = iblock_start; iblock_source < iblock_end; iblock_source++)
            for(size_t iblock_target = iblock_start; iblock_target < iblock_end; iblock_target++) {
              if(iblock_source == iblock_target)
                continue;

              // Maximum number to move
              Tbase num_i_source = number_of_particles_per_block[iblock_source];
              Tbase i_target_capacity = reference_occupations[iblock_target].n_elem*maximum_occupation_[iblock_target];
              Tbase i_target_capacity_left = i_target_capacity - arma::sum(reference_occupations[iblock_target]);
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
                trial_number.t().print("trial number of particles");
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
                arma::Col<Tbase> dummy;
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
                    Tbase i_target_capacity = reference_occupations[iblock_target].n_elem*maximum_occupation_[iblock_target];
                    Tbase i_target_capacity_left = i_target_capacity - arma::sum(reference_occupations[iblock_target]);
                    int num_i_max = std::ceil(std::min(num_i_source, i_target_capacity_left));
                    num_i_max = std::min(num_i_max, (int) std::round(std::min(maximum_occupation_[iblock_source], maximum_occupation_[iblock_target])));

                    Tbase num_j_source = number_of_particles_per_block[jblock_source];
                    Tbase j_target_capacity = reference_occupations[jblock_target].n_elem*maximum_occupation_[jblock_target];
                    Tbase j_target_capacity_left = j_target_capacity - arma::sum(reference_occupations[jblock_target]);
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
                        trial_number.t().print("trial number of particles");
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
                        arma::Col<Tbase> dummy;
                        fixed_number_of_particles_per_block_ = dummy;
                      }
                  }
              }
          }
        }

        // Sort the list in ascending order
        std::sort(list_of_energies.begin(), list_of_energies.end(), [](const std::pair<arma::Col<Tbase>,Tbase> & a, const std::pair<arma::Col<Tbase>,Tbase> & b) {return a.second < b.second;});

        printf("Configurations\n");
        for(size_t iconf=0;iconf<list_of_energies.size();iconf++) {
          printf("%4i E= % .10f with occupations\n",(int) iconf, list_of_energies[iconf].second);
          list_of_energies[iconf].first.t().print();
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
  };
}
