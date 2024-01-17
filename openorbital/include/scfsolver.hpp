#pragma once
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
  template<typename Torb, typename Tbase> using FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(DensityMatrix<Torb, Tbase>)>;

  /// The history of orbital optimization is defined by the orbitals
  /// and their occupations - together the density matrix - and the
  /// resulting energy and Fock matrix
  template<typename Torb, typename Tbase> using OrbitalHistoryEntry = std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>;
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

    /// Maximum number of iterations
    size_t maximum_iterations_ = 128;
    /// Start to mix in DIIS at this error threshold
    double diis_epsilon_ = 1e-1;
    /// Threshold for pure DIIS
    double diis_threshold_ = 1e-2;
    /// Threshold for a change in occupations
    double occupation_change_threshold_ = 1e-6;
    /// History length
    int maximum_history_length_ = 7;
    /// Convergence threshold for orbital gradient
    double convergence_threshold_ = 1e-7;
    /// Threshold that determines an acceptable increase in energy due to finite numerical precision
    double energy_update_threshold_ = 1e-9;
    /// Norm to use by default: maximum element (Pulay 1982)
    std::string error_norm_ = "inf";

    /// Minimal normalized projection of preconditioned search direction onto gradient
    const double minimal_gradient_projection_ = 1e-4;
    /// ADIIS/EDIIS regularization parameter
    const double adiis_regularization_parameter_ = 1e-3;

    /* Internal functions */
    /// Get a block of the density matrix for the ihist:th entry
    arma::Mat<Torb> get_density_matrix_block(size_t ihist, size_t iblock) const {
      auto & entry = orbital_history_[ihist];
      auto & density_matrix = entry.first;
      return density_matrix.first[iblock] * arma::diagmat(density_matrix.second[iblock]) * arma::trans(density_matrix.first[iblock]);
    }

    /// Get a block of the orbital occupations for the ihist:th entry
    arma::Mat<Torb> get_orbital_block(size_t ihist, size_t iblock) const {
      auto & entry = orbital_history_[ihist];
      return entry.first.first[iblock];
    }

    /// Get a block of the orbital occupations for the ihist:th entry
    arma::Col<Tbase> get_orbital_occupation_block(size_t ihist, size_t iblock) const {
      auto entry = orbital_history_[ihist];
      return entry.first.second[iblock];
    }

    /// Get a block of the density matrix for the ihist:th entry
    arma::Mat<Torb> get_fock_matrix_block(size_t ihist, size_t iblock) const {
      auto entry = orbital_history_[ihist];
      return entry.second.second[iblock];
    }

    /// Form DIIS error vector for ihist:th entry
    arma::Col<Tbase> diis_error_vector(size_t ihist) const {
      /// Helper function
      std::function<arma::Col<Tbase>(const arma::Mat<Torb> &)> extract_error_vector=[](const arma::Mat<Torb> & mat) {
        if constexpr (arma::is_real<Torb>::value) {
          return arma::vectorise(mat);
        } else {
          return arma::join_cols(arma::vectorise(arma::real(mat)),arma::vectorise(arma::imag(mat)));
        }
      };

      // Form error vectors
      std::vector<arma::Col<Tbase>> error_vectors(orbital_history_[ihist].second.second.size());
      for(size_t iblock = 0; iblock<number_of_blocks_;iblock++) {
        // Error is measured by FPS-SPF = FP - PF, since we have a unit metric.
        auto F = get_fock_matrix_block(ihist, iblock);
        auto P = get_density_matrix_block(ihist, iblock);
        auto FP = F*P;
        error_vectors[iblock] = extract_error_vector(FP - arma::trans(FP));
        //printf("ihist %i block %i density norm %e error vector norm %e\n",ihist,iblock,arma::norm(P, error_norm_.c_str()), arma::norm(error_vectors[iblock],error_norm_.c_str()));
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
      return return_vector;
    }

    /// Form DIIS error matrix
    arma::Mat<Tbase> diis_error_matrix() const {
      // The error matrix is given by the orbital gradient dot products
      size_t N=orbital_history_.size();
      arma::Mat<Tbase> B(N,N,arma::fill::zeros);
      for(size_t ihist=0; ihist<N; ihist++) {
        arma::Col<Tbase> ei = diis_error_vector(ihist);
        for(size_t jhist=0; jhist<=ihist; jhist++) {
          arma::Col<Tbase> ej = diis_error_vector(jhist);
          B(jhist,ihist) = B(ihist, jhist) = arma::dot(ei,ej);
        }
      }
      return B;
    }

    /// Calculate C1-DIIS weights
    arma::Col<Tbase> c1diis_weights() const {
      // Set up the DIIS error matrix
      size_t N=orbital_history_.size();
      arma::Mat<Tbase> B(diis_error_matrix());

      /*
        The C1-DIIS method is equivalent to solving the group of linear
        equations
        B w = lambda 1       (1)

        where B is the error matrix, w are the DIIS weights, lambda is the
        Lagrange multiplier that guarantees that the weights sum to unity,
        and 1 stands for a unit vector (1 1 ... 1)^T.

        By rescaling the weights as w -> w/lambda, equation (1) is
        reverted to the form
        B w = 1              (2)

        which can easily be solved using SVD techniques.

        Finally, the weights are renormalized to satisfy
        \sum_i w_i = 1
        which takes care of the Lagrange multipliers.
      */

      // Right-hand side of equation is
      arma::vec rh(N);
      rh.ones();

      // Solve C1-DIIS eigenproblem
      arma::Mat<Tbase> Bvec;
      arma::Col<Tbase> Bval;
      arma::eig_sym(Bval, Bvec, B);

      // Form solution
      arma::Col<Tbase> diis_weights(N,arma::fill::zeros);
      for(size_t i=0;i<N;i++)
        if(Bval(i)!=0.0)
          diis_weights += arma::dot(Bvec.col(i),rh)/Bval(i) * Bvec.col(i);

      // Sanity check for no elements: use even weights
      if(arma::sum(arma::abs(diis_weights))==0.0)
        diis_weights.ones();

      // Normalize solution
      diis_weights/=arma::sum(diis_weights);

      return diis_weights;
    }

    /// Calculate C2-DIIS weights
    arma::Mat<Tbase> c2diis_candidate_weights() const {
      // Set up the DIIS error matrix
      arma::Mat<Tbase> B(diis_error_matrix());

      // Solve C2-DIIS eigenproblem
      arma::Mat<Tbase> evec;
      arma::Col<Tbase> eval;
      arma::eig_sym(eval, evec, B);

      // Normalize solution vectors
      arma::Mat<Tbase> candidate_solutions(evec);
      for(size_t icol=0;icol<evec.n_cols;icol++)
        candidate_solutions.col(icol) /= arma::sum(candidate_solutions.col(icol));

      return candidate_solutions;
    }

    /// Calculate C2-DIIS weights
    arma::Col<Tbase> c2diis_weights(double rejection_threshold = 10.0) const {
      // Set up the DIIS error matrix
      arma::Mat<Tbase> B(diis_error_matrix());
      // Get the candidate solutions
      arma::Mat<Tbase> candidate_solutions = c2diis_candidate_weights();

      // Find best solution that satisfies rejection threshold. Error
      // norms for the extrapolated vectors
      arma::Col<Tbase> error_norms(candidate_solutions.n_cols,arma::fill::ones);
      error_norms *= std::numeric_limits<Tbase>::max();

      for(size_t icol=0; icol < candidate_solutions.n_cols; icol++) {
        arma::Col<Tbase> soln = candidate_solutions.col(icol);
        // Skip solutions that have large elements
        if(arma::max(arma::abs(soln)) >= rejection_threshold)
          continue;
        // Compute extrapolated error
        arma::Col<Tbase> extrapolated_error = B * soln;
        error_norms(icol) = arma::norm(extrapolated_error, 2);
      }

      // Sort the solutions in the extrapolated error
      arma::uvec sortidx;
      sortidx = arma::sort_index(error_norms);
      if(verbosity_ >= 10)
        error_norms(sortidx).print("Sorted C2DIIS errors");

      arma::Col<Tbase> diis_weights;
      for(auto index: sortidx) {
        diis_weights = candidate_solutions.col(index);
        // Skip solutions that have extrapolated error in the same order
        // of magnitude as the used floating point precision
        if(error_norms(index) >= 5*std::numeric_limits<Tbase>::epsilon()) {
          if(verbosity_ >= 10)
            printf("Using C2DIIS solution index %i\n",index);
          break;
        }
      }

      return diis_weights;
    }

    /// Calculate ADIIS weights
    arma::Col<Tbase> adiis_weights() const {
      // Form linear and quadratic terms
      auto linear_term = adiis_linear_term();
      auto quadratic_term = adiis_quadratic_term();

      // Function to compute weights from the parameters
      std::function<arma::Col<Tbase>(const arma::Col<Tbase> & x)> x_to_weight = [](const arma::Col<Tbase> & x) { arma::Col<Tbase> w=arma::square(x)/arma::dot(x,x); return w; };
      // and its Jacobian
      std::function<arma::Mat<Tbase>(const arma::Col<Tbase> & x)> x_to_weight_jacobian = [x_to_weight](const arma::Col<Tbase> & x) {
        auto w(x_to_weight(x));
        auto xnorm = arma::norm(x,2);
        arma::Mat<Tbase> jac(x.n_elem,x.n_elem,arma::fill::zeros);
        for(size_t i=0;i<x.n_elem;i++) {
          for(size_t j=0;j<x.n_elem;j++) {
            jac(i,j) -= w(j)*2.0*x(i)/xnorm;
          }
          jac(i,i) += 2.0*x(i)/xnorm;
        }
        //jac.print("Jacobian");
        return jac;
      };

      // Function to compute the ADIIS energy and gradient
      const double regularization_parameter = adiis_regularization_parameter_;
      std::function<std::pair<Tbase,arma::Col<Tbase>>(const arma::Col<Tbase> & x)> adiis_energy_gradient = [linear_term, quadratic_term, x_to_weight, x_to_weight_jacobian, regularization_parameter](const arma::Col<Tbase> & x) {
        auto w(x_to_weight(x));
        arma::Col<Tbase> g = x_to_weight_jacobian(x)*(linear_term + quadratic_term*w);
        auto fval = arma::dot(linear_term, w) + 0.5*arma::dot(w, quadratic_term*w);

        // Add regularization
        if(regularization_parameter != 0.0) {
          fval += regularization_parameter * arma::dot(x,x);
          g += 2 * regularization_parameter * x;
        }

        return std::make_pair(fval, g);
      };

      // Optimization
      arma::Col<Tbase> x(orbital_history_.size(),arma::fill::ones);
      x = ConjugateGradients::cg_optimize<Tbase>(x, adiis_energy_gradient);

      return x_to_weight(x);
    }

    /// Form <D_i - D_0 | F_i - F_0>
    arma::Col<Tbase> adiis_linear_term() const {
      arma::Col<Tbase> ret(orbital_history_.size(),arma::fill::zeros);
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        const auto & Dn = get_density_matrix_block(0, iblock);
        const auto & Fn = get_fock_matrix_block(0, iblock);
        for(size_t ihist=0;ihist<ret.size();ihist++) {
          // D_i - D_n
          arma::Mat<Torb> dD(get_density_matrix_block(ihist, iblock) - Dn);
          ret(ihist) += std::real(arma::trace(dD*Fn));
        }
      }
      return ret;
    }
    /// Form <D_i - D_j | F_i - F_j>
    arma::Mat<Tbase> adiis_quadratic_term() const {
      arma::Mat<Tbase> ret(orbital_history_.size(),orbital_history_.size(),arma::fill::zeros);
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
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
      // Only the symmetric part matters!
      return 0.5*(ret+ret.t());
    }

    /** Minimal Error Sampling Algorithm (MESA), doi:10.14288/1.0372885 */
    arma::Col<Tbase> minimal_error_sampling_algorithm() const {
      // Get C2-DIIS weights
      arma::Mat<Tbase> c2_diis_w(c2diis_candidate_weights());
      arma::Col<Tbase> c1_diis_w(c1diis_weights());
      arma::Col<Tbase> adiis_w(adiis_weights());

      // Candidates
      arma::Mat<Tbase> candidate_w(c2_diis_w.n_rows, c2_diis_w.n_cols+3, arma::fill::zeros);
      candidate_w.cols(0,c2_diis_w.n_cols-1)=c2_diis_w;
      size_t icol=c2_diis_w.n_cols;

      candidate_w.col(icol++) = c1_diis_w;
      candidate_w.col(icol++) = adiis_w;

      // Last try: just the reference state
      candidate_w(0,icol) = 1.0;
      icol++;

      arma::Col<Tbase> density_projections(candidate_w.n_cols, arma::fill::zeros);
      for(size_t iw=0;iw<candidate_w.n_cols;iw++) {
        density_projections(iw) = density_projection(candidate_w.col(iw));
      }
      density_projections.t().print("Density projections");

      arma::uword idx;
      density_projections.max(idx);
      printf("Max density %e with trial %i\n",density_projections(idx),idx);

      return candidate_w.col(idx);
    }

    /// Computes the difference between orbital occupations
    Tbase occupation_difference(const OrbitalOccupations<Tbase> & old_occ, const OrbitalOccupations<Tbase> & new_occ) const {
      Tbase diff = 0.0;
      for(size_t iblock = 0; iblock<old_occ.size(); iblock++)
        diff += arma::sum(arma::abs(new_occ[iblock]-old_occ[iblock]));
      return diff;
    }

    /// Perform DIIS extrapolation
    FockMatrix<Torb> extrapolate_fock(const arma::Col<Tbase> & weights) const {
      if(weights.n_elem != orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.n_elem << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated Fock matrix
      FockMatrix<Torb> extrapolated_fock(orbital_history_[0].second.second);
      for(size_t iblock = 0; iblock < extrapolated_fock.size(); iblock++) {
        // Apply the DIIS weight
        extrapolated_fock[iblock] *= weights(0);
        // and add the other blocks
        for(size_t ihist=1; ihist < orbital_history_.size(); ihist++)
          extrapolated_fock[iblock] += weights(ihist) * orbital_history_[ihist].second.second[iblock];
      }

      return extrapolated_fock;
    }

    /// Compute maximum overlap orbital occupations
    OrbitalOccupations<Tbase> determine_maximum_overlap_occupations(const OrbitalOccupations<Tbase> & reference_occupations, const Orbitals<Torb> & C_reference, const Orbitals<Torb> & C_new) const {
      OrbitalOccupations<Tbase> new_occupations(reference_occupations);
      for(size_t iblock=0; iblock<new_occupations.size(); iblock++) {
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
        // Get orbital coefficients and occupations
        const auto & lC = lorb[iblock];
        const auto & lo = locc[iblock];
        const auto & rC = rorb[iblock];
        const auto & ro = rocc[iblock];
        // Compute projection
        arma::Mat<Torb> Pl(lC*arma::diagmat(lo)*lC.t());
        arma::Mat<Torb> Pr(rC*arma::diagmat(ro)*rC.t());
        ovl += arma::trace(Pl*Pr);
      }
      return ovl;
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
      auto & reference_solution = orbital_history_[0];
      auto & reference_orbitals = reference_solution.first.first;
      auto reference_occupations = reference_solution.first.second;
      // Occupations corresponding to the reference orbitals
      auto maximum_overlap_occupations = determine_maximum_overlap_occupations(reference_occupations, reference_orbitals, new_orbitals);

      Tbase ref_overlap = density_overlap(new_orbitals, reference_occupations, reference_orbitals, reference_occupations);
      Tbase mom_overlap = 0.0;
      if(occupation_difference(maximum_overlap_occupations, reference_occupations) > occupation_change_threshold_) {
        mom_overlap = density_overlap(new_orbitals, maximum_overlap_occupations, reference_orbitals, reference_occupations);
      }

      Tbase occ_overlap = 0.0;
      if(occupation_difference(reference_occupations, new_occupations) > occupation_change_threshold_) {
        occ_overlap = density_overlap(new_orbitals, new_occupations, reference_orbitals, reference_occupations);
      }

      return std::max(std::max(ref_overlap, mom_overlap), occ_overlap);
    }

    /// Attempt extrapolation with given weights
    bool attempt_extrapolation(const arma::Col<Tbase> & weights) {
      // Get the extrapolated Fock matrix
      auto fock(extrapolate_fock(weights));

      // Diagonalize the extrapolated Fock matrix
      auto diagonalized_fock = compute_orbitals(fock);
      auto & new_orbitals = diagonalized_fock.first;
      auto & new_orbital_energies = diagonalized_fock.second;

      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      // Reference calculation
      auto & reference_solution = orbital_history_[0];
      auto & reference_orbitals = reference_solution.first.first;
      auto reference_occupations = reference_solution.first.second;
      // Occupations corresponding to the reference orbitals
      auto maximum_overlap_occupations = determine_maximum_overlap_occupations(reference_occupations, reference_orbitals, new_orbitals);

      // Try first updating the orbitals, but not the occupations
      bool ref_success = add_entry(std::make_pair(new_orbitals, reference_occupations));

      // If that did not succeed, try maximum overlap occupations; it
      // might be that the orbitals changed order
      if(not frozen_occupations_ and not ref_success and occupation_difference(maximum_overlap_occupations, reference_occupations) > occupation_change_threshold_) {
        if(verbosity_ >= 10)
          printf("attempt_extrapolation: occupation difference to maximum overlap orbitals %e\n",occupation_difference(reference_occupations, new_occupations));
        ref_success = add_entry(std::make_pair(new_orbitals, maximum_overlap_occupations));
      }

      // Finally, if occupations have changed, also check if updating
      // the occupations lowers the energy
      bool occ_success = false;
      if(occupation_difference(reference_occupations, new_occupations) > occupation_change_threshold_) {
        occ_success = add_entry(std::make_pair(new_orbitals, new_occupations));
        if(verbosity_>=5) {
          if(occ_success)
            printf("Changing occupations decreased energy\n");
          else
            printf("Changing occupations failed to decrease energy\n");
        }
      }

      // Clean up history from incorrect occupation data
      if(occ_success) {
        size_t nremoved=0;
        for(size_t ihist=orbital_history_.size()-1;ihist>0;ihist--)
          if(occupation_difference(orbital_history_[0].first.second, orbital_history_[ihist].first.second) > occupation_change_threshold_) {
            nremoved++;
            orbital_history_.erase(orbital_history_.begin()+ihist);
          }
        if(verbosity_>=10)
        printf("Removed %i entries corresponding to bad occupations\n",nremoved);
      }

      // Extrapolation was a success if either worked
      return ref_success or occ_success;
    }

    /// Form list of rotation angles
    std::vector<OrbitalRotation> degrees_of_freedom() const {
      std::vector<OrbitalRotation> dofs;
      // Reference calculation
      auto & reference_solution = orbital_history_[0];
      auto & reference_occupations = reference_solution.first.second;

      // List occupied-occupied rotations, in case some orbitals are not fully occupied
      for(size_t iblock = 0; iblock < reference_occupations.size(); iblock++) {
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
    arma::Col<Tbase> precondition_search_direction(const arma::Col<Tbase> & gradient, const arma::Col<Tbase> & diagonal_hessian, double shift=0.1) const {
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
      const Orbitals<Torb> & reference_orbitals(orbital_history_[0].first.first);

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
        Tbase block_maximum = 0.5*M_PI/arma::abs(eval).max();
        maximum_step = std::min(maximum_step, block_maximum);
      }

      return maximum_step;
    }

    /// Rotate the orbitals through the given parameters
    Orbitals<Torb> rotate_orbitals(const arma::Col<Tbase> & x) const {
      auto kappa(form_rotation_matrices(x));

      // Rotate the orbitals
      Orbitals<Torb> new_orbitals(orbital_history_[0].first.first);
      for(size_t iblock=0; iblock < new_orbitals.size(); iblock++) {
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
    /// Evaluate the energy with a given orbital rotation vector
    OrbitalHistoryEntry<Torb, Tbase> evaluate_rotation(const arma::Col<Tbase> & x) const {
      // Rotate orbitals
      auto new_orbitals(rotate_orbitals(x));
      // Compute the Fock matrix
      auto reference_occupations = orbital_history_[0].first.second;

      auto density_matrix = std::make_pair(new_orbitals, reference_occupations);
      auto fock = fock_builder_(density_matrix);
      return std::make_pair(density_matrix, fock);
    }
    /// Take a steepest descent step
    void steepest_descent_step() {
      // Reference energy
      auto reference_energy = orbital_history_[0].second.first;

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
        Tbase reference_energy(orbital_history_[0].second.first);
        if(length==0.0)
          // We just get the reference energy
          return reference_energy;
        auto p(search_direction*length);
        auto entry = evaluate_rotation(p);
        // We can add the evaluated Fock matrix to the history
        if(length!=0.0)
          add_entry(entry.first, entry.second);
        if(verbosity_>=5)
          printf("Evaluated step %e with energy %.10f change from reference %e\n", length, entry.second.first, entry.second.first-reference_energy);
        return entry.second.first;
      };
      std::function<Tbase(Tbase)> scan_step = [this, search_direction](Tbase length){
        auto p(search_direction*length);
        auto entry = evaluate_rotation(p);
        return entry.second.first;
      };

      // Determine the maximal step size
      double Tmu = maximum_rotation_step(search_direction);
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

        double hh=cbrt(DBL_EPSILON);
        //double hh=1e-10;

        std::function<Tbase(Tbase)> eval = [this, search_direction, i](double xi){
          auto p(search_direction);
          p.zeros();
          p(i) = xi;
          return evaluate_rotation(p);
        };

        auto E2mi = eval(-2*hh);
        auto Emi = eval(-hh);
        auto Ei = eval(hh);
        auto E2i = eval(2*hh);

        double twop = (Ei-initial_energy)/hh;
        double threep = (Ei-Emi)/(2*hh);
        printf("i=%i twop=%e threep=%e\n",i,twop,threep);

        double h2diff = (Ei - 2*initial_energy + Emi)/(hh*hh);
        double h4diff = (-1/12.0*E2mi +4.0/3.0*Emi - 5.0/2.0*initial_energy + 4.0/3.0*Ei -1./12.0*E2i)/(hh*hh);

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
              step = std::min(10.0*predicted_step, step/2.0);
            }
          }
        }
      }
      if(not search_success) {
        throw std::runtime_error("Failed to find suitable step size.\n");
      }
    }

  public:
    /// Constructor
    SCFSolver(const arma::uvec & number_of_blocks_per_particle_type, const arma::Col<Tbase> & maximum_occupation, const arma::Col<Tbase> & number_of_particles, const FockBuilder<Torb, Tbase> & fock_builder, const std::vector<std::string> & block_descriptions) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder), block_descriptions_(block_descriptions), verbosity_(5) {
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
      // Compute orbitals
      auto diagonalized_fock = compute_orbitals(fock_guess);
      const auto & orbitals = diagonalized_fock.first;
      const auto & orbital_energies = diagonalized_fock.second;

      // Disable frozen occupations for the initialization
      frozen_occupations_ = false;
      orbital_occupations_ = update_occupations(orbital_energies);
      initialize_with_orbitals(orbitals, orbital_occupations_);
    }

    /// Initialize with precomputed orbitals and occupations
    void initialize_with_orbitals(const Orbitals<Torb> & orbitals, const OrbitalOccupations<Tbase> & orbital_occupations) {
      orbital_history_.clear();
      add_entry(std::make_pair(orbitals, orbital_occupations));
    }

    /// Fix the number of occupied orbitals per block
    void set_fixed_number_of_particles_per_block(const arma::Col<Tbase> & number_of_particles_per_block) {
      fixed_number_of_particles_per_block_ = number_of_particles_per_block;
    }

    /// Get frozen occupations
    bool get_frozen_occupations() const {
      return frozen_occupations_;
    }

    /// Set frozen occupations
    void set_frozen_occupations(bool frozen) {
      frozen_occupations_ = frozen;
    }

    /// Get verbosity
    int get_verbosity() const {
      return verbosity_;
    }

    /// Set verbosity
    void set_verbosity(int verbosity) {
      verbosity_ = verbosity;
    }

    /// Get convergence threshold
    double get_convergence_threshold() const {
      return convergence_threshold_;
    }

    /// Set verbosity
    void set_convergence_threshold(double convergence_threshold) {
      convergence_threshold_ = convergence_threshold;
    }

    /// Get the used error norm
    std::string get_error_norm() const {
      return error_norm_;
    }

    /// Set the used error norm
    void set_error_norm(const std::string & error_norm) {
      // Check that the norm is a valid option to Armadillo
      arma::vec test(1,arma::fill::ones);
      (void) arma::norm(test,error_norm.c_str());
      // store it
      error_norm_ = error_norm;
    }

    /// Get the maximum number of iterations
    size_t get_maximum_iterations() const {
      return maximum_iterations_;
    }

    /// Set the maximum number of iterations
    void set_maximum_iterations(size_t maxit) {
      maximum_iterations_ = maxit;
    }

    /// Get maximum_history_length
    int get_maximum_history_length() const {
      return maximum_history_length_;
    }

    /// Set maximum_history_length
    void set_maximum_history_length(int maximum_history_length) {
      maximum_history_length_ = maximum_history_length;
    }

    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tbase> & density) {
      // Compute the Fock matrix
      auto fock = fock_builder_(density);
      if(verbosity_>=5) {
        printf("Evaluated energy % .10f\n",fock.first);
      }
      return add_entry(density, fock);
    }

    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tbase> & density, const FockBuilderReturn<Torb, Tbase> & fock) {
      // Make a pair
      orbital_history_.push_back(std::make_pair(density, fock));

      if(orbital_history_.size()==1)
        // First try is a success by definition
        return true;
      else {
        // Otherwise we have to check if we lowered the energy
        bool return_value = fock.first-orbital_history_[0].second.first < energy_update_threshold_;
        // and now we resort the stack in increasing energy
        std::sort(orbital_history_.begin(), orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tbase> & a, const OrbitalHistoryEntry<Torb, Tbase> & b) {return a.second.first < b.second.first;});
        // Drop last entry if we are over the history length limit
        if(orbital_history_.size() > maximum_history_length_)
          orbital_history_.pop_back();

        // Figure out the need for a reset
        arma::Mat<Tbase> Bmat(diis_error_matrix());

        arma::Col<Tbase> Bval;
        arma::Mat<Tbase> Bvec;
        arma::eig_sym(Bval,Bvec,Bmat);
        if(arma::min(Bval) < 10*std::numeric_limits<Tbase>::epsilon()) {
          if(verbosity_)
            printf("Minimal eigenvalue of DIIS error matrix is %e, resetting history\n",arma::min(Bval));
          while(orbital_history_.size() > 1)
            orbital_history_.pop_back();
        }


        return return_value;
      }
    }

    /// Computes orbitals and orbital energies by diagonalizing the Fock matrix
    DiagonalizedFockMatrix<Torb,Tbase> compute_orbitals(const FockMatrix<Torb> & fock) const {
      DiagonalizedFockMatrix<Torb, Tbase> diagonalized_fock;
      // Allocate memory for orbitals and orbital energies
      diagonalized_fock.first.resize(fock.size());
      diagonalized_fock.second.resize(fock.size());

      // Diagonalize all blocks
      for(size_t iblock = 0; iblock < fock.size(); iblock++) {
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
        return orbital_history_[0].first.second;

      // Number of particles per block
      arma::Col<Tbase> number_of_particles = (fixed_number_of_particles_per_block_.n_elem == number_of_blocks_) ? fixed_number_of_particles_per_block_ : determine_number_of_particles_by_aufbau(orbital_energies);

      // Determine the number of occupied orbitals
      OrbitalOccupations<Tbase> occupations(orbital_energies.size());
      for(size_t iblock=0; iblock<orbital_energies.size(); iblock++) {
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

      if(orbital_history_.size() and verbosity_>=0) {
        // Check if occupations have changed
        const auto & old_occupations = orbital_history_[0].first.second;
        double occ_diff = occupation_difference(old_occupations, occupations);
        if(occ_diff > occupation_change_threshold_) {
          std::cout << "Warning: occupations changed by " << occ_diff << " from previous iteration\n";
          if(verbosity_>=0) {
            for(size_t iblock=0; iblock < occupations.size(); iblock++) {
              for(size_t iorb=0; iorb < occupations[iblock].n_elem; iorb++) {
                if(occupations[iblock][iorb] != old_occupations[iblock][iorb])
                  printf("iblock= %i iorb= %i new= %e old= %e\n",iblock,iorb,occupations[iblock][iorb],old_occupations[iblock][iorb]);
              }
            }
          }
        }
      }

      return occupations;
    }

    /// Run the SCF
    void run() {
      double old_energy = orbital_history_[0].second.first;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        double diis_error = arma::norm(diis_error_vector(0),error_norm_.c_str());

        if(verbosity_>=5) {
          printf("\n\nIteration %i: energy % .10f change %e DIIS error vector %s norm %e\n", iteration, orbital_history_[0].second.first, orbital_history_[0].second.first-old_energy, error_norm_.c_str(), diis_error);
          printf("History size %i\n",orbital_history_.size());
        }
        if(diis_error < convergence_threshold_) {
          printf("Converged to energy % .10f!\n", orbital_history_[0].second.first);
          break;
        }

        if(verbosity_>=5) {
          auto & reference_solution = orbital_history_[0];
          auto & occupations = reference_solution.first.second;
          for(size_t l=0;l<occupations.size();l++) {
            arma::uvec occ_idx(arma::find(occupations[l]>=1e-6));
            if(occ_idx.n_elem)
              occupations[l].subvec(0,arma::max(occ_idx)).t().print(block_descriptions_[l] + " occupations");
          }
        }

        if(iteration == 1) {
          // The orbitals can be bad, so start with a steepest descent
          // step to give DIIS a better starting point
          double old_energy = orbital_history_[0].second.first;
          steepest_descent_step();

        } else {
          // Form DIIS and ADIIS weights
          //arma::Col<Tbase> c2diis_w(c2diis_weights());
          arma::Col<Tbase> c2diis_w(c2diis_weights());
          if(verbosity_>=10) c2diis_w.print("C2DIIS weights");
          arma::Col<Tbase> c1diis_w(c1diis_weights());
          if(verbosity_>=10) c1diis_w.print("C1DIIS weights");
          arma::Col<Tbase> adiis_w;
          bool adiis_ok = true;
          try {
            adiis_w = adiis_weights();
            if(verbosity_>=10) adiis_w.print("ADIIS weights");
          } catch(std::logic_error) {
            // Bad weights
            adiis_ok = false;
            adiis_w.clear();
          };

          arma::Mat<Tbase> diis_errmat(diis_error_matrix());
          if(verbosity_>=5) {
            printf("C1DIIS extrapolated error norm %e\n",arma::norm(diis_errmat*c1diis_w,error_norm_.c_str()));
            printf("C2DIIS extrapolated error norm %e\n",arma::norm(diis_errmat*c2diis_w,error_norm_.c_str()));
            if(adiis_ok)
              printf("ADIIS extrapolated error norm %e\n",arma::norm(diis_errmat*adiis_w,error_norm_.c_str()));
          }

          // Form DIIS weights
          arma::Col<Tbase> diis_weights(orbital_history_.size(), arma::fill::zeros);
          if(diis_error < diis_threshold_) {
            if(verbosity_>=5) printf("C2DIIS extrapolation\n");
            diis_weights = c2diis_w;
            //printf("C1DIIS extrapolation\n");
            //diis_weights = c1diis_w;
          } else {
            if(not adiis_ok) {
              if(verbosity_>=5) printf("Large gradient and ADIIS minimization failed, taking a steepest descent step instead.\n");
              steepest_descent_step();
              continue;
            }

            if(diis_error < diis_epsilon_) {
              if(verbosity_>=10) printf("Mixed DIIS and ADIIS\n");
              double adiis_coeff = (diis_error-diis_threshold_)/(diis_epsilon_-diis_threshold_);
              double c2diis_coeff = 1.0 - adiis_coeff;
              diis_weights = adiis_coeff * adiis_w + c2diis_coeff * c2diis_w;
            } else {
              diis_weights = adiis_w;
            }
          }
          if(verbosity_>=10)
            diis_weights.print("Extrapolation weigths");

          // Perform extrapolation. If it does not lower the energy, we do
          // a scaled steepest descent step, instead.
          old_energy = orbital_history_[0].second.first;
          if(!attempt_extrapolation(diis_weights)) {
            if(verbosity_>=10) printf("Warning: did not go down in energy!\n");
            steepest_descent_step();
          }
        }
      }
    }

    /// Get the SCF solution
    DensityMatrix<Torb, Tbase> get_solution() const {
      return orbital_history_[0].first;
    }

    /// Get the Fock matrix
    FockBuilderReturn<Torb, Tbase> get_fock_build() const {
      return orbital_history_[0].second;
    }

    /// Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search
    void brute_force_search_for_lowest_configuration() {
      // Make sure we have a solution
      if(orbital_history_.size() == 0)
        run();
      else {
        double diis_error = arma::norm(diis_error_vector(0),error_norm_.c_str());
        if(diis_error >= diis_threshold_)
          run();
      }

      // Get the reference orbitals and orbital occupations
      auto & reference_solution = orbital_history_[0];
      auto reference_orbitals = reference_solution.first.first;
      auto reference_occupations = reference_solution.first.second;
      auto reference_energy = reference_solution.second.first;
      auto reference_fock = reference_solution.second.second;

      // We also need the orbital energies below
      auto diagonalized_fock = compute_orbitals(reference_fock);
      const auto & orbital_energies = diagonalized_fock.second;

      verbosity_ = 0;
      frozen_occupations_ = false;
      while(true) {
        // Count the number of particles in each block
        arma::Col<Tbase> number_of_particles_per_block(number_of_blocks_);
        for(size_t iblock=0; iblock<number_of_particles_per_block.size(); iblock++)
          number_of_particles_per_block[iblock] = arma::sum(reference_occupations[iblock]);
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
                list_of_energies.push_back(std::make_pair(trial_number, orbital_history_[0].second.first));
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
                        list_of_energies.push_back(std::make_pair(trial_number, orbital_history_[0].second.first));
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
          reference_orbitals = reference_solution.first.first;
          reference_occupations = reference_solution.first.second;
          reference_energy = reference_solution.second.first;
          reference_fock = reference_solution.second.second;
        } else {
          // Restore the reference calculation
          initialize_with_orbitals(reference_orbitals, reference_occupations);
          run();
          printf("Search converged!\n");
          break;
        }
      }
    }
  };
}
