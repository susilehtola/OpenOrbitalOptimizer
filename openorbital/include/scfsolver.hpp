#pragma once
#include <vector>
#include <armadillo>

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

    /* Internal data section */
    /// The number of blocks
    size_t number_of_blocks_;
    /// The orbital history used for convergence acceleration
    OrbitalHistory<Torb, Tbase> orbital_history_;
    /// Orbital energies, updated each iteration from the lowest-energy solution
    OrbitalEnergies<Tbase> orbital_energies_;

    /// Maximum number of iterations
    const size_t maximum_iterations_ = 128;
    /// Start to mix in DIIS at this error threshold
    const double diis_epsilon_ = 1e-1;
    /// Threshold for pure DIIS
    const double diis_threshold_ = 1e-2;
    /// Threshold for a change in occupations
    const double occupation_change_threshold_ = 1e-6;
    /// History length
    const double maximum_history_length_ = 7;
    /// Convergence threshold for orbital gradient
    const double convergence_threshold_ = 1e-7;
    /// Norm to use: rms
    const std::string error_norm_ = "fro";

    /* Internal functions */
    /// Get a block of the density matrix for the ihist:th entry
    arma::Mat<Torb> get_density_matrix_block(size_t ihist, size_t iblock) const;
    /// Get a block of the orbital occupations for the ihist:th entry
    arma::Mat<Torb> get_orbital_block(size_t ihist, size_t iblock) const;
    /// Get a block of the orbital occupations for the ihist:th entry
    arma::Col<Tbase> get_orbital_occupation_block(size_t ihist, size_t iblock) const;
    /// Get a block of the density matrix for the ihist:th entry
    arma::Mat<Torb> get_fock_matrix_block(size_t ihist, size_t iblock) const;
    /// Form DIIS error vector for ihist:th entry
    arma::Col<Tbase> diis_error_vector(size_t ihist) const;
    /// Form DIIS error matrix
    arma::Mat<Tbase> diis_error_matrix() const;
    /// Calculate C1-DIIS weights
    arma::Col<Tbase> c1diis_weights() const;
    /// Calculate C2-DIIS weights
    arma::Col<Tbase> c2diis_weights(double rejection_threshold = 10.0) const;

    /// Calculate ADIIS weights
    arma::Col<Tbase> adiis_weights() const;
    /// Form <D_i - D_0 | F_i - F_0>
    arma::Col<Tbase> adiis_linear_term() const;
    /// Form <D_i - D_j | F_i - F_j>
    arma::Mat<Tbase> adiis_quadratic_term() const;

    /// Computes the difference between orbital occupations
    Tbase occupation_difference(const OrbitalOccupations<Tbase> & old_occ, const OrbitalOccupations<Tbase> & new_occ) const;

    /// Perform DIIS extrapolation
    FockMatrix<Torb> extrapolate_fock(const arma::Col<Tbase> & weights) const;
    /// Attempt extrapolation with given weights
    bool attempt_extrapolation(const arma::Col<Tbase> & weights);

    /// Form list of rotation angles
    std::vector<OrbitalRotation> degrees_of_freedom() const;
    /// Formulate the orbital gradient vector
    arma::Col<Tbase> orbital_gradient_vector() const;
    /// Formulate the diagonal orbital Hessian
    arma::Col<Tbase> diagonal_orbital_hessian() const;
    /// Formulate the diagonal orbital Hessian
    arma::Col<Tbase> precondition_search_direction(const arma::Col<Tbase> & grad, const arma::Col<Tbase> & hess, double shift=0.1) const;

    /// Rotation matrices
    Orbitals<Torb> form_rotation_matrices(const arma::Col<Tbase> & x) const;
    /// Determine maximum step size
    Tbase maximum_rotation_step(const arma::Col<Tbase> & x) const;
    /// Rotate the orbitals through the given parameters
    Orbitals<Torb> rotate_orbitals(const arma::Col<Tbase> & x) const;
    /// Evaluate the energy with a given orbital rotation vector
    OrbitalHistoryEntry<Torb, Tbase> evaluate_rotation(const arma::Col<Tbase> & x) const;
    /// Take a steepest descent step
    void steepest_descent_step();

  public:
    /// Constructor
    SCFSolver(const arma::uvec & number_of_blocks_per_particle_type, const arma::Col<Tbase> & maximum_occupation, const arma::Col<Tbase> & number_of_particles, const FockBuilder<Torb, Tbase> & fock_builder, const std::vector<std::string> & block_descriptions);
    /// Initialize with Fock matrix
    void initialize_with_fock(const FockMatrix<Torb> & fock_guess);
    /// Initialize with orbitals
    void initialize_with_orbitals(const Orbitals<Torb> & orbitals, const OrbitalOccupations<Tbase> & orbital_occupations);

    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tbase> & density);
    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tbase> & density, const FockBuilderReturn<Torb, Tbase> & fock);

    /// Computes orbitals and orbital energies by diagonalizing the Fock matrix
    DiagonalizedFockMatrix<Torb,Tbase> compute_orbitals(const FockMatrix<Torb> & fock) const;
    /// Computes Aufbau occupations based on the current orbital energies
    OrbitalOccupations<Tbase> determine_occupations(const OrbitalEnergies<Tbase> & orbital_energies) const;

    /// Run the SCF
    void run();
  };
}
