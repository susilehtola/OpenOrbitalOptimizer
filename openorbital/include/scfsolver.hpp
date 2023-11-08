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

  /// The occupations for each orbitals are floating point numbers
  template<typename T> using OrbitalBlockOccupations = arma::Col<T>;
  /// The occupations for the whole set of orbitals are again a
  /// vector
  template<typename T> using OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;

  /// The pair of orbitals and occupations defines the density matrix
  template<typename Torb, typename Tocc> using DensityMatrix = std::pair<Orbitals<Torb>,OrbitalOccupations<Tocc>>;

  /// Orbital energies are stored as a vector of vectors
  template<typename T> using OrbitalEnergies = std::vector<arma::Col<T>>;

  /// A symmetry block in a Fock matrix is likewise defined by a N x
  /// N matrix
  template<typename T> using FockMatrixBlock = arma::Mat<T>;
  /// The whole set of Fock matrices is a vector of blocks
  template<typename T> using FockMatrix = std::vector<FockMatrixBlock<T>>;
  /// The return of Fock matrix diagonalization is
  template<typename T> using DiagonalizedFockMatrix = std::pair<Orbitals<T>,OrbitalEnergies<T>>;

  /// The Fock matrix builder returns the energy and the Fock
  /// matrices for each orbital block
  template<typename T> using FockBuilderReturn = std::pair<double, FockMatrix<T>>;
  /// The Fock builder takes in the orbitals and orbital occupations,
  /// and returns the energy and Fock matrices
  template<typename Torb, typename Tocc> using FockBuilder = std::function<FockBuilderReturn<Torb>(DensityMatrix<Torb, Tocc>)>;

  /// The history of orbital optimization is defined by the orbitals
  /// and their occupations - together the density matrix - and the
  /// resulting energy and Fock matrix
  template<typename Torb, typename Tocc> using OrbitalHistoryEntry = std::pair<DensityMatrix<Torb, Tocc>, FockBuilderReturn<Torb>>;
  /// The history is then a vector
  template<typename Torb, typename Tocc> using OrbitalHistory = std::vector<OrbitalHistoryEntry<Torb, Tocc>>;

  /// SCF solver class
  template<typename Torb, typename Tocc> class SCFSolver {
    /* Input data section */
    /// The number of orbital blocks per particle type (length ntypes)
    arma::uvec number_of_blocks_per_particle_type_;
    /// The maximal capacity of each orbital block
    arma::Col<Tocc> maximum_occupation_;
    /// The number of particles of each class in total (length ntypes, used to determine Aufbau occupations)
    arma::Col<Tocc> number_of_particles_;
    /// The Fock builder used to evaluate energies and Fock matrices
    FockBuilder<Torb, Tocc> fock_builder_;

    /* Internal data section */
    /// The orbital history used for convergence acceleration
    OrbitalHistory<Torb, Tocc> orbital_history_;
    /// Orbital energies, updated each iteration from the lowest-energy solution
    OrbitalEnergies<Torb> orbital_energies_;
  public:
    SCFSolver(const arma::uvec & number_of_blocks_per_particle_type, const arma::Col<Tocc> & maximum_occupation, const arma::Col<Tocc> & number_of_particles, const FockMatrix<Torb> & fock_guess, const FockBuilder<Torb, Tocc> & fock_builder);
    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tocc> & density);

    /// Computes orbitals and orbital energies by diagonalizing the Fock matrix
    DiagonalizedFockMatrix<Torb> compute_orbitals(const FockMatrix<Torb> & fock, Tocc level_shift = 0.0);
    /// Computes Aufbau occupations based on the current orbital energies
    OrbitalOccupations<Tocc> determine_occupations(const OrbitalEnergies<Torb> & orbital_energies);

    /// Perform DIIS extrapolation
    FockMatrix<Torb> diis_extrapolation();
  };
}
