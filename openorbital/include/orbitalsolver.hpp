#pragma once

namespace OpenOrbitalOptimizer {
  namespace SCFSolver {
    /// A symmetry block of orbitals is defined by the corresponding N x
    /// N matrix of orbital coefficients
    template<typename T> using OrbitalBlock = arma::Mat<T>;
    /// The set of orbitals is defined by a vector of orbital blocks,
    /// corresponding to each symmetry block of each particle type
    template<typename T> using Orbitals = std::vector<OrbitalBlock<T>>;

    /// The occupations for each orbitals are floating point numbers
    template<typename T> using OrbitalBlockOccupations = arma::Vec<T>;
    /// The occupations for the whole set of orbitals are again a
    /// vector
    template<typename T> using OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;

    /// A symmetry block in a Fock matrix is likewise defined by a N x
    /// N matrix
    template<typename T> using FockMatrixBlock = arma::Mat<T>;
    /// The whole set of Fock matrices is a vector of blocks
    template<typename T> using FockMatrix = std::vector<FockMatrixBlock<T>>;
    /// The Fock matrix builder returns the energy and the Fock
    /// matrices for each orbital block
    template<typename T> using FockBuilderReturn = std::pair<double,FockMatrix<T>>;
    /// The Fock builder takes in the orbitals and orbital
    /// occupations, and returns the energy and Fock matrices
    template<typename Torb, typename Tocc> using FockBuilder = std::function<FockBuilderReturn<T>(Orbitals<T>,OrbitalOccupations<Tocc>)>;

  ///
  /// A block of orbitals is defined by the pair (orbital matrix, Fock matrix)
  template<typename T> using OrbitalBlock = std::pair<arma::Mat<T>,arma::Mat<T>>;

  template<typename T> class SCFSolver {
    /// Builds energy and Fock matrix from given orbital blocks
     fock_builder_;
    /// Number of particles of each type, e.g. spin-up and spin-down electrons
    arma::vec number_of_particles_;
    /// Maximal occupations for each particle in each orbital block
    arma::umat maximal_occupation_;

    /// Occupation numbers for each parti
    std::vector<arma::vec> occupation_numbers_;
    /// Build density matrix
    std::vector<arma::Mat> form_density_matrix();
  public:
    /// Constructor
    SpinRestricted(const arma::Mat & core_hamiltonian, const std::function<std::pair<arma::Mat>(const std::vector<arma::Mat> &)> & jk_builder, const arma::uvec & maximal_occupation, double number_of_electrons) : core_hamiltonian_(core_hamiltonian), jk_builder_(jk_builder), maximal_occupation_(maximal_occupation), number_of_electrons_(number_of_electrons);
    /// Destructor
    ~SpinRestricted();
    /// Form Aufbau occupations
    void aufbau_occupations();
  };
}
