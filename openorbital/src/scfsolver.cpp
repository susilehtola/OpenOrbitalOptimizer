#include "scfsolver.hpp"
#include <algorithm>

namespace OpenOrbitalOptimizer {
  using namespace OpenOrbitalOptimizer;
  
  template<typename Torb, typename Tocc>
  SCFSolver<Torb, Tocc>::SCFSolver(const arma::uvec & number_of_blocks_per_particle_type, const arma::Col<Tocc> & maximum_occupation, const arma::Col<Tocc> & number_of_particles, const FockMatrix<Torb> & guess_fock, const FockBuilder<Torb, Tocc> & fock_builder) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder) {
    // Run sanity checks
    size_t num_expected_blocks = arma::sum(number_of_blocks_per_particle_type_);
    if(maximum_occupation_.size() != num_expected_blocks)
      throw std::logic_error("Vector of maximum occupation is not of expected length!\n");
    if(number_of_particles_.size() != number_of_blocks_per_particle_type_.size())
      throw std::logic_error("Vector of number of particles is not of expected length!\n");

    // Compute orbitals
    auto diagonalized_fock = compute_orbitals(guess_fock);
    const auto & orbitals = diagonalized_fock.first();
    orbital_energies_ = diagonalized_fock.second();
    auto orbital_occupations = determine_occupations(orbital_energies_);
    add_entry(std::make_pair(orbitals, orbital_occupations));
  }

  template<typename Torb, typename Tocc>
  bool SCFSolver<Torb, Tocc>::add_entry(const DensityMatrix<Torb, Tocc> & density) {
    // Compute the Fock matrix
    auto fock = fock_builder_(density);
    // Make this into a pair
    orbital_history_.push_back(std::make_pair(density, fock));

    if(orbital_history_.size()==1)
      // First try is a success by definition
      return true;
    else {
      // Otherwise we have to check if we lowered the energy
      bool return_value = fock.first() < orbital_history_[0].second().first();
      // and now we resort the stack in increasing energy
      std::sort(orbital_history_.begin(), orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tocc> & a, const OrbitalHistoryEntry<Torb, Tocc> & b) {return a.second().first() < b.second().first();});
      return return_value;
    }
  }

  template<typename Torb, typename Tocc>
  DiagonalizedFockMatrix<Torb> SCFSolver<Torb, Tocc>::compute_orbitals(const FockMatrix<Torb> & fock, Tocc level_shift) {
    DiagonalizedFockMatrix<Torb> diagonalized_fock;
    // Allocate memory for orbitals and orbital energies
    diagonalized_fock.first().resize(fock.size());
    diagonalized_fock.second().resize(fock.size());

    // Diagonalize all blocks
    if(level_shift == 0.0) {
      for(size_t iblock = 0; iblock < fock.size(); iblock++)
        arma::eig_sym(diagonalized_fock.second()[iblock], diagonalized_fock.first()[iblock], fock[iblock]);
    } else {
      assert(orbital_history_.size()>0);
      for(size_t iblock = 0; iblock < fock.size(); iblock++) {
        // Form the virtual density matrix in this block. First get the reference density matrix
        const auto & reference_density_matrix = orbital_history_[0].first();
        const auto & reference_orbitals = reference_density_matrix.first();
        const auto & reference_occupations = reference_density_matrix.second();
        // Find orbitals with zero occupation
        arma::uvec virtual_indices = arma::find(reference_occupations == 0.0);
        // Virtual space projector is then
        arma::Mat<Torb> Pvirt = reference_orbitals.cols(virtual_indices).T * reference_orbitals.cols(virtual_indices);

        // Shift virtual orbitals up by the level shift
        arma::eig_sym(diagonalized_fock.second()[iblock], diagonalized_fock.first()[iblock], fock[iblock] + level_shift * Pvirt);
      }
    }
  }

  template<typename Torb, typename Tocc>
  OrbitalOccupations<Tocc> SCFSolver<Torb, Tocc>::determine_occupations(const OrbitalEnergies<Torb> & orbital_energies) {
    // Allocate the return
    OrbitalOccupations<Tocc> occupations(orbital_energies.size());
    for(size_t iblock=0; iblock<orbital_energies.size(); iblock++)
      occupations[iblock].zeros(orbital_energies[iblock].size());

    // Loop over particle types
    for(size_t particle_type = 0; particle_type < number_of_blocks_per_particle_type_.size(); particle_type++) {
      // Compute the offset in the block array
      size_t block_offset = (particle_type>0) ? arma::sum(number_of_blocks_per_particle_type_.subvec(0,particle_type-1)) : 0;
      
      // Collect the orbital energies with the block index and the in-block index for this particle type
      std::vector<std::tuple<Torb, size_t, size_t>> all_energies;
      for(size_t iblock = block_offset; iblock < block_offset + number_of_blocks_per_particle_type_(particle_type); iblock++)
        for(size_t iorb = 0; iorb < orbital_energies[iblock].size(); iorb++)
          all_energies.push_back(std::make_tuple(orbital_energies[iblock](iorb), iblock, iorb));

      // Sort the energies in increasing order
      std::sort(all_energies.begin(), all_energies.end(), [](const std::tuple<Torb, size_t, size_t> & a, const std::tuple<Torb, size_t, size_t> & b) {return a.first() < b.first();});

      // Fill the orbitals in increasing energy. This is how many
      // particles we have to place
      Tocc num_left = number_of_particles_(particle_type);
      for(auto fill_orbital : all_energies) {
        // Increase number of occupied orbitals
        auto iblock = std::get<1>(fill_orbital);
        auto iorb = std::get<2>(fill_orbital);
        occupations[iblock](iorb) = std::min(maximum_occupation_(iblock), num_left);
        // It is probably safer to do this for the sake of floating
        // point accuracy, since comparison to zero can be difficult
        if(num_left <= maximum_occupation_(iblock))
          break;
        num_left -= occupations[iblock](iorb);
      }
    }

    return occupations;
  }

  template<typename Torb, typename Tocc>
  FockMatrix<Torb> SCFSolver<Torb, Tocc>::diis_extrapolation() {
    
  }
  
}
