#include "scfsolver.hpp"
#include <algorithm>
#include <cfloat>

#define OPTIM_ENABLE_ARMA_WRAPPERS
#include "optim.hpp"

namespace OpenOrbitalOptimizer {
  using namespace OpenOrbitalOptimizer;
  
  template<typename Torb, typename Tocc>
  SCFSolver<Torb, Tocc>::SCFSolver(const arma::uvec & number_of_blocks_per_particle_type, const arma::Col<Tocc> & maximum_occupation, const arma::Col<Tocc> & number_of_particles, const FockMatrix<Torb> & guess_fock, const FockBuilder<Torb, Tocc> & fock_builder, const std::vector<std::string> & block_descriptions) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder), block_descriptions_(block_descriptions) {
    // Run sanity checks
    size_t num_expected_blocks = arma::sum(number_of_blocks_per_particle_type_);
    if(maximum_occupation_.size() != num_expected_blocks)
      throw std::logic_error("Vector of maximum occupation is not of expected length!\n");
    if(number_of_particles_.size() != number_of_blocks_per_particle_type_.size())
      throw std::logic_error("Vector of number of particles is not of expected length!\n");
    number_of_blocks_ = num_expected_blocks;

    // Compute orbitals
    auto diagonalized_fock = compute_orbitals(guess_fock);
    const auto & orbitals = diagonalized_fock.first;
    orbital_energies_ = diagonalized_fock.second;
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
      bool return_value = fock.first < orbital_history_[0].second.first;
      // and now we resort the stack in increasing energy
      std::sort(orbital_history_.begin(), orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tocc> & a, const OrbitalHistoryEntry<Torb, Tocc> & b) {return a.second.first < b.second.first;});
      return return_value;
    }
  }

  template<typename Torb, typename Tocc>
  DiagonalizedFockMatrix<Torb, Tocc> SCFSolver<Torb, Tocc>::compute_orbitals(const FockMatrix<Torb> & fock) {
    DiagonalizedFockMatrix<Torb, Tocc> diagonalized_fock;
    // Allocate memory for orbitals and orbital energies
    diagonalized_fock.first.resize(fock.size());
    diagonalized_fock.second.resize(fock.size());

    // Diagonalize all blocks
    for(size_t iblock = 0; iblock < fock.size(); iblock++)
      arma::eig_sym(diagonalized_fock.second[iblock], diagonalized_fock.first[iblock], fock[iblock]);
    return diagonalized_fock;
  }

  template<typename Torb, typename Tocc>
  DiagonalizedFockMatrix<Torb, Tocc> SCFSolver<Torb, Tocc>::semicanonical_orbitals(const FockMatrix<Torb> & fock) {
    DiagonalizedFockMatrix<Torb, Tocc> diagonalized_fock;
    // Allocate memory for orbitals and orbital energies
    diagonalized_fock.first.resize(fock.size());
    diagonalized_fock.second.resize(fock.size());

    // Reference calculation
    auto & reference_solution = orbital_history_[0];
    auto & reference_orbitals = reference_solution.first.first;
    auto & reference_occupations = reference_solution.first.second;
    auto & reference_fock = reference_solution.second.second;
    
    // Diagonalize all blocks
    for(size_t iblock = 0; iblock < fock.size(); iblock++) {
      // Find the occupied and virtual blocks
      arma::uvec occupied_indices = arma::find(reference_occupations[iblock] > 0.0);
      arma::uvec virtual_indices = arma::find(reference_occupations[iblock] == 0.0);

      // Allocate memory for orbitals and eigenvalues
      auto & orbitals = diagonalized_fock.first[iblock];
      auto & energies = diagonalized_fock.second[iblock];
      orbitals.zeros(fock[iblock].n_rows, fock[iblock].n_cols);
      energies.zeros(fock[iblock].n_rows);
      
      arma::Col<Tocc> Eo, Ev;
      arma::Mat<Torb> Co, Cv;
      if(occupied_indices.n_elem) {
        arma::eig_sym(Eo, Co, reference_fock[iblock](occupied_indices, occupied_indices));
        orbitals.cols(0,occupied_indices.n_elem-1) = Co;
        energies.subvec(0,occupied_indices.n_elem-1) = Eo;
      }
      if(virtual_indices.n_elem) {
        arma::eig_sym(Ev, Cv, reference_fock[iblock](virtual_indices, virtual_indices));
        orbitals.cols(occupied_indices.n_elem,orbitals.n_cols-1) = Cv;
        energies.subvec(occupied_indices.n_elem, energies.n_elem-1) = Ev;
      }
    }
    return diagonalized_fock;
  }

  template<typename Torb, typename Tocc>
  std::vector<OrbitalRotation> SCFSolver<Torb, Tocc>::degrees_of_freedom() const {
    std::vector<OrbitalRotation> dofs;
    // Reference calculation
    auto & reference_solution = orbital_history_[0];
    auto & reference_occupations = reference_solution.first.second;

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

  template<typename Torb, typename Tocc>
  arma::Col<Tocc> SCFSolver<Torb, Tocc>::orbital_gradient(const FockMatrix<Torb> & fock) {
    // Get the degrees of freedom
    auto dof_list = degrees_of_freedom();
    arma::Col<Tocc> orb_grad;

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
      orb_grad(idof) = std::real(fock[iblock](iorb,jorb));
    }
    if constexpr (!arma::is_real<Torb>::value) {
        for(size_t idof = 0; idof < dof_list.size(); idof++) {
          auto dof(dof_list[idof]);
          auto iblock = std::get<0>(dof);
          auto iorb = std::get<1>(dof);
          auto jorb = std::get<2>(dof);
          orb_grad(dof_list.size() + idof) = std::imag(fock[iblock](iorb,jorb));
        }
      }
    return orb_grad;
  }

  template<typename Torb, typename Tocc>
  Orbitals<Torb> SCFSolver<Torb, Tocc>::rotate_orbitals(const Orbitals<Torb> & orbitals, const arma::Col<Tocc> & angle) {
    Orbitals<Torb> new_orbitals(orbitals.size());
    
    // Get the degrees of freedom
    auto dof_list = degrees_of_freedom();
    arma::Col<Tocc> orb_grad(dof_list.size());
    // Block them by symmetry
    std::vector<std::vector<std::tuple<arma::uword, arma::uword, size_t>>> blocked_dof(orbitals.size());
    for(size_t idof=0; idof<dof_list.size(); idof++) {
      auto dof = dof_list[idof];
      auto iblock = std::get<0>(dof);
      auto iorb = std::get<1>(dof);
      auto jorb = std::get<2>(dof);
      blocked_dof[iblock].push_back(std::make_tuple(iorb,jorb,idof));
    }

    // Rotate the orbitals
    for(size_t iblock=0; iblock < orbitals.size(); iblock++) {
      // Collect the rotation parameters
      arma::Mat<Torb> kappa(orbitals[iblock].n_cols, orbitals[iblock].n_cols, arma::fill::zeros);
      for(auto dof: blocked_dof[iblock]) {
        auto iorb = std::get<0>(dof);
        auto jorb = std::get<1>(dof);
        auto idof = std::get<2>(dof);
        kappa(iorb,jorb) = angle(idof);
      }
      // imaginary parameters
      if constexpr (!arma::is_real<Torb>::value) {
          for(auto dof: blocked_dof[iblock]) {
            auto iorb = std::get<0>(dof);
            auto jorb = std::get<1>(dof);
            auto idof = std::get<2>(dof);
            kappa(iorb,jorb) += Torb(0.0,angle(dof_list.size()+idof));
          }
        }
      // Antisymmetrize
      kappa -= arma::trans(kappa);

      // Rotate orbitals
      new_orbitals[iblock] = orbitals[iblock]*arma::expmat(kappa);
    }

    return new_orbitals;
  }

  template<typename Torb, typename Tocc>
  OrbitalOccupations<Tocc> SCFSolver<Torb, Tocc>::determine_occupations(const OrbitalEnergies<Tocc> & orbital_energies) {
    // Allocate the return
    OrbitalOccupations<Tocc> occupations(orbital_energies.size());
    for(size_t iblock=0; iblock<orbital_energies.size(); iblock++)
      occupations[iblock].zeros(orbital_energies[iblock].size());

    // Loop over particle types
    for(size_t particle_type = 0; particle_type < number_of_blocks_per_particle_type_.size(); particle_type++) {
      // Compute the offset in the block array
      size_t block_offset = (particle_type>0) ? arma::sum(number_of_blocks_per_particle_type_.subvec(0,particle_type-1)) : 0;
      
      // Collect the orbital energies with the block index and the in-block index for this particle type
      std::vector<std::tuple<Tocc, size_t, size_t>> all_energies;
      for(size_t iblock = block_offset; iblock < block_offset + number_of_blocks_per_particle_type_(particle_type); iblock++)
        for(size_t iorb = 0; iorb < orbital_energies[iblock].size(); iorb++)
          all_energies.push_back(std::make_tuple(orbital_energies[iblock](iorb), iblock, iorb));

      // Sort the energies in increasing order
      std::sort(all_energies.begin(), all_energies.end(), [](const std::tuple<Tocc, size_t, size_t> & a, const std::tuple<Tocc, size_t, size_t> & b) {return std::get<0>(a) < std::get<0>(b);});

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

    if(orbital_history_.size()) {
      // Check if occupations have changed
      double occupation_difference = 0.0;
      for(size_t iblock = 0; iblock<occupations.size(); iblock++)
        occupation_difference += arma::sum(arma::abs(occupations[iblock]-orbital_history_[0].first.second[iblock]));
      if(occupation_difference > 1e-6) {
        std::cout << "Warning: occupations changed by " << occupation_difference << " from previous iteration\n";
      }
    }

    return occupations;
  }

  // Function to extract error vectors
  template <typename Tmatrix, typename Tbase>
  arma::Col<Tbase> extract_error_vector(const arma::Mat<Tmatrix> & mat) {
    if constexpr (arma::is_real<Tmatrix>::value) {
      return arma::vectorise(mat);
    } else {
      return arma::join_cols(arma::vectorise(arma::real(mat)),arma::vectorise(arma::imag(mat)));
    }
  }

  template<typename Torb, typename Tocc>
  arma::Mat<Torb> SCFSolver<Torb, Tocc>::get_density_matrix_block(size_t ihist, size_t iblock) const {
    auto entry = orbital_history_[ihist];
    auto & density_matrix = entry.first;
    return density_matrix.first[iblock] * arma::diagmat(density_matrix.second[iblock]) * arma::trans(density_matrix.first[iblock]);
  }

  template<typename Torb, typename Tocc>
  arma::Mat<Torb> SCFSolver<Torb, Tocc>::get_fock_matrix_block(size_t ihist, size_t iblock) const {
    auto entry = orbital_history_[ihist];
    return entry.second.second[iblock];
  }

  template<typename Torb, typename Tocc>
  arma::Col<Tocc> SCFSolver<Torb, Tocc>::diis_error_vector(size_t ihist) const {
    // Form error vectors
    std::vector<arma::Col<Tocc>> error_vectors(orbital_history_[ihist].second.second.size());
    for(size_t iblock = 0; iblock<number_of_blocks_;iblock++) {
      // Error is measured by FPS-SPF = FP - PF, since we have a unit metric.
      auto F = get_fock_matrix_block(ihist, iblock);
      auto P = get_density_matrix_block(ihist, iblock);
      auto FP = F*P;
      error_vectors[iblock] = extract_error_vector<Torb, Tocc>(FP - arma::trans(FP));
    }

    // Compound error vector
    size_t nelem = 0;
    for(auto & block: error_vectors)
      nelem += block.size();

    arma::Col<Tocc> return_vector(nelem);
    size_t ioff=0;
    for(auto & block: error_vectors) {
      return_vector.subvec(ioff,ioff+block.size()-1) = block;
      ioff += block.size();
    }
    return return_vector;    
  }
  
  template<typename Torb, typename Tocc>
  arma::Col<Tocc> SCFSolver<Torb, Tocc>::c2diis_weights(double rejection_threshold) const {
    // Set up the DIIS error matrix
    size_t N=orbital_history_.size();
    arma::Mat<Tocc> B(N+1,N+1,arma::fill::zeros);

    // These are the orbital gradient dot products
    for(size_t ihist=0; ihist<N; ihist++) {
      arma::Col<Tocc> ei = diis_error_vector(ihist);
      for(size_t jhist=0; jhist<=ihist; jhist++) {
        arma::Col<Tocc> ej = diis_error_vector(jhist);
        B(ihist, jhist) = arma::dot(ei,ej);
      }
    }
    // Set last row and column to -1 except for N,N element
    B.row(N).fill(-1.0);
    B.col(N).fill(-1.0);
    B(N,N)=0.0;
    
    // Solve C2-DIIS eigenproblem
    arma::Mat<Tocc> evec;
    arma::Col<Tocc> eval;
    arma::eig_sym(eval, evec, B);

    // Normalize solution vectors
    arma::Mat<Tocc> candidate_solutions(evec.rows(0,N-1));
    for(size_t icol=0;icol<=N;icol++)
      candidate_solutions.col(icol) /= arma::sum(candidate_solutions.col(icol));

    // Find best solution that satisfies rejection threshold. Error
    // norms for the extrapolated vectors
    arma::Col<Tocc> error_norms(N+1,arma::fill::ones);
    error_norms *= std::numeric_limits<Tocc>::max();
    arma::Mat<Tocc> error_matrix(B.submat(0,0,N-1,N-1));
    for(size_t icol=0; icol <= N; icol++) {
      arma::Col<Tocc> soln = candidate_solutions.col(icol);
      // Skip solutions that have large elements
      if(arma::max(arma::abs(soln)) >= rejection_threshold)
        continue;
      // Compute extrapolated error
      arma::Col<Tocc> extrapolated_error = error_matrix * soln;
      error_norms(icol) = arma::norm(extrapolated_error, 2);
    }

    // Sort the solutions in the extrapolated error
    arma::uvec sortidx;
    sortidx = arma::sort_index(error_norms);

    arma::Col<Tocc> diis_weights;
    for(auto index: sortidx) {
      diis_weights = candidate_solutions.col(index);
      // Skip solutions that have extrapolated error in the same order
      // of magnitude as the used floating point precision
      if(error_norms(index) >= 5*std::numeric_limits<Tocc>::epsilon())
        break;
    }

    return diis_weights;
  }
  
  template<typename Torb, typename Tocc>
  FockMatrix<Torb> SCFSolver<Torb, Tocc>::extrapolate_fock(const arma::Col<Tocc> & weights) const {
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

  template<typename Torb, typename Tocc>
  arma::Col<Tocc> SCFSolver<Torb, Tocc>::adiis_linear_term() const {
    arma::Col<Tocc> ret(orbital_history_.size(),arma::fill::zeros);
    for(size_t ihist=0;ihist<ret.size();ihist++) {
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        auto dD = get_density_matrix_block(ihist, iblock) - get_density_matrix_block(0, iblock);
        auto dF = get_fock_matrix_block(ihist, iblock) - get_fock_matrix_block(0, iblock);
        auto result = std::real(arma::trace(dD*dF));
        ret[ihist] += result;
      }
    }
    return ret;
  }

  template<typename Torb, typename Tocc>
  arma::Mat<Tocc> SCFSolver<Torb, Tocc>::adiis_quadratic_term() const {
    arma::Mat<Tocc> ret(orbital_history_.size(),orbital_history_.size(),arma::fill::zeros);
    for(size_t ihist=0;ihist<ret.size();ihist++) {
      for(size_t jhist=0;jhist<=ihist;jhist++) {
        for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
          auto dD = get_density_matrix_block(ihist, iblock) - get_density_matrix_block(0, iblock);
          auto dF = get_fock_matrix_block(jhist, iblock) - get_fock_matrix_block(0, iblock);
          ret(ihist,jhist) = std::real(arma::trace(dD*dF));
        }
      }
    }
    return ret;
  }

  template<typename Torb, typename Tocc>
  arma::Col<Tocc> SCFSolver<Torb, Tocc>::adiis_weights() const {
    // Form linear and quadratic terms
    auto linear_term = adiis_linear_term();
    auto quadratic_term = adiis_quadratic_term();

    // OptimLib doesn't support float, so these routines are all in double precision.
    // Function to compute weights from the parameters
    std::function<arma::vec(const arma::vec & x)> x_to_weight = [](const arma::vec & x) { return arma::square(x)/arma::sum(arma::square(x)); };
    // and its Jacobian
    std::function<arma::mat(const arma::vec & x)> x_to_weight_jacobian = [x_to_weight](const arma::vec & x) {
      auto w(x_to_weight(x));
      auto xnorm = arma::norm(x,2);
      arma::mat jac(x.n_elem,x.n_elem,arma::fill::zeros);
      for(size_t i=0;i<x.n_elem;i++) {
        for(size_t j=0;j<x.n_elem;j++) {
          jac(i,j) -= w(j)*2.0*x(i)/xnorm;
        }
        jac(i,i) += 2.0*x(i)/xnorm;
      }
      return jac;
    };
    
    // Function to compute the ADIIS energy and gradient
    std::function<Tocc(const arma::vec & x, arma::vec *grad, void *opt_data)> adiis_energy_gradient = [linear_term, quadratic_term, x_to_weight, x_to_weight_jacobian](const arma::vec & x, arma::vec *grad, void *opt_data) {
      (void) opt_data;
      auto w(x_to_weight(x));
      if(grad!=nullptr) {
        *grad = x_to_weight_jacobian(x)*(linear_term + quadratic_term*w);
      }
      
      return arma::dot(linear_term, w) + 0.5*arma::dot(w, quadratic_term*w);
    };
      
    // Optimization
    arma::vec x(orbital_history_.size(),arma::fill::zeros);
    x(0)=1.0;
    bool success = optim::bfgs(x, adiis_energy_gradient, nullptr);
    if (success) {
      std::cout << "ADIIS optimization successful\n";
    } else {
      std::cout << "ADIIS optimization failed\n";
    }
      
    return arma::conv_to<arma::Col<Tocc>>::from(x_to_weight(x));
  }
  
  // Instantiate myclass for the supported template type parameters
  template class SCFSolver<float, float>;
  template class SCFSolver<std::complex<float>, float>;
  template class SCFSolver<double, double>;
  template class SCFSolver<std::complex<double>, double>;
}
