#include "openorbitaloptimizer/scfsolver.hpp"

double rhf_solve_nosym_cpp(
                           void (*c_fock_builder)(int64_t Norb, const double *C, const double *n, double *F, double *Etot),
                           void (*c_print_callback)(const void *ptr),
                           int64_t Norb, int64_t Nelec, double *Cp, double *np, double conv) {
  // Guess orbitals
  arma::mat C(Cp, Norb, Norb, false, true);

  // Guess occupations
  arma::vec n(np, Norb, false, true);

  // Fock builder
  std::function<std::pair<double, OpenOrbitalOptimizer::FockMatrix<double>>(const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm)> fock_builder = [&](const OpenOrbitalOptimizer::DensityMatrix<double, double> & dm) {
    const auto & orbitals = dm.first;
    const auto & occupations = dm.second;

    std::vector<arma::mat> F(1);
    F[0].resize(orbitals[0].n_cols, orbitals[0].n_cols);
    F[0].zeros();
    double Etot;

    c_fock_builder(orbitals[0].n_cols, orbitals[0].memptr(), occupations[0].memptr(), F[0].memptr(), &Etot);
    return std::make_pair(Etot,F);
  };

  std::function<void(const std::map<std::string,std::any> &)> callback_function = [&](const std::map<std::string,std::any> & data) {
    c_print_callback(&data);
  };

  // Number of blocks per particle type
  arma::uvec number_of_blocks_per_particle_type({1});
  arma::vec maximum_occupation({2});
  arma::vec number_of_particles({(double) Nelec});
  std::vector<std::string> block_descriptions(1);
  block_descriptions[0]="alpha";

  // Need to pad orbitals into vector
  std::vector<arma::vec> nv(1);
  nv[0]=n;
  std::vector<arma::mat> Cv(1);
  Cv[0]=C;

  OpenOrbitalOptimizer::SCFSolver scfsolver(number_of_blocks_per_particle_type, maximum_occupation, number_of_particles, fock_builder, block_descriptions);
  scfsolver.verbosity(0);
  if (conv > 0.0) {
    scfsolver.convergence_threshold(conv);
  }
  scfsolver.callback_function(callback_function);
  scfsolver.initialize_with_orbitals(Cv, nv);
  scfsolver.run();

  // Get solution
  auto density_matrix = scfsolver.get_solution();
  auto orbitals = density_matrix.first;
  auto occupations = density_matrix.second;
  C = orbitals[0];
  n = occupations[0];

  auto fock_build = scfsolver.get_fock_build();
  return fock_build.first;
}

extern "C" {
  /** Solves RHF/RKS without symmetry. The orbital and Fock matrices are Norb x Norb. */
  double rhf_solve_nosym(void (*c_fock_builder)(int64_t Norb, const double *C, const double *n, double *F, double *Etot),
                         void (*c_print_callback)(const void *ptr),
                         int64_t Norb, int64_t Nelec, double* Cp, double *np, double conv) {
    return rhf_solve_nosym_cpp(c_fock_builder, c_print_callback, Norb, Nelec, Cp, np, conv);
  }

  /** Get an entry from a std::map */
  double ooo_get_double(const void *ptr, const char *label) {
    const std::map<std::string,std::any> *map = (const std::map<std::string,std::any> *) ptr;
    return std::any_cast<double>(map->at(label));
  }
  /** Get an entry from a std::map */
  int64_t ooo_get_int64(const void *ptr, const char *label) {
    const std::map<std::string,std::any> *map = (const std::map<std::string,std::any> *)(ptr);
    return (int64_t) std::any_cast<size_t>(map->at(label));
  }
}
