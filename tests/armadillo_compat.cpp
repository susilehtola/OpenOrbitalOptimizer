#include <openorbitaloptimizer/armadillo_compat.hpp>

// Instantiate the legacy Armadillo-typed wrapper for all four scalar
// pairs it supports. This is a compile-only sanity check.
template class OpenOrbitalOptimizer::Armadillo::SCFSolver<float, float>;
template class OpenOrbitalOptimizer::Armadillo::SCFSolver<double, double>;
template class OpenOrbitalOptimizer::Armadillo::SCFSolver<std::complex<float>, float>;
template class OpenOrbitalOptimizer::Armadillo::SCFSolver<std::complex<double>, double>;

int main(void) {
  return 0;
}
