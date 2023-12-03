#include "scfsolver.hpp"
#include "cg_optimizer.hpp"
#include <algorithm>
#include <cfloat>


namespace OpenOrbitalOptimizer {
  using namespace OpenOrbitalOptimizer;



  // Instantiate myclass for the supported template type parameters
  template class SCFSolver<float, float>;
  template class SCFSolver<std::complex<float>, float>;
  template class SCFSolver<double, double>;
  template class SCFSolver<std::complex<double>, double>;
}
