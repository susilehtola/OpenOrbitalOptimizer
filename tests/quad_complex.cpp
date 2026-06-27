#include <openorbitaloptimizer/quad_support.hpp>
#include <openorbitaloptimizer/scfsolver.hpp>

template class OpenOrbitalOptimizer::SCFSolver<_Float128, true>;

int main(void) {
  return 0;
}
