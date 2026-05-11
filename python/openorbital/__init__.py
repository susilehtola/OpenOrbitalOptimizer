"""OpenOrbitalOptimizer Python bindings.

OpenOrbitalOptimizer is a header-only C++17 library for self-consistent
field (SCF) orbital optimization in quantum chemistry. The library
expects the caller to supply a Fock-builder callback that maps a
density matrix to an energy and Fock matrix in an orthonormal basis;
the library then performs DIIS / A-DIIS / E-DIIS extrapolation,
optimal damping with skeleton density matrices for degenerate shells,
and preconditioned Polak-Ribiere conjugate-gradient orbital rotations.

The single class exposed by this binding is ``SCFSolver`` (for real
orbital coefficients and real bases at double precision); the four
template instantiations of the underlying C++ library can be added if
the use case requires complex orbital coefficients or single precision.
"""

from ._ext import SCFSolver

__all__ = ["SCFSolver"]
