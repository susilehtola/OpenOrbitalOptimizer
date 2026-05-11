"""PySCF integration for OpenOrbitalOptimizer.

The submodule wraps PySCF mean-field instances as Fock-builder callbacks
for ``openorbital.SCFSolver``, providing a robust SCF convergence layer
on top of PySCF's integral and exchange-correlation machinery.
"""

from .molecular import OpenOrbitalSCF

__all__ = ["OpenOrbitalSCF"]
