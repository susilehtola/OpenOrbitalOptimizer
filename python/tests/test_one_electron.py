"""Smoke test: 1-electron SCF in an orthonormal basis.

With no electron-electron interaction the Fock matrix is the (fixed)
core Hamiltonian and the SCF converges in one iteration to the lowest
eigenvalue of h. The test validates that the C++ -> Python binding,
the Fock-builder callback, and the SCF driver all wire together
correctly without invoking any quantum-chemistry machinery.
"""

import numpy as np
import openorbital


def test_one_electron_diag():
    # Four-dimensional orthonormal basis with diagonal core Hamiltonian.
    h = np.diag(np.array([-1.0, -0.5, 0.25, 1.0]))

    def fock_builder(density):
        orbitals, occupations = density
        # 1-electron: F = h, E = tr(h P) = sum_i n_i (C^T h C)_ii
        C = orbitals[0]
        n = occupations[0]
        F = h.copy(order="F")
        energy = float(np.einsum("i,ki,kj,ji->", n, C, h, C))
        return energy, [F]

    solver = openorbital.SCFSolver(
        number_of_blocks_per_particle_type=np.array([1], dtype=np.uintp),
        maximum_occupation=np.array([2.0]),       # closed-shell s-block
        number_of_particles=np.array([1.0]),       # one electron
        fock_builder=fock_builder,
        block_descriptions=["s"],
    )
    solver.set("verbosity", 0)
    solver.set("convergence_threshold", 1e-10)
    solver.set("maximum_iterations", 50)
    solver.initialize_with_fock([h.copy(order="F")])
    solver.run()

    energy = solver.get_energy(0)
    occupations = solver.get_orbital_occupations(0)[0]
    np.testing.assert_allclose(energy, -1.0, atol=1e-9)
    # Single electron should occupy the lowest orbital exclusively.
    leading = np.argmax(occupations)
    np.testing.assert_allclose(occupations[leading], 1.0, atol=1e-9)
    np.testing.assert_allclose(
        occupations[np.arange(len(occupations)) != leading], 0.0, atol=1e-9
    )


if __name__ == "__main__":
    test_one_electron_diag()
    print("OK")
