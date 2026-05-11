# OpenOrbitalOptimizer Python bindings

A `pybind11`-based binding that exposes `SCFSolver<double, double>` from
the header-only C++ library to Python. The bindings convert
`arma::Mat<double>` and `arma::Col<double>` to F-contiguous NumPy
arrays by copy; F order matches Armadillo's column-major layout, so no
implicit transpose is performed.

## Build

Requires `pybind11` (3.x), Armadillo, and a C++17 compiler.

```bash
cmake -S . -B objdir-py \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenOrbitalOptimizer_BUILD_PYTHON=ON \
      -DOpenOrbitalOptimizer_BUILD_TESTING=OFF
cmake --build objdir-py -j
```

The build tree's `objdir-py/python/openorbital/` is directly
importable; add it to `PYTHONPATH` to use without installing:

```bash
PYTHONPATH=$PWD/objdir-py/python python python/tests/test_one_electron.py
```

## Usage

The minimum-viable Python use:

```python
import numpy as np
import openorbital

def fock_builder(density):
    orbitals_list, occupations_list = density
    # ... compute F and E in the orthonormal basis you handed to the solver
    return energy, [F]

solver = openorbital.SCFSolver(
    number_of_blocks_per_particle_type=np.array([1], dtype=np.uintp),
    maximum_occupation=np.array([2.0]),
    number_of_particles=np.array([1.0]),
    fock_builder=fock_builder,
    block_descriptions=["s"],
)
solver.convergence_threshold(1e-7)
solver.initialize_with_fock([initial_fock_F_contiguous])
solver.run()
print(solver.get_energy(0))
```

## Conventions

* The basis is always treated as **orthonormal**. The caller is
  responsible for any AO -> orthonormal transformation (Löwdin,
  canonical, Cholesky, ...).
* Matrices are F-contiguous (column-major). NumPy arrays are
  automatically force-cast to F order on the way into the binding;
  arrays returned from the binding are F-contiguous by construction.
* Particle types are independent (electrons, alpha/beta, NEO
  protons). Each particle has one or more symmetry blocks; the
  per-block `maximum_occupation` carries the spatial degeneracy of
  that block (e.g. an atomic p-shell is one Armadillo block with
  `max_occ = 6` for restricted or `3` for unrestricted; the three
  magnetic components share the same radial Fock matrix).

## Planned drivers (not in this scaffold)

* `openorbital.pyscf.molecular`: an RHF/UHF/ROHF driver that wraps a
  PySCF mean-field instance as a Fock builder.
* `openorbital.pyscf.atomic`: an atomic SCF driver that builds one
  block per `(n, l, spin)` shell.
* `openorbital.pyscf.diatomic`: a diatomic SCF driver that builds
  one block per `(Lambda, spin)` (and parity for D_inf_h), with
  `max_occ` capturing the 2-fold m-degeneracy of pi, delta, ...
  representations.
