# OpenOrbitalOptimizer: a reusable orbital optimization library

This is a general library aimed for orbital optimization problems that
arise with various methods in quantum chemistry, ranging from
self-consistent field (SCF) methods like Hartree-Fock (HF) and density
functional theory (DFT) to more elaborate methods like
multiconfigurational self-consistent field theory, orbital-optimized
coupled-cluster theory, generalized valence bond theories, etc.

This library is designed to do Roothaan-type SCF for HF and DFT
calculations using direct inversion in the iterative subspace (DIIS)
type methods. General algorithms for direct minimization on the
Grassmann and Stiefel manifolds are possibly forthcoming. However,
direct minimization can already be more readily undertaken with the
second-order implementations in the
[OpenTrustRegion](https://github.com/eriksen-lab/opentrustregion)
library, which does require

The library is open source, licensed under the permissive Mozilla
Public License 2.0.

## Usage

The library has been described in the open-access article
[J. Phys. Chem. A 129, 5651
(2025)](https://doi.org/10.1021/acs.jpca.5c02110). In short, the
library is primarily aimed for solving the fixed-point equations of
Hartree-Fock and Kohn-Sham density-functional theory: when the
molecular or crystalline orbitals are expanded in a basis set, the
variation of the energy with respect to the orbital expansion
coefficients C results in the generalized eigenvalue equation `FC =
SCE`. This equation is in practice always solved in the orthonormal
basis, obtained by diagonalizing the overlap matrix `S` (and omitting
any linearly dependent eigenvectors). Therefore, OpenOrbitalOptimizer
is designed to solve the equations directly in the orthonormal basis,
`FC = CE`.

The library supports an arbitrary number of symmetries and an
arbitrary number of particle types, and it can solve both real-valued
and complex-valued self-consistent field equations. The library does
not know anything about the problem it is solving; this is defined by
the user. The library employs
[Armadillo](https://arma.sourceforge.net/) for linear algebra in C++, and
is templated as
```
  template<typename Torb, typename Tbase> class SCFSolver;
```
`Torb` is the datatype for the orbitals, while `Tbase` is the datatype
for the orbital energies and occupations.  The template arguments
`<double, double>` request the library to be instantiated with
`arma::Mat<double>` for the orbitals, and `arma::Col<double>` for the
orbital energies. Since the Fock matrix is Hermitian, the orbital
energy datatype should always be real. Valid choices for the template
types are therefore `<float, float>` or `<double, double>` for
single-precision or double-precision orbitals, and
`<std::complex<float>, float>` or `<std::complex<double>, double>` for
single-precision or double-precision complex-orbitals.


A minimal example of using the library is as follows:
```
OpenOrbitalOptimizer::SCFSolver<double, double> scfsolver(number_of_blocks_per_particle_type, maximum_occupations, number_of_particles, fock_builder, block_descriptions);
scfsolver.convergence_threshold(maximum_diis_error); // set convergence threshold for orbital gradient
scfsolver.callback_function(callback_function); // set a custom callback function, used to customize the printout of the library
scfsolver.initialize_with_fock_matrix(guess_fock_matrix); // initializes the orbitals and occupations by diagonalizing the given guess for the Fock matrix
// scfsolver.initialize_with_orbitals(guess_orbitals, guess_orbital_occupations); // alternatively, one can also initialize the solver by specifying the orbitals and occupations directly
scfsolver.run(); // runs the solver
```

The arguments to the constructor are as follows
* `number_of_blocks_per_particle_type` is an unsigned integer vector (`arma::uvec`) specifying the number of symmetry blocks for each particle. The number of elements specifies the number of particle types `Ntypes`, and the sum of the elements specifies the total number of symmetry blocks `Nblocks`.
* `maximum_occupations` is a floating-point vector of the orbital energy datatype of length `Nblocks`. The elements specify the maximum occupation per orbital in each symmetry block.
* `number_of_particles` is a floating-point vector of the orbital energy datatype of length `Ntypes`. The elements specify the number of particles of each type.
* `fock_builder` is a function that takes in a `DensityMatrix`, which is an alias for `std::pair<Orbitals<Torb>,OrbitalOccupations<Tbase>>`: this is a pair of a vector of orbital coefficient matrices, and a vector of orbital occupation vectors. The function returns a `std::pair` of the total energy (`Tbase`) and a vector of Fock matrices for each block.
* `block_descriptions` is a `std::vector` of `std::string` that contains human-readable descriptions of the blocks.

To use the library, you basically only need to write the Fock build function. For a non-trivial example, see the test code for [spin-restricted](https://github.com/susilehtola/OpenOrbitalOptimizer/blob/e4d54d016ded33fc2ff97e353fc9047d32f34316/tests/atomtest.cpp#L472) and [spin-unrestricted](https://github.com/susilehtola/OpenOrbitalOptimizer/blob/e4d54d016ded33fc2ff97e353fc9047d32f34316/tests/atomtest.cpp#L610) atoms using spherically symmetric orbitals and orbital occupations, as in [Phys. Rev. A 101, 012516 (2020)](https://doi.org/10.1103/PhysRevA.101.012516) and [J. Chem. Theory Comput. 19, 2502 (2023)](https://doi.org/10.1021/acs.jctc.3c00183).
In these examples, the `s`, `p`, `d`, and `f` orbitals are all treated equivalently, and the SCF problem splits into coupled radial subproblems. Each `s`, `p`, `d`, and `f` orbital fits 2, 6, 10, and 14 (1, 3, 5, and 7) electrons in spin-unrestricted (spin-unrestricted) mode.

For another example, see [the implementation of the nuclear-electronic orbital method in the ERKALE code](https://github.com/susilehtola/erkale/blob/141d6ffe458534d5babbcecd1c2d72c4ca9f07d4/src/contrib/neo.cpp#L470), which was used in the article describing OpenOrbitalOptimizer.

## Dependencies

#### header-only library

* [Armadillo](https://gitlab.com/conradsnicta/armadillo-code)

#### testing

* [Libxc](https://gitlab.com/libxc/libxc)
* [IntegratorXX](https://github.com/wavefunction91/IntegratorXX)
* [nlohmann-JSON](https://github.com/nlohmann/json)
