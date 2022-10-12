2022-10-12

openorbital: a reusable orbital optimization library

This is a general library aimed for orbital optimization problems that
arise with various methods in quantum chemistry, ranging from
self-consistent field methods like Hartree-Fock and density functional
theory to more elaborate methods like multiconfigurational
self-consistent field theory, orbital-optimized coupled-cluster
theory, generalized valence bond theories, etc.

The library splits into two components:

- liborbopt for direct minimization of the energy via orbital rotation
  techniques

- liborbopt-scf for the traditional Roothaan scheme for orbital
  optimization
  