# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

OpenOrbitalOptimizer is a header-only C++17 library for orbital optimization in quantum chemistry (Hartree-Fock, DFT, and related SCF methods). The library itself is problem-agnostic: it solves the fixed-point equation `FC = CE` in an orthonormal basis, and the caller supplies a Fock builder. Reference paper: J. Phys. Chem. A 129, 5651 (2025).

The library is templated on two types: `SCFSolver<Torb, Tbase>` where `Torb` is the orbital coefficient type (real or complex) and `Tbase` is the (always real) type used for orbital energies and occupations. Valid pairs: `<float,float>`, `<double,double>`, `<std::complex<float>,float>`, `<std::complex<double>,double>`.

## Layout

- `openorbitaloptimizer/scfsolver.hpp` — the entire SCF solver as one header (~2450 lines). All public API lives in `class SCFSolver` (see line 1599 onward). Includes DIIS/EDIIS/ADIIS history mixing and Aufbau occupation logic across arbitrary numbers of particle types and symmetry blocks.
- `openorbitaloptimizer/cg_optimizer.hpp` — small Polak-Ribière conjugate gradient routine used internally for line-search-style sub-problems.
- `openorbitaloptimizer/oda.hpp` — optimal damping algorithm step (included into the solver).
- `tests/atomtest.cpp` — the main functional test: an atomic SCF/DFT driver using a radial grid (IntegratorXX), Libxc functionals, and BSE-format JSON or ADF-format STO basis sets. Has restricted, unrestricted, and nuclear-electronic-orbital (NEO) drivers. CLI parsed via `tests/cmdline.h`.
- `tests/atomicsolver.hpp` — radial basis abstractions (GTO + STO) used only by `atomtest`.
- `tests/{float_float,cplxfloat_float,cplxdouble_double}.cpp` — compile-only template instantiation tests for the non-default `(Torb,Tbase)` pairs.
- `cmake/` — `OpenOrbitalOptimizerConfig.cmake.in` and an Armadillo-target healing helper for Conda Windows builds.

## Build and test

The library is header-only; "build" really means building the test suite.

```bash
cmake -S . -B objdir -DCMAKE_BUILD_TYPE=Release
cmake --build objdir
ctest --output-on-failure --test-dir objdir
```

A pre-existing `objdir/` is checked into the working tree and is the conventional build directory.

Test dependencies (only required when `OpenOrbitalOptimizer_BUILD_TESTING=ON`, which is the default for top-level builds): Armadillo (always required), Libxc, IntegratorXX, nlohmann_json. CI installs these via conda-forge.

CTest targets defined in `tests/CMakeLists.txt`:
- `openorbopt/atomtest/build` — builds `openorbopt-atomtest`.
- `openorbopt/atomtest/run1` — closed-shell oxygen with PBE/cc-pVDZ.
- `openorbopt/atomtest/run2` — open-shell oxygen (M=3) with PBE/cc-pVDZ.
- `openorbopt/{float-float,cplxfloat-float,cplxdouble-double}/build` — compile-only checks for the alternate template instantiations (these targets are `EXCLUDE_FROM_ALL`).

Run a single test by name, e.g.:
```bash
ctest --test-dir objdir -R openorbopt/atomtest/run2 --output-on-failure
```

Run the atom driver directly (useful when iterating on the solver):
```bash
./objdir/tests/openorbopt-atomtest --Z 8 --M 3 \
  --xfunc GGA_X_PBE --cfunc GGA_C_PBE \
  --basis tests/cc-pvdz.json
```
Other notable flags: `--Q` (charge), `--restricted` (-1=auto), `--Ngrid`, `--sto` (parse ADF STO basis instead of BSE JSON GTO), `--pbasis` (enables NEO mode), `--convthr`, `--lindepthresh`, `--verbosity`.

## Conventions

- Mozilla Public License 2.0; preserve the MPL header on existing files and add it to new ones.
- The library is a single header. Keep the public API on `SCFSolver` and put helper utilities in the existing namespaces (`OpenOrbitalOptimizer`, `OpenOrbitalOptimizer::ConjugateGradients`).
- The library does not depend on Libxc/IntegratorXX/nlohmann_json — only the tests do. Do not introduce these (or any other) dependencies into `openorbitaloptimizer/`.
- The library must remain instantiable for all four `(Torb,Tbase)` combinations; when changing template code, build the `openorbopt-instantiation-*` targets to verify.
- `objdir/`, `runs/`, `psi4/`, `openorbital.old/`, and various `*~` / `#*#` editor backups in the working tree are local artifacts — do not commit them.
