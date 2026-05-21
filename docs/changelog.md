# Changelog

<!--
## vX.Y.0 / YYYY-MM-DD (Unreleased)

#### Breaking Changes

#### New Features

#### Enhancements

#### Bug Fixes

#### Misc.
-->


## v0.3.0 / 2026-05-21

#### Breaking Changes

#### New Features
* [\#34](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/34) Add Python interface!
  This covers the most common `SCFSolver<double, double>` case; others added upon request.
* [\#34](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/34) Add a PySCF integration layer
  with three usage-aware drivers, exposed as the `openorbital` package.
* [\#35](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/35) Add option `oda_restart_steps`
  to set the number of steps with no DIIS energy improvement after which to use ODA independently of
  DIIS history length. Previously used `maximum_history_length`/2

#### Enhancements

* [\#35](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/35) Explicitly symmetrize matices
  for DIIS error calculation to avoid numerical issues.

#### Bug Fixes
* Fix four correctness bugs in SCF solver and CG optimizer
  - get_energy: bounds check used > instead of >=, allowing out-of-bounds access when
    `ihist == orbital_history_.size()`.
  - matricise(vec, dim): the per-block offset was never advanced, so every block read from offset 0
    of the input vector.
  - steepest_descent line search: the "decrease step" branch used std::max, which actually grew the
    step whenever the parabolic prediction was larger than step/20, stalling the search.
  - CG Polak-Ribière: divided by dot(g_prev, g_prev) without guarding against a zero previous gradient,
    propagating NaN into the search direction. Fall back to steepest descent in that case.

#### Misc.
* [\#30](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/30) `max(idx)` has been deprecated
  in Armadillo in favor of `index_max()` so switching to the new syntax.
* [\#36](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/36) Use modern FindPython with pybind11 module.
* Added a `CLAUDE.md` file to aid agents.



## v0.2.0 / 2025-08-12

#### Breaking Changes

#### New Features

#### Enhancements
 * [\#26](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/26) add callback so caller can
   apply its own convergence criteria.
 * MESA should use Aufbau occupations not MOM
 * Undo minimum error criterion by Garza and Scuseria to avoid penalizing large steps leading to a decrease in energy
 * Increase `pure_ediis_factor`
 * Implement callback functionality

#### Bug Fixes
 * [\#27](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/27) fix a `Col::subvec()` error
   with minimal basis sets.
 * [\#27](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/27) fix UHF with frozen
   occupations by disabling ODA.

#### Misc.
 * Added user guide to readme
 * Deploy docs site
 * Fix GCC 15 warnings


## v0.1.0 / 2025-03-30

#### New Features
 * [\#20](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/20) Intf -- allow OOO and
   IntegratorXX to work on Windows.
 * All start-up functionality making library operational.

