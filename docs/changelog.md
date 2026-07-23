# Changelog

<!--
## vX.Y.0 / YYYY-MM-DD (Unreleased)

#### Breaking Changes

#### New Features

#### Enhancements

#### Bug Fixes

#### Misc.
-->


## v0.5.0 / (Unreleased)

#### Breaking Changes
* The per-option typed getters and setters on `SCFSolver` have been
  removed. Use the string-keyed façade instead:
  `solver.set(key, value)`, `solver.get_real / get_int /
  get_string(key)`, `SCFSolver::options()`. The Armadillo compatibility
  shim keeps its typed forwarders and now routes them through the
  façade.
* `SCFSolver::run()` no longer takes a method-mix argument; it
  consumes the `methods` setting, which is normalised to canonical
  uppercase on `set()`. Migrate `solver.run("ODA + CG")` to
  `solver.set("methods", "ODA + CG"); solver.run()`.
* Python bindings dropped the per-option typed methods (`verbosity`,
  `convergence_threshold`, `maximum_iterations`, ...). Use
  `solver.set(key, value)` or the attribute-style
  `solver.settings.<key> = value`. `SCFSolver` and `OptionInfo` are
  now importable from the `openorbital` package top level.

#### New Features
* SCF convergence threshold is now clamped to an arithmetic-precision
  floor: the effective threshold is
  `max(convergence_threshold, K * noise_floor)`, where `noise_floor`
  is a per-run estimate of the roundoff floor of the DIIS residual
  `C^dagger [F, P] C` frozen from the initial Fock. `K` defaults to
  10 and is tunable via `noise_safety_factor`. `__float128` runs are
  unaffected because their epsilon is tiny; the clamp mainly rescues
  low-precision runs from spinning below what the arithmetic can
  resolve. Callback-driven convergence
  (`callback_convergence_function`) is untouched.
* Introduce a string-keyed settings façade on `SCFSolver`:
  `set(key, value)`, `get_real/get_int/get_string(key)`, and a
  static `options()` catalog listing every knob and read-only
  diagnostic with its type and one-line description. Downstream
  callers now only need to know this triple to reach any setting,
  making JSON/dict-shaped configuration pipe-throughs trivial.
* Add `openorbital.Settings`: an attribute-style proxy that dispatches
  through the C++ catalog, so
  `solver.settings.convergence_threshold = 1e-9` is equivalent to
  `solver.set("convergence_threshold", 1e-9)` with the same catalog
  validation, `dir()` completion, and read-only diagnostic guards.
* The three PySCF drivers grew an `options=None` dict argument on
  `kernel()`; entries are forwarded via `solver.set(key, value)`
  before the run.

#### Enhancements
* `L-BFGS` history depth is now controlled by the shared
  `maximum_history_length` setting; the private
  `lbfgs_history_size_` knob has been folded away.
* `brute_force_search_for_lowest_configuration` now saves and
  restores `verbosity` and `frozen_occupations`, so calling it no
  longer permanently silences and thaws the parent solver.
* Removed the deprecated `run_optimal_damping()` alias on the
  Armadillo compatibility shim.


## v0.4.0 / 2026-07-20

#### Breaking Changes
* The linear-algebra backend switched from Armadillo to Eigen 3.4. The public
  API keeps the `SCFSolver<Torb, Tbase>` template signature via a compat shim,
  but callers that reached into Armadillo types directly must migrate to the
  new `OpenOrbitalOptimizer::Matrix<T>` / `Vector<T>` / `IndexVector` aliases.

#### New Features
* [\#38](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/38) Port the
  library and `atomtest` from Armadillo to Eigen 3, unlocking arbitrary-precision
  scalar types. Adds a `_Float128` (libquadmath) instantiation of `SCFSolver`
  via `openorbitaloptimizer/quad_support.hpp`.
* [\#10](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/10) Bi-level
  optimal-damping + preconditioned CG state machine with a skeleton-density-matrix
  polytope for degenerate shells, orbital-rotation bursts after ODA, and an
  L-BFGS phase. `SCFSolver::run("DIIS + ODA + CG")` is the new default; the
  method mix is user-selectable via `run("DIIS")`, `run("ODA + CG")`, etc.
* [\#10](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/10) Optional
  batched Fock-builder callback (`set_batched_fock_builder`) that receives a
  list of trial densities in one call, letting integral / grid setup amortise
  across the ODA polytope axis-vertex sweep.
* [\#10](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/10) New
  `--oda` / `--odadegthresh` / `--maxiter` flags in the `atomtest` driver.

#### Enhancements
* Python bindings expose the new API surface: `run(methods=...)`,
  `set_batched_fock_builder`, `has_batched_fock_builder`,
  `clear_batched_fock_builder`, `optimal_damping_degeneracy_threshold`,
  `orbital_rotation_steps_after_oda`, `last_polytope_dimension`,
  `last_active_rotation_count`, `number_of_fock_evaluations`, `converged`.
* The three PySCF drivers (`atomic`, `molecular`, `diatomic`) register
  batched Fock builders automatically and forward `methods` through
  `kernel(methods=...)`.

#### Misc.
* Armadillo is no longer a dependency of the header-only library. `atomtest`
  and the compat shim still require Eigen 3.4+; Armadillo is only pulled in
  by the compat shim's `<-> arma::Mat` conversion helpers when a caller
  explicitly opts in.


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

