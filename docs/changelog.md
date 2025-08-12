# Changelog

<!--
## vX.Y.0 / 2025-MM-DD (Unreleased)

#### Breaking Changes

#### New Features

#### Enhancements

#### Bug Fixes

#### Misc.
-->


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

