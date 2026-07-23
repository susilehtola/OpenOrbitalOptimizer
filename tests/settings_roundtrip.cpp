/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

// Exercises the SCFSolver settings façade: options() catalog,
// set(key, value) / get_*(key) round-trip on every entry, and
// invalid-key rejection. Runs at ctest time so a stale catalog or
// dispatch mismatch trips CI before it reaches downstream callers.

#include <openorbitaloptimizer/scfsolver.hpp>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

using OpenOrbitalOptimizer::SCFSolver;
using OpenOrbitalOptimizer::Matrix;
using OpenOrbitalOptimizer::Vector;
using OpenOrbitalOptimizer::IndexVector;
using OpenOrbitalOptimizer::FockBuilder;
using OpenOrbitalOptimizer::FockBuilderReturn;
using OpenOrbitalOptimizer::DensityMatrix;
using OpenOrbitalOptimizer::FockMatrix;

namespace {

int failures = 0;

#define REQUIRE(cond)                                                   \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

SCFSolver<double, double> make_solver() {
  IndexVector blocks_per_particle(1);
  blocks_per_particle(0) = 1;
  Vector<double> maxocc(1);
  maxocc(0) = 2.0;
  Vector<double> nparticles(1);
  nparticles(0) = 2.0;
  FockBuilder<double, double> fb =
    [](const DensityMatrix<double, double> &) {
      FockMatrix<double> F(1);
      F[0] = Matrix<double>::Identity(2, 2) * -1.0;
      return FockBuilderReturn<double, double>{0.0, F};
    };
  return SCFSolver<double, double>(blocks_per_particle, maxocc, nparticles,
                                   fb, {"s"});
}

}  // namespace

int main() {
  auto solver = make_solver();

  // Seed the history so read-only diagnostics that walk it
  // (converged, DIIS-error-based ones) have data to read.
  FockMatrix<double> guess(1);
  guess[0] = Matrix<double>::Identity(2, 2) * -1.0;
  solver.initialize_with_fock(guess);

  const auto & catalog = SCFSolver<double, double>::options();
  REQUIRE(!catalog.empty());

  // Round-trip every writable entry and read every read-only entry.
  for (const auto & opt : catalog) {
    const std::string key = opt.key;
    const std::string type = opt.type;
    if (opt.writable) {
      if (type == "real") {
        double v = 0.1234567;
        solver.set(key, v);
        REQUIRE(solver.get_real(key) == v);
      } else if (type == "int") {
        // frozen_occupations rides on int but coerces to bool; 1
        // round-trips through both cleanly for every int entry.
        int v = 1;
        solver.set(key, v);
        REQUIRE(solver.get_int(key) == v);
      } else if (type == "string") {
        // Every catalog string is validated on set(); pick an input
        // we know each accepts. `methods` is stored in canonical
        // uppercase, so the round-trip echoes back "ODA".
        std::string v_in, v_expect;
        if (key == "error_norm")   { v_in = "rms";  v_expect = "rms"; }
        else if (key == "methods") { v_in = "oda";  v_expect = "ODA"; }
        else                       { v_in = "";     v_expect = ""; }
        solver.set(key, v_in);
        REQUIRE(solver.get_string(key) == v_expect);
      } else {
        std::printf("FAIL: unknown type '%s' in catalog for '%s'\n",
                    type.c_str(), key.c_str());
        ++failures;
      }
    } else {
      // Read-only: just make sure the getter doesn't throw.
      try {
        if      (type == "real")   (void) solver.get_real(key);
        else if (type == "int")    (void) solver.get_int(key);
        else if (type == "string") (void) solver.get_string(key);
      } catch (const std::exception & e) {
        std::printf("FAIL: read-only get for '%s' threw: %s\n",
                    key.c_str(), e.what());
        ++failures;
      }
    }
  }

  // Unknown key rejection.
  bool threw = false;
  try { solver.set(std::string("no_such_setting"), 1.0); }
  catch (const std::invalid_argument &) { threw = true; }
  REQUIRE(threw);

  threw = false;
  try { (void) solver.get_int("no_such_setting"); }
  catch (const std::invalid_argument &) { threw = true; }
  REQUIRE(threw);

  // Wrong-type rejection: convergence_threshold is real, not int.
  threw = false;
  try { (void) solver.get_int("convergence_threshold"); }
  catch (const std::invalid_argument &) { threw = true; }
  REQUIRE(threw);

  // print_settings covers every catalog entry; every key appears in the
  // output, and print_citation names the paper.
  {
    std::ostringstream oss;
    solver.print_settings(oss);
    std::string dump = oss.str();
    for (const auto & o : catalog) {
      if (dump.find(o.key) == std::string::npos) {
        std::printf("FAIL: print_settings missed key '%s'\n", o.key);
        ++failures;
      }
    }
  }
  {
    std::ostringstream oss;
    SCFSolver<double, double>::print_citation(oss);
    std::string cite = oss.str();
    REQUIRE(cite.find("Lehtola") != std::string::npos);
    REQUIRE(cite.find("10.1021/acs.jpca.5c02110") != std::string::npos);
    REQUIRE(cite.find("J. Phys. Chem. A") != std::string::npos);
  }

  std::printf("%s: %d failure(s)\n", __FILE__, failures);
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
