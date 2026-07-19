/*
 Copyright (C) 2026- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <openorbitaloptimizer/scfsolver.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
  m.doc() = "OpenOrbitalOptimizer Python bindings";

  using namespace OpenOrbitalOptimizer;
  using Solver = SCFSolver<double, double>;

  py::class_<Solver>(m, "SCFSolver",
      "SCF solver supporting fractional/degenerate occupations through\n"
      "skeleton density matrices and bi-level ODA + preconditioned CG\n"
      "minimization. Bound for SCFSolver<double, double>:\n"
      "double-precision real orbital coefficients and energies.")
    .def(py::init<IndexVector, Vector<double>, Vector<double>,
                  FockBuilder<double, double>, std::vector<std::string>>(),
         py::arg("number_of_blocks_per_particle_type"),
         py::arg("maximum_occupation"),
         py::arg("number_of_particles"),
         py::arg("fock_builder"),
         py::arg("block_descriptions"),
         "Construct an SCF solver. The Fock builder is a Python callable\n"
         "taking (orbitals: list[ndarray], occupations: list[ndarray])\n"
         "and returning (energy: float, fock: list[ndarray]).")
    .def("verbosity",
         py::overload_cast<int>(&Solver::verbosity),
         py::arg("verbosity"))
    .def("convergence_threshold",
         py::overload_cast<double>(&Solver::convergence_threshold),
         py::arg("threshold"))
    .def("maximum_iterations",
         py::overload_cast<size_t>(&Solver::maximum_iterations),
         py::arg("max_iter"))
    .def("number_of_fock_evaluations",
         &Solver::number_of_fock_evaluations,
         "Number of Fock-matrix builds performed by the most recent run().")
    .def("optimal_damping_degeneracy_threshold",
         py::overload_cast<double>(&Solver::optimal_damping_degeneracy_threshold),
         py::arg("threshold"))
    .def("set_batched_fock_builder",
         &Solver::set_batched_fock_builder,
         py::arg("builder"),
         "Register a batched Fock-builder callback. The callback receives a\n"
         "list of (orbitals, occupations) tuples and must return a list of\n"
         "(energy, fock) tuples in the same order. Used by the ODA polytope\n"
         "step to amortise integral / grid setup across the axis-vertex\n"
         "sweep. Pass None (or omit the call) to fall back to per-density\n"
         "evaluation through fock_builder.")
    .def("clear_batched_fock_builder",
         [](Solver & self) {
           self.set_batched_fock_builder(BatchedFockBuilder<double, double>{});
         },
         "Drop any registered batched Fock builder.")
    .def("has_batched_fock_builder",
         &Solver::has_batched_fock_builder)
    .def("initialize_with_fock",
         &Solver::initialize_with_fock,
         py::arg("fock"))
    .def("initialize_with_orbitals",
         &Solver::initialize_with_orbitals,
         py::arg("orbitals"), py::arg("occupations"))
    .def("run", &Solver::run,
         py::arg("methods") = std::string("DIIS + ODA + CG"),
         "Run the SCF loop with the methods named in the input string.\n"
         "Tokens (case-insensitive, '+'-separated): 'DIIS', 'ODA', 'CG', 'LBFGS'.\n"
         "Examples: 'DIIS', 'ODA', 'DIIS + ODA + CG' (the default),\n"
         "'ODA + CG' (former run_optimal_damping body).")
    .def("get_solution", &Solver::get_solution,
         py::arg("ihist") = 0,
         "Return (orbitals, occupations) of the ihist:th entry.")
    .def("get_orbitals", &Solver::get_orbitals,
         py::arg("ihist") = 0,
         "Return orbital coefficient blocks of the ihist:th entry.")
    .def("get_orbital_occupations", &Solver::get_orbital_occupations,
         py::arg("ihist") = 0,
         "Return orbital occupation blocks of the ihist:th entry.")
    .def("get_fock_matrix", &Solver::get_fock_matrix,
         py::arg("ihist") = 0,
         "Return Fock matrix blocks of the ihist:th entry.")
    .def("get_energy",
         py::overload_cast<size_t>(&Solver::get_energy, py::const_),
         py::arg("ihist") = 0,
         "Total energy of history entry ihist (0 = lowest energy).")
    .def("converged", &Solver::converged,
         "True iff the most recent run() reached the convergence\n"
         "threshold (or the user-supplied callback returned True).")
    .def("last_polytope_dimension", &Solver::last_polytope_dimension,
         "Skeleton dimension (N_par) of the most recent ODA call.")
    .def("last_active_rotation_count", &Solver::last_active_rotation_count,
         "Count of orbital-rotation DOFs inside degenerate groups at the\n"
         "iterate produced by the most recent ODA call. Default orbital-rotation burst\n"
         "length when orbital_rotation_steps_after_oda is left at 0.")
    .def("orbital_rotation_steps_after_oda",
         py::overload_cast<size_t>(&Solver::orbital_rotation_steps_after_oda),
         py::arg("n"),
         "Override the post-ODA orbital-rotation burst length. 0 restores the\n"
         "default of using last_active_rotation_count (with a floor of 1).");
}
