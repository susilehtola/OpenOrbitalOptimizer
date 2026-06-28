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
  using Solver = SCFSolver<double, false>;

  py::class_<Solver>(m, "SCFSolver",
      "SCF solver supporting fractional/degenerate occupations through\n"
      "skeleton density matrices and bi-level ODA + preconditioned CG\n"
      "minimization. Bound for SCFSolver<double, IsComplex=false>:\n"
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
    .def("initialize_with_fock",
         &Solver::initialize_with_fock,
         py::arg("fock"))
    .def("initialize_with_orbitals",
         &Solver::initialize_with_orbitals,
         py::arg("orbitals"), py::arg("occupations"))
    .def("run", &Solver::run,
         "Run the hybrid DIIS / ODA+CG SCF loop.")
    .def("run_optimal_damping", &Solver::run_optimal_damping,
         "Run the standalone ODA+CG loop without DIIS.")
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
         "Total energy of history entry ihist (0 = lowest energy).");
}
