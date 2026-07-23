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

#include <iostream>
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
  m.doc() = "OpenOrbitalOptimizer Python bindings";

  using namespace OpenOrbitalOptimizer;
  using Solver = SCFSolver<double, double>;

  py::class_<Solver::OptionInfo>(m, "OptionInfo",
      "Descriptor for one solver option: key, type, writability, doc.")
    .def_readonly("key",      &Solver::OptionInfo::key)
    .def_readonly("type",     &Solver::OptionInfo::type)
    .def_readonly("writable", &Solver::OptionInfo::writable)
    .def_readonly("doc",      &Solver::OptionInfo::doc)
    .def("__repr__", [](const Solver::OptionInfo & o) {
      return std::string("<OptionInfo ") + o.key + " (" + o.type + ")>";
    });

  py::class_<Solver>(m, "SCFSolver",
      "SCF solver supporting fractional/degenerate occupations through\n"
      "skeleton density matrices and bi-level ODA + preconditioned CG\n"
      "minimization. Bound for SCFSolver<double, double>:\n"
      "double-precision real orbital coefficients and energies.\n\n"
      "Configure via set(key, value) / get_real / get_int / get_string.\n"
      "Enumerate every knob with SCFSolver.options().")
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

    // --- Settings façade -----------------------------------------------
    .def_static("options", &Solver::options,
         "Return the catalog of every solver option: name, type,\n"
         "writability, one-line description.",
         py::return_value_policy::reference)
    .def("set",
         py::overload_cast<const std::string&, double>(&Solver::set),
         py::arg("key"), py::arg("value"),
         "Set a real-valued option by name.")
    .def("set",
         py::overload_cast<const std::string&, int>(&Solver::set),
         py::arg("key"), py::arg("value"),
         "Set an integer-valued option by name.")
    .def("set",
         py::overload_cast<const std::string&, const std::string&>(&Solver::set),
         py::arg("key"), py::arg("value"),
         "Set a string-valued option by name.")
    .def("get_real", &Solver::get_real, py::arg("key"),
         "Get a real-valued option or diagnostic by name.")
    .def("get_int",  &Solver::get_int,  py::arg("key"),
         "Get an integer-valued option or diagnostic by name.")
    .def("get_string", &Solver::get_string, py::arg("key"),
         "Get a string-valued option by name.")
    .def("print_settings",
         [](const Solver & self) {
           self.print_settings(std::cout);
         },
         "Print every catalog entry with its current value to stdout.\n"
         "For a formatted string, prefer str(solver.settings).")
    .def("settings_as_string",
         [](const Solver & self) {
           std::ostringstream oss;
           self.print_settings(oss);
           return oss.str();
         },
         "Return the print_settings() output as a string.")
    .def_static("citation", &Solver::citation,
         "Canonical citation for the library (single line).")
    .def_static("print_citation",
         []() { Solver::print_citation(std::cout); },
         "Print the two-line 'please cite' block to stdout.")

    // --- Callback registration ----------------------------------------
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

    // --- Log sink -----------------------------------------------------
    .def("logger",
         py::overload_cast<std::function<void(int, const std::string &)>>(&Solver::logger),
         py::arg("sink") = std::function<void(int, const std::string &)>(),
         "Register a log sink. The callback receives ``(level, message)``\n"
         "where ``level`` is the minimum verbosity at which the message\n"
         "would print and ``message`` is the finished, formatted text\n"
         "(newlines included). Pass None to restore the stdout default.")
    .def("has_logger", &Solver::has_logger)

    // --- Initialization + drive ---------------------------------------
    .def("initialize_with_fock",
         &Solver::initialize_with_fock,
         py::arg("fock"))
    .def("initialize_with_orbitals",
         &Solver::initialize_with_orbitals,
         py::arg("orbitals"), py::arg("occupations"))
    .def("run",
         [](Solver& s, const std::string& methods) {
             if (!methods.empty()) s.set(std::string("methods"), methods);
             s.run();
         },
         py::arg("methods") = std::string(""),
         "Run the SCF loop. If ``methods`` is given, it is stored as the\n"
         "``methods`` setting before running; otherwise the current value\n"
         "is used. Tokens (case-insensitive, '+'-separated): 'DIIS',\n"
         "'ODA', 'CG', 'LBFGS'. Examples: 'DIIS', 'ODA + CG',\n"
         "'DIIS + ODA + CG' (the default).")

    // --- State queries ------------------------------------------------
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
         "True iff the DIIS error is at or below the effective threshold\n"
         "(or the user-supplied callback returned True). Equivalent to\n"
         "get_int(\"converged\") but returns bool.");
}
