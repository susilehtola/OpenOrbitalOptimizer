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

#include <armadillo>
#include <cstring>

#include <openorbitaloptimizer/scfsolver.hpp>

namespace py = pybind11;

/// pybind11 type casters between arma::Mat / arma::Col / arma::uvec and
/// NumPy arrays. Conversion is by copy in both directions; the F-order
/// layout of arma::Mat is exposed as an F-contiguous numpy array on the
/// Python side, which matches the convention used elsewhere in the
/// library and avoids implicit transposes.
namespace pybind11 { namespace detail {

template<>
struct type_caster<arma::Mat<double>> {
public:
  PYBIND11_TYPE_CASTER(arma::Mat<double>, _("numpy.ndarray[float64, ndim=2]"));

  bool load(handle src, bool /*convert*/) {
    auto arr = array_t<double, array::f_style | array::forcecast>::ensure(src);
    if (!arr) return false;
    if (arr.ndim() != 2) return false;
    value = arma::Mat<double>(arr.data(), arr.shape(0), arr.shape(1));
    return true;
  }

  static handle cast(const arma::Mat<double> & src, return_value_policy, handle) {
    array_t<double, array::f_style> out(
      std::vector<py::ssize_t>{(py::ssize_t)src.n_rows, (py::ssize_t)src.n_cols}
    );
    if (src.n_elem > 0)
      std::memcpy(out.mutable_data(), src.memptr(), src.n_elem * sizeof(double));
    return out.release();
  }
};

template<>
struct type_caster<arma::Col<double>> {
public:
  PYBIND11_TYPE_CASTER(arma::Col<double>, _("numpy.ndarray[float64, ndim=1]"));

  bool load(handle src, bool /*convert*/) {
    auto arr = array_t<double, array::forcecast>::ensure(src);
    if (!arr) return false;
    if (arr.ndim() != 1) return false;
    value = arma::Col<double>(arr.data(), arr.shape(0));
    return true;
  }

  static handle cast(const arma::Col<double> & src, return_value_policy, handle) {
    array_t<double> out(std::vector<py::ssize_t>{(py::ssize_t)src.n_elem});
    if (src.n_elem > 0)
      std::memcpy(out.mutable_data(), src.memptr(), src.n_elem * sizeof(double));
    return out.release();
  }
};

template<>
struct type_caster<arma::uvec> {
public:
  PYBIND11_TYPE_CASTER(arma::uvec, _("numpy.ndarray[uint, ndim=1]"));

  bool load(handle src, bool /*convert*/) {
    auto arr = array_t<arma::uword, array::forcecast>::ensure(src);
    if (!arr) return false;
    if (arr.ndim() != 1) return false;
    value = arma::uvec(arr.data(), arr.shape(0));
    return true;
  }

  static handle cast(const arma::uvec & src, return_value_policy, handle) {
    array_t<arma::uword> out(std::vector<py::ssize_t>{(py::ssize_t)src.n_elem});
    if (src.n_elem > 0)
      std::memcpy(out.mutable_data(), src.memptr(), src.n_elem * sizeof(arma::uword));
    return out.release();
  }
};

}} // namespace pybind11::detail

PYBIND11_MODULE(_ext, m) {
  m.doc() = "OpenOrbitalOptimizer Python bindings";

  using namespace OpenOrbitalOptimizer;
  using Solver = SCFSolver<double, double>;

  py::class_<Solver>(m, "SCFSolver",
      "SCF solver supporting fractional/degenerate occupations through\n"
      "skeleton density matrices and bi-level ODA + preconditioned CG\n"
      "minimization. Templated on (double, double).")
    .def(py::init<arma::uvec, arma::Col<double>, arma::Col<double>,
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
