"""Exercise the Settings attribute-style proxy over SCFSolver.

These checks are independent of PySCF and the Fock builder -- they
only touch the catalog dispatch and the underlying set/get façade.
"""

import numpy as np
import pytest

from openorbital import SCFSolver, Settings


@pytest.fixture
def solver():
    def fock_builder(orbitals_occs):
        # Not called in these tests; just needs to be a valid callable.
        raise RuntimeError("fock builder should not be invoked here")

    return SCFSolver(
        number_of_blocks_per_particle_type=np.array([1], dtype=np.uintp),
        maximum_occupation=np.array([2.0]),
        number_of_particles=np.array([2.0]),
        fock_builder=fock_builder,
        block_descriptions=["s"],
    )


def test_options_catalog_nonempty():
    catalog = SCFSolver.options()
    assert len(catalog) > 0
    for entry in catalog:
        assert entry.type in ("real", "int", "string")


def test_settings_roundtrip_real(solver):
    solver.settings.convergence_threshold = 1e-9
    assert solver.settings.convergence_threshold == pytest.approx(1e-9)


def test_settings_roundtrip_int(solver):
    solver.settings.maximum_iterations = 33
    assert solver.settings.maximum_iterations == 33


def test_settings_roundtrip_string(solver):
    solver.settings.error_norm = "fro"
    assert solver.settings.error_norm == "fro"


def test_settings_methods_uppercase(solver):
    # Stored form is canonical uppercase, per the C++-side normaliser.
    solver.settings.methods = "oda + cg"
    assert solver.settings.methods == "ODA + CG"


def test_settings_bool_via_int(solver):
    solver.settings.frozen_occupations = 1
    assert solver.settings.frozen_occupations == 1
    solver.settings.frozen_occupations = 0
    assert solver.settings.frozen_occupations == 0


def test_settings_readonly_diagnostic_rejects_write(solver):
    with pytest.raises(AttributeError, match="read-only"):
        solver.settings.noise_floor = 1e-6


def test_settings_unknown_name_raises(solver):
    with pytest.raises(AttributeError, match="Unknown solver setting"):
        _ = solver.settings.not_a_real_setting
    with pytest.raises(AttributeError, match="Unknown solver setting"):
        solver.settings.also_not_a_setting = 1.0


def test_dir_lists_catalog(solver):
    keys = dir(solver.settings)
    catalog_keys = [o.key for o in SCFSolver.options()]
    assert sorted(catalog_keys) == keys


def test_settings_class_direct_use(solver):
    # Callers who want a wrapper without going through .settings
    # can construct one directly.
    s = Settings(solver)
    s.convergence_threshold = 5e-8
    assert s.convergence_threshold == pytest.approx(5e-8)
