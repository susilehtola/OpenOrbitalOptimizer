"""End-to-end test of the PySCF molecular driver.

For each test case we run PySCF's reference SCF, then run the
``OpenOrbitalSCF`` wrapper with the same mean-field object configured
identically, and require the two total energies to agree to micro-
Hartree. The cases cover:

    1. closed-shell RHF on water without symmetry,
    2. closed-shell RHF on water with C2v symmetry (each irrep is its
       own OpenOrbitalOptimizer block),
    3. unrestricted UHF on a triplet methylene without symmetry.
"""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf  # noqa: E402

from openorbital.pyscf import OpenOrbitalSCF


def _agrees_with_pyscf(mf, oo_driver, atol=1e-6):
    pyscf_energy = mf.kernel()
    oo_energy = oo_driver.kernel()
    return abs(pyscf_energy - oo_energy) < atol, pyscf_energy, oo_energy


def test_rhf_water_no_symmetry():
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='sto-3g',
        symmetry=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    oo = OpenOrbitalSCF(mf)
    oo.solver.set("verbosity", 0)
    oo.solver.set("convergence_threshold", 1e-8)
    ok, e_ref, e_oo = _agrees_with_pyscf(mf, oo)
    assert ok, f"PySCF = {e_ref:.10f}, OO = {e_oo:.10f}"
    # Single block (no symmetry).
    assert oo.irrep_names == ["A"]


def test_rhf_water_c2v():
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='sto-3g',
        symmetry='C2v',
        verbose=0,
    )
    mf = scf.RHF(mol)
    oo = OpenOrbitalSCF(mf)
    oo.solver.set("verbosity", 0)
    oo.solver.set("convergence_threshold", 1e-8)
    ok, e_ref, e_oo = _agrees_with_pyscf(mf, oo)
    assert ok, f"PySCF = {e_ref:.10f}, OO = {e_oo:.10f}"
    # C2v has four irreps; non-empty ones become OO blocks.
    assert len(oo.irrep_names) >= 2
    assert all(name in {"A1", "A2", "B1", "B2"} for name in oo.irrep_names)


def test_uhf_triplet_methylene():
    mol = gto.M(
        atom='C 0 0 0; H 0 0.99 0.59; H 0 -0.99 0.59',
        basis='sto-3g',
        spin=2,
        symmetry=False,
        verbose=0,
    )
    mf = scf.UHF(mol)
    oo = OpenOrbitalSCF(mf)
    oo.solver.set("verbosity", 0)
    oo.solver.set("convergence_threshold", 1e-8)
    ok, e_ref, e_oo = _agrees_with_pyscf(mf, oo)
    assert ok, f"PySCF = {e_ref:.10f}, OO = {e_oo:.10f}"


if __name__ == "__main__":
    test_rhf_water_no_symmetry()
    print("RHF water (C1)        OK")
    test_rhf_water_c2v()
    print("RHF water (C2v)       OK")
    test_uhf_triplet_methylene()
    print("UHF CH2 triplet (C1)  OK")
