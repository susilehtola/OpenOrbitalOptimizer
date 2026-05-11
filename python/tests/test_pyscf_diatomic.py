"""Diatomic SCF driver tests.

Compares the cylindrical-symmetry-respecting ``OpenOrbitalDiatomicSCF``
against PySCF's reference SCF on diatomics whose ground state has only
integer Aufbau-tied fillings per spin channel (so the two
implementations must agree). N2 and CO singlets exercise sigma+pi
collapsing in Dooh and Coov, respectively; O2 triplet exercises pi_g
with one alpha electron per component.
"""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf  # noqa: E402

from openorbital.pyscf.diatomic import OpenOrbitalDiatomicSCF


def _run_pair(mol, mf_cls, **solver_kwargs):
    mf_ref = mf_cls(mol)
    e_ref = mf_ref.kernel()
    mf_oo_base = mf_cls(mol)
    oo = OpenOrbitalDiatomicSCF(mf_oo_base)
    oo.solver.verbosity(0)
    for k, v in solver_kwargs.items():
        getattr(oo.solver, k)(v)
    e_oo = oo.kernel()
    return e_ref, e_oo, oo


def test_h2_dooh():
    """H2 singlet in Dooh: only sigma orbitals are occupied; cc-pVDZ
    polarization functions produce empty pi blocks that the driver
    correctly enumerates without affecting the converged energy."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        symmetry="Dooh",
        verbose=0,
    )
    e_ref, e_oo, oo = _run_pair(
        mol, scf.RHF,
        convergence_threshold=1e-9, maximum_iterations=200,
    )
    assert abs(e_ref - e_oo) < 1e-6, "H2: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)
    assert "A1g" in oo.block_labels and "A1u" in oo.block_labels


def test_n2_singlet_dooh():
    """N2 singlet in Dooh: full pi_u (4 electrons), empty pi_g."""
    mol = gto.M(
        atom="N 0 0 0; N 0 0 1.10",
        basis="cc-pvdz",
        symmetry="Dooh",
        verbose=0,
    )
    e_ref, e_oo, oo = _run_pair(
        mol, scf.RHF,
        convergence_threshold=1e-9, maximum_iterations=200,
    )
    assert abs(e_ref - e_oo) < 1e-5, "N2: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)
    assert "pi_u" in oo.block_labels


def test_o2_triplet_dooh():
    """O2 triplet: pi_g half-filled, both electrons same spin.

    With ms=2 the alpha channel carries both pi_g components singly
    occupied, the beta channel has empty pi_g; both spin channels have
    only integer Aufbau-tied fillings, so the driver and unconstrained
    PySCF UHF must agree.
    """
    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.21",
        basis="cc-pvdz",
        spin=2,
        symmetry="Dooh",
        verbose=0,
    )
    e_ref, e_oo, oo = _run_pair(
        mol, scf.UHF,
        convergence_threshold=1e-9, maximum_iterations=200,
    )
    assert abs(e_ref - e_oo) < 1e-5, "O2 triplet: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)
    assert "pi_g" in oo.block_labels


def test_co_singlet_coov():
    """CO singlet in Coov: heteronuclear diatomic with pi pair in C2v."""
    mol = gto.M(
        atom="C 0 0 0; O 0 0 1.13",
        basis="cc-pvdz",
        symmetry="Coov",
        verbose=0,
    )
    e_ref, e_oo, oo = _run_pair(
        mol, scf.RHF,
        convergence_threshold=1e-9, maximum_iterations=200,
    )
    assert abs(e_ref - e_oo) < 1e-5, "CO: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)
    assert "pi" in oo.block_labels


if __name__ == "__main__":
    test_h2_dooh()
    print("H2 RHF / cc-pvdz Dooh         OK")
    test_n2_singlet_dooh()
    print("N2 RHF / cc-pvdz Dooh         OK")
    test_o2_triplet_dooh()
    print("O2 UHF triplet / cc-pvdz Dooh OK")
    test_co_singlet_coov()
    print("CO RHF / cc-pvdz Coov         OK")
