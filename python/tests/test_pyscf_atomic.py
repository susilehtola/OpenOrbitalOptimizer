"""Atomic SCF driver tests.

Compares the spherically symmetric ``OpenOrbitalAtomicSCF`` against
PySCF's reference SCF on cases where the converged ground state is
itself spherically symmetric, so the two implementations must agree.
"""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from pyscf import gto, scf  # noqa: E402

from openorbital.pyscf.atomic import OpenOrbitalAtomicSCF


def _run_pair(mol, mf_cls, **solver_kwargs):
    mf_ref = mf_cls(mol)
    e_ref = mf_ref.kernel()
    mf_oo_base = mf_cls(mol)
    oo = OpenOrbitalAtomicSCF(mf_oo_base)
    oo.solver.set("verbosity", 0)
    for k, v in solver_kwargs.items():
        oo.solver.set(k, v)
    e_oo = oo.kernel()
    return e_ref, e_oo, oo


def test_neon_rhf_cc_pvdz():
    """Closed-shell Ne in cc-pVDZ: 3 s-radials, 2 p-radials, 1 d-polar
    -- non-trivial virtual space."""
    mol = gto.M(atom="Ne 0 0 0", basis="cc-pvdz", verbose=0)
    e_ref, e_oo, oo = _run_pair(
        mol, scf.RHF,
        convergence_threshold=1e-9, maximum_iterations=200,
    )
    assert abs(e_ref - e_oo) < 1e-6, "Ne RHF: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)
    assert oo.block_labels == ["s", "p", "d"]


def test_argon_uhf_cc_pvdz():
    """Closed-shell Ar with UHF in cc-pVDZ."""
    mol = gto.M(atom="Ar 0 0 0", basis="cc-pvdz", spin=0, verbose=0)
    e_ref, e_oo, oo = _run_pair(
        mol, scf.UHF,
        convergence_threshold=1e-9, maximum_iterations=200,
    )
    assert abs(e_ref - e_oo) < 1e-6, "Ar UHF: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)


def test_nitrogen_quartet_cc_pvdz():
    """Open-shell N (2p^3, all-same-spin alpha) in cc-pVDZ.

    The alpha p shell is exactly half-filled with all three magnetic
    components carrying one electron each; the beta p shell is empty;
    both spin channels have only integer Aufbau-tied fillings, so the
    spherically symmetric driver and unconstrained PySCF UHF must
    converge to the same energy.
    """
    mol = gto.M(atom="N 0 0 0", basis="cc-pvdz", spin=3, verbose=0)
    e_ref, e_oo, oo = _run_pair(
        mol, scf.UHF,
        convergence_threshold=1e-9, maximum_iterations=300,
    )
    assert abs(e_ref - e_oo) < 1e-5, "N quartet: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)


def test_phosphorus_quartet_cc_pvdz():
    """P 3p^3 in cc-pVDZ (analogue of N quartet, with 3s/3p valence)."""
    mol = gto.M(atom="P 0 0 0", basis="cc-pvdz", spin=3, verbose=0)
    e_ref, e_oo, oo = _run_pair(
        mol, scf.UHF,
        convergence_threshold=1e-9, maximum_iterations=300,
    )
    assert abs(e_ref - e_oo) < 1e-5, "P quartet: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)


def test_chromium_septet_cc_pvdz():
    """Cr (3d^5 4s^1) in cc-pVDZ: 6 valence electrons same-spin alpha.

    Both spin channels still have only integer Aufbau-tied fillings
    (alpha: full 1s..3p, 4s, and the 5-fold-degenerate 3d; beta: full
    1s..3p only), so the spherical driver must agree with unconstrained
    UHF.
    """
    mol = gto.M(atom="Cr 0 0 0", basis="cc-pvdz", spin=6, verbose=0)
    e_ref, e_oo, oo = _run_pair(
        mol, scf.UHF,
        convergence_threshold=1e-9, maximum_iterations=300,
    )
    assert abs(e_ref - e_oo) < 1e-5, "Cr septet: PySCF=%.10f OO=%.10f" % (e_ref, e_oo)


if __name__ == "__main__":
    test_neon_rhf_cc_pvdz()
    print("Ne RHF / cc-pvdz          OK")
    test_argon_uhf_cc_pvdz()
    print("Ar UHF / cc-pvdz          OK")
    test_nitrogen_quartet_cc_pvdz()
    print("N quartet UHF / cc-pvdz   OK")
    test_phosphorus_quartet_cc_pvdz()
    print("P quartet UHF / cc-pvdz   OK")
    test_chromium_septet_cc_pvdz()
    print("Cr septet UHF / cc-pvdz   OK")
