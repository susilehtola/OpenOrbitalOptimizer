"""Atomic SCF driver with enforced spherical symmetry.

For an atomic calculation OpenOrbitalOptimizer is fed one block per
``(l, spin)`` shell. Each radial function in the ``l`` shell is one
``orbital`` from the solver's point of view, and the per-block
``maximum_occupation`` carries the magnetic-component multiplicity:
``2(2l+1)`` for restricted (s=2, p=6, d=10, f=14), ``2l+1`` for
unrestricted (s=1, p=3, d=5, f=7).

The Fock matrix per block is built by averaging the AO Fock over the
``2l+1`` magnetic components of that shell, enforcing spherical
invariance every iteration. Symmetric fractional occupations in
partially filled degenerate shells (e.g. Fe d^6 -> 6/5 per m-component
in the alpha channel) emerge automatically from OpenOrbitalOptimizer's
skeleton enumeration.

This driver expects a single-atom PySCF molecule with spherical AOs
(``mol.cart == False``) and PySCF basis shells with one contracted
function each. The standard BSE/PySCF basis sets satisfy both
conditions.
"""

from __future__ import annotations

import numpy as np

from openorbital import SCFSolver

from .molecular import _canonical_orthogonalizer


_ANG_LABELS = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i", 7: "k"}


def _identify_atomic_l_shells(mol):
    """Group radial basis functions by angular momentum.

    Returns ``{l: list of ao_start_per_radial}``, where each entry is the
    starting AO index of one radial function in spherical-AO order. A
    PySCF shell with generalized contractions (n_ctr > 1) contributes
    n_ctr entries to the list. Spherical AOs are assumed (``mol.cart ==
    False``), with PySCF's ``(ctr, m)`` ordering (contraction outer,
    magnetic component inner).
    """
    if mol.natm != 1:
        raise ValueError(
            "atomic driver requires single-atom mol; got natm=%d" % mol.natm)
    if getattr(mol, "cart", False):
        raise ValueError(
            "atomic driver requires spherical AOs; rebuild mol with cart=False")

    ao_loc = mol.ao_loc_nr()
    shells_by_l = {}
    for shell in range(mol.nbas):
        l = mol.bas_angular(shell)
        n_ctr = mol.bas_nctr(shell)
        ao_start = ao_loc[shell]
        expected = n_ctr * (2 * l + 1)
        actual = ao_loc[shell + 1] - ao_start
        if actual != expected:
            raise ValueError(
                "Shell %d has angular momentum %d and %d contractions but "
                "%d AOs (expected %d); unsupported basis layout."
                % (shell, l, n_ctr, actual, expected))
        for ctr in range(n_ctr):
            shells_by_l.setdefault(l, []).append(
                ao_start + ctr * (2 * l + 1))
    return shells_by_l


class OpenOrbitalAtomicSCF:
    """Atomic SCF driver wrapping a PySCF mean-field with spherical symmetry.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Single-atom mean-field instance (RHF, RKS, UHF, UKS).
    linear_dependency_threshold : float, optional
        Per-``l`` overlap eigenvalues below this threshold are dropped
        from the orthonormal radial basis. Default ``1e-8``.

    Attributes
    ----------
    solver : openorbital.SCFSolver
    e_tot : float or None
    """

    def __init__(self,
                 mf,
                 linear_dependency_threshold: float = 1e-8):
        from pyscf.scf import uhf

        self.mf = mf
        self.mol = mf.mol
        self._is_uhf = isinstance(mf, uhf.UHF)
        self._nao = self.mol.nao_nr()

        self._shells_by_l = _identify_atomic_l_shells(self.mol)
        self._l_values = sorted(self._shells_by_l)
        self._n_blocks = len(self._l_values)

        # Per-l data: number of radial functions and (2l+1, n_radial) array
        # of AO indices (one row per magnetic component).
        self._n_radial = {}
        self._m_aos = {}
        for l in self._l_values:
            radial_starts = self._shells_by_l[l]
            n_radial = len(radial_starts)
            self._n_radial[l] = n_radial
            m_aos = np.zeros((2 * l + 1, n_radial), dtype=int)
            for r, ao_start in enumerate(radial_starts):
                for m in range(2 * l + 1):
                    m_aos[m, r] = ao_start + m
            self._m_aos[l] = m_aos

        # Per-l canonical orthogonalizer on the radial-only overlap.
        overlap = mf.get_ovlp()
        self._X = {}
        for l in self._l_values:
            S_l = self._radialize(overlap, l)
            self._X[l] = _canonical_orthogonalizer(
                S_l, linear_dependency_threshold)

        # Build solver. Block descriptions are spectroscopic labels.
        block_descriptions = [_ANG_LABELS.get(l, "l=%d" % l)
                              for l in self._l_values]
        if self._is_uhf:
            n_alpha, n_beta = self.mol.nelec
            max_occ = np.array(
                [2 * l + 1 for l in self._l_values] +
                [2 * l + 1 for l in self._l_values], dtype=float)
            self.solver = SCFSolver(
                number_of_blocks_per_particle_type=np.array(
                    [self._n_blocks, self._n_blocks], dtype=np.uintp),
                maximum_occupation=max_occ,
                number_of_particles=np.array(
                    [float(n_alpha), float(n_beta)]),
                fock_builder=self._fock_builder_unrestricted,
                block_descriptions=[d + " alpha" for d in block_descriptions]
                                 + [d + " beta" for d in block_descriptions],
            )
        else:
            n_elec = self.mol.nelectron
            max_occ = np.array(
                [2 * (2 * l + 1) for l in self._l_values], dtype=float)
            self.solver = SCFSolver(
                number_of_blocks_per_particle_type=np.array(
                    [self._n_blocks], dtype=np.uintp),
                maximum_occupation=max_occ,
                number_of_particles=np.array([float(n_elec)]),
                fock_builder=self._fock_builder_restricted,
                block_descriptions=block_descriptions,
            )

        self.e_tot = None

    # --- helpers --------------------------------------------------------------

    def _radialize(self, M_ao: np.ndarray, l: int) -> np.ndarray:
        """Average a (nao, nao) AO matrix over the (2l+1) m-components of l."""
        m_aos = self._m_aos[l]
        n_radial = self._n_radial[l]
        block = np.zeros((n_radial, n_radial))
        for m in range(2 * l + 1):
            ao_idx = m_aos[m]
            block += M_ao[np.ix_(ao_idx, ao_idx)]
        return block / (2 * l + 1)

    def _density_from_orth(self,
                           orbitals_orth_blocks,
                           occupations_blocks) -> np.ndarray:
        """AO density built from OOO's per-l orth orbitals.

        For each radial orbital with occupation ``n`` the per-m
        contribution is ``n / (2l+1)``; this is placed identically into
        every m-block of the AO matrix, which keeps the AO density
        block-diagonal in m and rotationally invariant.
        """
        P_ao = np.zeros((self._nao, self._nao))
        for l, C_orth, n in zip(self._l_values,
                                orbitals_orth_blocks,
                                occupations_blocks):
            if C_orth.size == 0:
                continue
            X = self._X[l]
            C_radial = X @ C_orth
            D_per_m = (C_radial * (n / (2 * l + 1))) @ C_radial.T
            for m in range(2 * l + 1):
                ao_idx = self._m_aos[l][m]
                P_ao[np.ix_(ao_idx, ao_idx)] += D_per_m
        return P_ao

    def _fock_per_block(self, F_ao: np.ndarray):
        out = []
        for l in self._l_values:
            F_l = self._radialize(F_ao, l)
            X = self._X[l]
            out.append(np.asfortranarray(X.T @ F_l @ X))
        return out

    # --- Fock-builder callbacks ----------------------------------------------

    def _fock_builder_restricted(self, density):
        orbitals_orth, occupations = density
        P_ao = self._density_from_orth(orbitals_orth, occupations)
        h1 = self.mf.get_hcore()
        veff = self.mf.get_veff(self.mol, dm=P_ao)
        E_elec, _ = self.mf.energy_elec(dm=P_ao, h1e=h1, vhf=veff)
        return (float(E_elec + self.mol.energy_nuc()),
                self._fock_per_block(h1 + veff))

    def _fock_builder_unrestricted(self, density):
        orbitals_orth, occupations = density
        ni = self._n_blocks
        Pa_ao = self._density_from_orth(orbitals_orth[:ni], occupations[:ni])
        Pb_ao = self._density_from_orth(orbitals_orth[ni:], occupations[ni:])
        dm = np.asarray((Pa_ao, Pb_ao))
        h1 = self.mf.get_hcore()
        veff = self.mf.get_veff(self.mol, dm=dm)
        E_elec, _ = self.mf.energy_elec(dm=dm, h1e=h1, vhf=veff)
        Fa = self._fock_per_block(h1 + veff[0])
        Fb = self._fock_per_block(h1 + veff[1])
        return (float(E_elec + self.mol.energy_nuc()), Fa + Fb)

    # --- public API ----------------------------------------------------------

    def initial_fock(self, dm_init=None):
        if dm_init is None:
            dm_init = self.mf.get_init_guess()
        h1 = self.mf.get_hcore()
        if self._is_uhf:
            if dm_init.ndim == 2:
                dm_init = np.asarray((dm_init * 0.5, dm_init * 0.5))
            veff = self.mf.get_veff(self.mol, dm=dm_init)
            Fa = self._fock_per_block(h1 + veff[0])
            Fb = self._fock_per_block(h1 + veff[1])
            return Fa + Fb
        veff = self.mf.get_veff(self.mol, dm=dm_init)
        return self._fock_per_block(h1 + veff)

    def kernel(self, dm_init=None):
        self.solver.initialize_with_fock(self.initial_fock(dm_init))
        self.solver.run()
        self.e_tot = float(self.solver.get_energy(0))
        return self.e_tot

    def make_rdm1(self):
        orbitals = self.solver.get_orbitals(0)
        occupations = self.solver.get_orbital_occupations(0)
        if self._is_uhf:
            ni = self._n_blocks
            Pa = self._density_from_orth(orbitals[:ni], occupations[:ni])
            Pb = self._density_from_orth(orbitals[ni:], occupations[ni:])
            return np.asarray((Pa, Pb))
        return self._density_from_orth(orbitals, occupations)

    @property
    def block_labels(self):
        return [_ANG_LABELS.get(l, "l=%d" % l) for l in self._l_values]
