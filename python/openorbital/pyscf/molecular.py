"""Molecular SCF driver wrapping PySCF mean-fields.

Provides ``OpenOrbitalSCF``, a thin adapter that uses
``openorbital.SCFSolver`` for the convergence loop and a PySCF
mean-field instance for the Fock build. Supports closed-shell (RHF,
RKS) and unrestricted (UHF, UKS) ground states with and without
point-group symmetry.

When PySCF's ``mol.symmetry`` is enabled, every irreducible
representation becomes its own OpenOrbitalOptimizer block, so the
SCF respects symmetry by construction: orbitals never mix across
irreps, and the symmetry-mandated degeneracies inside multi-
dimensional irreps (E, T, ...) come out automatically as degenerate
eigenvalues of the per-irrep Fock matrix. Without symmetry, the
driver uses a single block per spin channel.
"""

from __future__ import annotations

import numpy as np

from openorbital import SCFSolver


def _canonical_orthogonalizer(overlap: np.ndarray,
                              linear_dependency_threshold: float) -> np.ndarray:
    """Return X with X.T @ overlap @ X = I, dropping linear dependencies."""
    eigvals, eigvecs = np.linalg.eigh(overlap)
    keep = eigvals > linear_dependency_threshold
    return eigvecs[:, keep] * np.power(eigvals[keep], -0.5)


def _irrep_decomposition(mol):
    """List of (irrep_name, U) where U is the AO -> SALC basis transformation
    for an irreducible representation. Returns a single ("A", identity) entry
    when ``mol.symmetry`` is off or empty so the rest of the driver can be
    spelled in terms of a uniform irrep loop."""
    if getattr(mol, 'symmetry', False) and getattr(mol, 'symm_orb', None):
        return list(zip(mol.irrep_name, mol.symm_orb))
    nao = mol.nao_nr()
    return [("A", np.eye(nao))]


class OpenOrbitalSCF:
    """OpenOrbitalOptimizer-backed SCF driver wrapping a PySCF mean-field.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        A PySCF mean-field instance (RHF, RKS, UHF, UKS, ...). The
        wrapper does not call ``mf.kernel``; the SCF loop is driven by
        OpenOrbitalOptimizer with this object only used as a Fock
        builder. If ``mf.mol.symmetry`` is true, each irrep becomes a
        separate OpenOrbitalOptimizer block.
    linear_dependency_threshold : float, optional
        AO overlap eigenvalues below this threshold are dropped from
        the orthonormal basis (per irrep). Default ``1e-8``.

    Attributes
    ----------
    solver : openorbital.SCFSolver
        Underlying SCF solver. Configuration knobs
        (``convergence_threshold``, ``maximum_iterations``,
        ``optimal_damping_degeneracy_threshold``, ``verbosity``)
        go through this object.
    e_tot : float or None
        Total energy after ``kernel`` has been called.
    """

    def __init__(self,
                 mf,
                 linear_dependency_threshold: float = 1e-8):
        # Local PySCF import so the rest of openorbital is usable without PySCF.
        from pyscf.scf import uhf

        self.mf = mf
        self.mol = mf.mol
        self._is_uhf = isinstance(mf, uhf.UHF)

        # Per-irrep orthogonalizer T_i: a (nao, n_orth_i) matrix that maps
        # an orthonormal-irrep coefficient vector back to the AO basis.
        irreps = _irrep_decomposition(self.mol)
        overlap = mf.get_ovlp()
        self._irrep_names = [name for name, _ in irreps]
        self._T = []
        for _, U in irreps:
            S_irrep = U.T @ overlap @ U
            X = _canonical_orthogonalizer(S_irrep, linear_dependency_threshold)
            self._T.append(U @ X)
        self._n_irreps = len(self._T)
        self._nao = self.mol.nao_nr()

        # Set up SCFSolver: one block per (irrep, spin).
        if self._is_uhf:
            n_alpha, n_beta = self.mol.nelec
            self.solver = SCFSolver(
                number_of_blocks_per_particle_type=np.array(
                    [self._n_irreps, self._n_irreps], dtype=np.uintp),
                maximum_occupation=np.ones(2 * self._n_irreps),
                number_of_particles=np.array(
                    [float(n_alpha), float(n_beta)]),
                fock_builder=self._fock_builder_unrestricted,
                block_descriptions=[name + " alpha" for name in self._irrep_names]
                                 + [name + " beta" for name in self._irrep_names],
            )
        else:
            n_elec = self.mol.nelectron
            self.solver = SCFSolver(
                number_of_blocks_per_particle_type=np.array(
                    [self._n_irreps], dtype=np.uintp),
                maximum_occupation=np.full(self._n_irreps, 2.0),
                number_of_particles=np.array([float(n_elec)]),
                fock_builder=self._fock_builder_restricted,
                block_descriptions=list(self._irrep_names),
            )

        self.e_tot = None

    # --- basis-change helpers ------------------------------------------------

    def _density_from_irrep_orbitals(self,
                                     orbitals_blocks,
                                     occupations_blocks) -> np.ndarray:
        """Sum per-irrep orth-basis (orbitals, occupations) into an AO density."""
        P_ao = np.zeros((self._nao, self._nao))
        for T, C, n in zip(self._T, orbitals_blocks, occupations_blocks):
            if C.size == 0:
                continue
            C_ao = T @ C
            P_ao += (C_ao * n) @ C_ao.T
        return P_ao

    def _fock_per_irrep(self, F_ao: np.ndarray):
        """Project an AO Fock matrix into the orthonormal basis of each irrep."""
        return [np.asfortranarray(T.T @ F_ao @ T) for T in self._T]

    # --- Fock-builder callbacks ----------------------------------------------

    def _fock_builder_restricted(self, density):
        orbitals_orth, occupations = density
        P_ao = self._density_from_irrep_orbitals(orbitals_orth, occupations)
        h1 = self.mf.get_hcore()
        veff = self.mf.get_veff(self.mol, dm=P_ao)
        E_elec, _ = self.mf.energy_elec(dm=P_ao, h1e=h1, vhf=veff)
        return (float(E_elec + self.mol.energy_nuc()),
                self._fock_per_irrep(h1 + veff))

    def _fock_builder_unrestricted(self, density):
        orbitals_orth, occupations = density
        ni = self._n_irreps
        Pa_ao = self._density_from_irrep_orbitals(orbitals_orth[:ni],
                                                  occupations[:ni])
        Pb_ao = self._density_from_irrep_orbitals(orbitals_orth[ni:],
                                                  occupations[ni:])
        dm = np.asarray((Pa_ao, Pb_ao))
        h1 = self.mf.get_hcore()
        veff = self.mf.get_veff(self.mol, dm=dm)
        E_elec, _ = self.mf.energy_elec(dm=dm, h1e=h1, vhf=veff)
        Fa_blocks = self._fock_per_irrep(h1 + veff[0])
        Fb_blocks = self._fock_per_irrep(h1 + veff[1])
        return (float(E_elec + self.mol.energy_nuc()),
                Fa_blocks + Fb_blocks)

    # --- public API ----------------------------------------------------------

    def initial_fock(self, dm_init=None):
        """Per-irrep Fock blocks built from ``dm_init`` (PySCF init guess by default)."""
        if dm_init is None:
            dm_init = self.mf.get_init_guess()
        h1 = self.mf.get_hcore()
        if self._is_uhf:
            if dm_init.ndim == 2:
                dm_init = np.asarray((dm_init * 0.5, dm_init * 0.5))
            veff = self.mf.get_veff(self.mol, dm=dm_init)
            Fa = self._fock_per_irrep(h1 + veff[0])
            Fb = self._fock_per_irrep(h1 + veff[1])
            return Fa + Fb
        veff = self.mf.get_veff(self.mol, dm=dm_init)
        return self._fock_per_irrep(h1 + veff)

    def kernel(self, dm_init=None):
        """Run the SCF and return the total energy."""
        self.solver.initialize_with_fock(self.initial_fock(dm_init))
        self.solver.run()
        self.e_tot = float(self.solver.get_energy(0))
        return self.e_tot

    # --- post-SCF accessors --------------------------------------------------

    def make_rdm1(self):
        """AO-basis density matrix(es) of the converged state."""
        orbitals = self.solver.get_orbitals(0)
        occupations = self.solver.get_orbital_occupations(0)
        if self._is_uhf:
            ni = self._n_irreps
            Pa = self._density_from_irrep_orbitals(orbitals[:ni],
                                                    occupations[:ni])
            Pb = self._density_from_irrep_orbitals(orbitals[ni:],
                                                    occupations[ni:])
            return np.asarray((Pa, Pb))
        return self._density_from_irrep_orbitals(orbitals, occupations)

    def mo_coeff_ao(self):
        """AO-basis MO coefficients, irrep-block list per spin channel."""
        orbitals = self.solver.get_orbitals(0)
        if self._is_uhf:
            ni = self._n_irreps
            Ca = [T @ C for T, C in zip(self._T, orbitals[:ni])]
            Cb = [T @ C for T, C in zip(self._T, orbitals[ni:])]
            return Ca, Cb
        return [T @ C for T, C in zip(self._T, orbitals)]

    def mo_occ(self):
        """Per-irrep occupation arrays, separated by spin for unrestricted."""
        occupations = self.solver.get_orbital_occupations(0)
        if self._is_uhf:
            ni = self._n_irreps
            return list(occupations[:ni]), list(occupations[ni:])
        return list(occupations)

    @property
    def irrep_names(self):
        """Irrep labels used for each block (one per irrep)."""
        return list(self._irrep_names)
