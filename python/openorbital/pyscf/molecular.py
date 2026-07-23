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

        # Register a batched Fock-builder callback so the ODA
        # axis-vertex sweep can amortise integral / grid setup via
        # PySCF's vectorised get_veff (which accepts (N, nao, nao) and
        # (N, 2, nao, nao) density stacks).
        if self._is_uhf:
            self.solver.set_batched_fock_builder(
                self._fock_builder_batched_unrestricted)
        else:
            self.solver.set_batched_fock_builder(
                self._fock_builder_batched_restricted)

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

    def _fock_builder_batched_restricted(self, densities):
        """Batched restricted Fock builder.

        Stacks the per-density AO matrices into one ``(N, nao, nao)``
        array, calls PySCF's vectorised ``get_veff`` once, then peels
        the results back into the per-density return tuples.

        N=1 hits a PySCF DFT bug ("non-broadcastable output operand")
        where rks.get_veff allocates Vxc unbatched and then fails to
        add the (1, nao, nao) Coulomb back in. Skip the batched path
        in that case -- there is nothing to amortise anyway.
        """
        if len(densities) == 1:
            return [self._fock_builder_restricted(densities[0])]
        P_ao_stack = np.stack(
            [self._density_from_irrep_orbitals(orb, occ)
             for orb, occ in densities],
            axis=0,
        )
        h1 = self.mf.get_hcore()
        veff_stack = self.mf.get_veff(self.mol, dm=P_ao_stack)
        results = []
        for P_ao, veff in zip(P_ao_stack, veff_stack):
            E_elec, _ = self.mf.energy_elec(dm=P_ao, h1e=h1, vhf=veff)
            results.append((
                float(E_elec + self.mol.energy_nuc()),
                self._fock_per_irrep(h1 + veff),
            ))
        return results

    def _fock_builder_batched_unrestricted(self, densities):
        """Unrestricted batched Fock builder.

        PySCF's UKS / UHF do not accept the ``(N, 2, nao, nao)`` batched
        input shape -- ``dft.uks.get_veff`` internally does
        ``dma, dmb = dms`` and fails with "too many values to unpack
        (expected 2)" whenever N != 2 and misinterprets the spin axis
        when N == 2. So we cannot batch on the PySCF side; loop the
        single-density callback instead. The C++ ODA polytope step
        still benefits from the batched Fock-builder hook (the call
        is made once per ODA call rather than N times), but the
        amortisation must happen at the level of integral / grid
        setup inside PySCF if at all.
        """
        return [self._fock_builder_unrestricted(d) for d in densities]

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

    def kernel(self, methods=None, *, dm_init=None, options=None):
        """Run the SCF and return the total energy.

        ``methods`` is stored as the solver's ``methods`` setting
        before running: a ``+``-separated token list
        (case-insensitive; choices are ``DIIS``, ``ODA``, ``CG``,
        ``LBFGS``). If ``None`` (default), whatever was previously
        set stays in force.

        ``options`` is a dict of solver settings applied via
        ``SCFSolver.set(key, value)`` before the run. Keys must be
        names listed by ``SCFSolver.options()``. If both ``options``
        and ``methods`` mention ``methods``, the ``methods`` kwarg
        wins.

        ``dm_init`` is keyword-only so a positional first argument
        always selects the method string -- the much more common case
        -- and a stray ``oo.kernel("...")`` cannot silently bind a
        string to ``dm_init`` and crash inside PySCF's Fock builder.
        """
        if options:
            for key, value in options.items():
                self.solver.set(key, value)
        if methods is not None:
            self.solver.set("methods", methods)
        self.solver.initialize_with_fock(self.initial_fock(dm_init))
        self.solver.run()
        self.e_tot = float(self.solver.get_energy(0))
        self._populate_mf_results()
        return self.e_tot

    def _populate_mf_results(self):
        """Write the converged OOO state onto ``self.mf`` so PySCF
        post-SCF routines (``analyze``, ``dip_moment``, density-fitting
        post-HF, ...) see the same answer. AO-basis MO coefficients
        are concatenated across irreps and globally sorted by orbital
        energy; ``mo_energy`` is taken from the diagonal of C^T F C in
        each irrep (canonical-orbital energies on the orthonormal-
        basis pseudo orbitals). ``mo_occ`` carries the OOO
        occupations in PySCF's convention (max 2 per orbital
        restricted; max 1 per spin channel unrestricted)."""
        orbitals = self.solver.get_orbitals(0)
        occupations = self.solver.get_orbital_occupations(0)
        fock = self.solver.get_fock_matrix(0)

        def _flatten_spin(orb_blocks, occ_blocks, fock_blocks):
            Cs, eps_list, ns = [], [], []
            for T, C, n, F in zip(self._T, orb_blocks, occ_blocks, fock_blocks):
                if C.size == 0:
                    continue
                F_mo = C.T @ F @ C
                Cs.append(T @ C)
                eps_list.append(np.real(np.diag(F_mo)))
                ns.append(np.asarray(n, dtype=float))
            if not Cs:
                return (np.zeros((self._nao, 0)),
                        np.zeros(0), np.zeros(0))
            C_full = np.hstack(Cs)
            e_full = np.concatenate(eps_list)
            n_full = np.concatenate(ns)
            order = np.argsort(e_full, kind="stable")
            return C_full[:, order], e_full[order], n_full[order]

        if self._is_uhf:
            ni = self._n_irreps
            Ca, ea, na = _flatten_spin(orbitals[:ni], occupations[:ni], fock[:ni])
            Cb, eb, nb = _flatten_spin(orbitals[ni:], occupations[ni:], fock[ni:])
            self.mf.mo_coeff = np.asarray((Ca, Cb))
            self.mf.mo_energy = np.asarray((ea, eb))
            self.mf.mo_occ = np.asarray((na, nb))
        else:
            C, e, n = _flatten_spin(orbitals, occupations, fock)
            self.mf.mo_coeff = C
            self.mf.mo_energy = e
            self.mf.mo_occ = n
        self.mf.e_tot = self.e_tot
        self.mf.converged = bool(self.solver.converged())

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
