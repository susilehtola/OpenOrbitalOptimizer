"""Diatomic SCF driver with cylindrical symmetry.

For diatomic molecules orbitals come in groups by ``|m_l|``: ``sigma``
(``|m|=0``), ``pi`` (``|m|=1``), ``delta`` (``|m|=2``), etc.; ``pi`` and
``delta`` are two-fold degenerate, so a single radial orbital can hold
``4`` (restricted) or ``2`` (unrestricted) electrons per radial in
those shells. OpenOrbitalOptimizer should therefore see one block per
``(|m|, parity, spin)`` with the ``max_occ`` carrying the
two-fold-degeneracy of ``|m| > 0`` irreps; symmetric occupation
splitting across the two components in a partially filled shell falls
out of the polytope automatically.

PySCF's ``Dooh`` and ``Coov`` symmetry labels the two components of an
``|m|>0`` irrep with explicit ``x``/``y`` suffixes (``E1gx``/``E1gy``
for ``pi_g``, ``E2gx``/``E2gy`` for ``delta_g``, ``E3ux``/``E3uy`` for
``phi_u``, ...). The driver pairs the suffix-twins of every such irrep
into a single OpenOrbitalOptimizer block with ``max_occ`` doubled.
"""

from __future__ import annotations

import numpy as np

from openorbital import SCFSolver

from .molecular import _canonical_orthogonalizer


# Logical block names for the first few ``|m|>0`` irreps. Anything
# beyond gamma is reported with its raw ``E_n`` base label.
_M_TO_LABEL = {1: "pi", 2: "delta", 3: "phi", 4: "gamma"}


def _e_irrep_base(name):
    """Strip a trailing ``x``/``y`` from a PySCF Dooh/Coov irrep name.

    Returns ``(base, suffix)`` if the name is an ``E_n`` component, or
    ``None`` if it is a one-dimensional ``A``/``B`` irrep.
    """
    if len(name) > 1 and name[-1] in ("x", "y") and name[0] == "E":
        return name[:-1], name[-1]
    return None


def _pair_irreps_for_diatomic(irrep_names, symm_orbs):
    """Pair the ``x``/``y`` components of each ``E_n`` irrep.

    Returns a list of ``(label, kind, members)`` records describing
    each OOO block. ``kind == 'sigma'`` -> ``members`` is a single SALC
    matrix; ``kind == 'degenerate'`` -> ``members`` is a list of two
    SALC matrices (the ``x`` and ``y`` components), whose radial Fock
    matrices are identical by symmetry and whose density is
    half-occupied each.
    """
    by_name = dict(zip(irrep_names, symm_orbs))

    used = set()
    blocks = []
    for name in irrep_names:
        if name in used:
            continue
        parsed = _e_irrep_base(name)
        if parsed is None:
            blocks.append((name, "sigma", by_name[name]))
            used.add(name)
            continue
        base, suffix = parsed
        partner = base + ("y" if suffix == "x" else "x")
        if partner in by_name and partner not in used:
            # Identify the |m| index from the base, e.g. "E1g" -> 1.
            m_digit = ""
            for ch in base[1:]:
                if ch.isdigit():
                    m_digit += ch
                else:
                    break
            try:
                m = int(m_digit)
            except ValueError:
                m = None
            parity_suffix = base[1 + len(m_digit):]  # 'g' / 'u' or ''
            label_root = _M_TO_LABEL.get(m, base)
            label = label_root + ("_" + parity_suffix if parity_suffix else "")
            ux, uy = (by_name[name], by_name[partner]) if suffix == "x" \
                     else (by_name[partner], by_name[name])
            blocks.append((label, "degenerate", [ux, uy]))
            used.update((name, partner))
        else:
            blocks.append((name, "sigma", by_name[name]))
            used.add(name)
    return blocks


class OpenOrbitalDiatomicSCF:
    """Diatomic SCF driver wrapping a PySCF mean-field with cylindrical symmetry.

    The wrapped PySCF mean-field must be built from a molecule whose
    ``symmetry`` is ``'Dooh'`` or ``'Coov'`` (PySCF handles these
    through their D2h/C2v subgroups). Each ``pi`` irrep -- whose two
    D2h/C2v components are mandatorily degenerate by symmetry -- is
    collapsed into a single OpenOrbitalOptimizer block whose
    ``max_occ`` is doubled so that the polytope sees one radial channel
    per ``(pi, parity, spin)``. ``sigma`` irreps remain one OOO block
    each. ``delta`` and higher ``|m|`` irreps are *not* paired in this
    draft and would land in whichever D2h irrep they happen to share
    with ``sigma`` or ``pi``; the driver therefore targets first- and
    second-row diatomics where ``delta``/``phi`` are unoccupied.
    """

    def __init__(self,
                 mf,
                 linear_dependency_threshold: float = 1e-8):
        from pyscf.scf import uhf

        self.mf = mf
        self.mol = mf.mol
        self._is_uhf = isinstance(mf, uhf.UHF)
        self._nao = self.mol.nao_nr()

        groupname = getattr(self.mol, "groupname", "")
        if groupname not in ("Dooh", "Coov"):
            raise ValueError(
                "diatomic driver expects mol.symmetry in {'Dooh','Coov'}, "
                "got groupname=%r" % groupname)

        # Pair the two-component E_n irreps; one-dimensional irreps
        # stay as-is.
        blocks = _pair_irreps_for_diatomic(
            list(self.mol.irrep_name), list(self.mol.symm_orb))
        self._block_records = blocks  # list of (label, kind, members)
        self._n_blocks = len(blocks)

        # Per-block canonical orthogonalizer.
        overlap = mf.get_ovlp()
        self._T = []  # for sigma: single (nao, n_orth) matrix
                      # for pi:    list of two (nao, n_orth) matrices (degenerate)
        for label, kind, members in blocks:
            if kind == "sigma":
                U = members
                S_irrep = U.T @ overlap @ U
                X = _canonical_orthogonalizer(
                    S_irrep, linear_dependency_threshold)
                self._T.append(U @ X)
            else:  # pi: two components with identical radial overlap
                Ux, Uy = members
                Sx = Ux.T @ overlap @ Ux
                X = _canonical_orthogonalizer(
                    Sx, linear_dependency_threshold)
                self._T.append((Ux @ X, Uy @ X))

        # Set up the solver. max_occ is 2 / 1 for sigma, 4 / 2 for pi.
        block_labels = [b[0] for b in blocks]
        if self._is_uhf:
            n_alpha, n_beta = self.mol.nelec
            max_occ_one = [2.0 if kind == "degenerate" else 1.0
                           for _, kind, _ in blocks]
            self.solver = SCFSolver(
                number_of_blocks_per_particle_type=np.array(
                    [self._n_blocks, self._n_blocks], dtype=np.uintp),
                maximum_occupation=np.array(max_occ_one + max_occ_one),
                number_of_particles=np.array(
                    [float(n_alpha), float(n_beta)]),
                fock_builder=self._fock_builder_unrestricted,
                block_descriptions=[lbl + " alpha" for lbl in block_labels]
                                 + [lbl + " beta" for lbl in block_labels],
            )
        else:
            n_elec = self.mol.nelectron
            max_occ = [4.0 if kind == "degenerate" else 2.0
                       for _, kind, _ in blocks]
            self.solver = SCFSolver(
                number_of_blocks_per_particle_type=np.array(
                    [self._n_blocks], dtype=np.uintp),
                maximum_occupation=np.array(max_occ),
                number_of_particles=np.array([float(n_elec)]),
                fock_builder=self._fock_builder_restricted,
                block_descriptions=list(block_labels),
            )

        if self._is_uhf:
            self.solver.set_batched_fock_builder(
                self._fock_builder_batched_unrestricted)
        else:
            self.solver.set_batched_fock_builder(
                self._fock_builder_batched_restricted)

        self.e_tot = None

    # --- helpers --------------------------------------------------------------

    def _density_from_orth(self, orbitals_blocks, occupations_blocks):
        P_ao = np.zeros((self._nao, self._nao))
        for record, T_entry, C, n in zip(
                self._block_records, self._T, orbitals_blocks, occupations_blocks):
            if C.size == 0:
                continue
            label, kind, _ = record
            if kind == "sigma":
                C_ao = T_entry @ C
                P_ao += (C_ao * n) @ C_ao.T
            else:
                # Pi: two degenerate components share the same radial
                # orbital; each carries n/2 of the occupation.
                Tx, Ty = T_entry
                Cx_ao = Tx @ C
                Cy_ao = Ty @ C
                P_ao += (Cx_ao * (n / 2.0)) @ Cx_ao.T
                P_ao += (Cy_ao * (n / 2.0)) @ Cy_ao.T
        return P_ao

    def _fock_per_block(self, F_ao):
        out = []
        for record, T_entry in zip(self._block_records, self._T):
            label, kind, _ = record
            if kind == "sigma":
                F = T_entry.T @ F_ao @ T_entry
            else:
                Tx, Ty = T_entry
                # Components are degenerate; average the two for cleanliness.
                F = 0.5 * (Tx.T @ F_ao @ Tx + Ty.T @ F_ao @ Ty)
            out.append(np.asfortranarray(F))
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

    def _fock_builder_batched_restricted(self, densities):
        if len(densities) == 1:
            return [self._fock_builder_restricted(densities[0])]
        P_ao_stack = np.stack(
            [self._density_from_orth(orb, occ) for orb, occ in densities],
            axis=0,
        )
        h1 = self.mf.get_hcore()
        veff_stack = self.mf.get_veff(self.mol, dm=P_ao_stack)
        results = []
        for P_ao, veff in zip(P_ao_stack, veff_stack):
            E_elec, _ = self.mf.energy_elec(dm=P_ao, h1e=h1, vhf=veff)
            results.append((
                float(E_elec + self.mol.energy_nuc()),
                self._fock_per_block(h1 + veff),
            ))
        return results

    def _fock_builder_batched_unrestricted(self, densities):
        # PySCF's UKS / UHF do not accept (N, 2, nao, nao) batched input;
        # loop the single-density callback instead. See the molecular
        # driver's note for details.
        return [self._fock_builder_unrestricted(d) for d in densities]

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

    def kernel(self, methods="DIIS + ODA + CG", *, dm_init=None):
        """Run the SCF and return the total energy.

        ``methods`` is forwarded verbatim to ``SCFSolver.run`` as the
        ``+``-separated token list (case-insensitive; choices are
        ``DIIS``, ``ODA``, ``CG``, ``LBFGS``).
        ``dm_init`` is keyword-only so a stray positional
        ``oo.kernel("...")`` cannot silently bind a string to it and
        crash inside PySCF's Fock builder.
        """
        self.solver.initialize_with_fock(self.initial_fock(dm_init))
        self.solver.run(methods=methods)
        self.e_tot = float(self.solver.get_energy(0))
        self._populate_mf_results()
        return self.e_tot

    def _populate_mf_results(self):
        """Write the converged OOO state onto ``self.mf``. Sigma blocks
        carry one PySCF orbital per OOO orbital; pi/delta/... blocks
        carry two (the x and y E_n components), each with half the OOO
        occupation. Orbital energies come from the diagonal of C^T F C
        in the orthonormal-basis pseudo orbitals; the AO-basis
        coefficients are globally sorted by energy per spin channel."""
        orbitals = self.solver.get_orbitals(0)
        occupations = self.solver.get_orbital_occupations(0)
        fock = self.solver.get_fock_matrix(0)

        def _flatten_spin(orb_blocks, occ_blocks, fock_blocks):
            Cs, eps_list, ns = [], [], []
            for record, T_entry, C, n, F in zip(
                    self._block_records, self._T,
                    orb_blocks, occ_blocks, fock_blocks):
                if C.size == 0:
                    continue
                _, kind, _ = record
                F_mo = C.T @ F @ C
                eps = np.real(np.diag(F_mo))
                n_arr = np.asarray(n, dtype=float)
                if kind == "sigma":
                    Cs.append(T_entry @ C)
                    eps_list.append(eps)
                    ns.append(n_arr)
                else:
                    Tx, Ty = T_entry
                    Cs.append(Tx @ C)
                    Cs.append(Ty @ C)
                    eps_list.append(eps)
                    eps_list.append(eps)
                    half_n = n_arr / 2.0
                    ns.append(half_n)
                    ns.append(half_n)
            if not Cs:
                return (np.zeros((self._nao, 0)),
                        np.zeros(0), np.zeros(0))
            C_full = np.hstack(Cs)
            e_full = np.concatenate(eps_list)
            n_full = np.concatenate(ns)
            order = np.argsort(e_full, kind="stable")
            return C_full[:, order], e_full[order], n_full[order]

        if self._is_uhf:
            ni = self._n_blocks
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
        return [label for label, _, _ in self._block_records]
