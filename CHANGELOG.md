# Changelog

<!--
## vX.Y.0 / YYYY-MM-DD (Unreleased)

#### Breaking Changes

#### New Features

#### Enhancements
* Fully numerical regression tests, run through HelFEM's finite-element
  drivers, behind ``OpenOrbitalOptimizer_BUILD_HELFEM_TESTS`` (default
  OFF). HelFEM's drivers already use this library for their orbital
  optimization, so there is nothing to interface -- what is new is
  building them against the working tree and asserting on the result.
    - A converged HelFEM atom carries no basis-set error, which
      separates the solver from the basis cleanly. On iron at M = 0 the
      4s/3d split comes out 1.09114/6.90886 against cc-pVDZ's
      1.09916/6.90084, so the basis defines that split to about 8e-3
      while the stationarity conditions settle it to 3e-8.
    - HelFEM is built from source against this tree, not against an
      installed copy. HelFEM pins the OpenOrbitalOptimizer it fetches to
      an explicit commit -- deliberately, so its own history stays
      bisectable -- and a build made the ordinary way therefore links
      that snapshot. Ask such a build for the ARH step and it answers
      "Unknown method 'arh'", which is what a test suite run against it
      would silently be reporting on.
    - Built as an ExternalProject rather than a subdirectory, so
      HelFEM's own fifty-odd tests and its targets stay out of this
      project's suite; only the driver binary crosses.
    - Off by default because it is expensive: HelFEM wants BLAS/LAPACK,
      HDF5, libxc and OpenMP beyond what the rest of the suite needs,
      and a clean clone plus build plus run is some 23 minutes against
      64 seconds for everything else.
* The build tree is consumable in place. ``install(EXPORT)`` writes the
  targets file only on install, so ``-DOpenOrbitalOptimizer_DIR=<builddir>``
  found the generated Config and then failed on the missing Targets
  include; an ``export(EXPORT)`` alongside it fixes that. For a
  header-only library the build-tree interface points straight at the
  sources, which is what a consumer asking for a build tree wants.
* New accessor ``diis_error_norm(ihist=0)``: the norm of the error
  vector that ``converged()`` compares against the threshold. It pairs
  with ``diis_error_vector``, throws on an index past the end as the
  other entry accessors do, and the seven places that had been taking
  that norm by hand now go through it -- including the stall recovery,
  which had also open-coded the per-entry form to rank stored
  iterates.
* A run no longer announces convergence it cannot back. The Aufbau
  cleanup is adopted on the energy and does not consult the gradient,
  so it can settle the occupations at the cost of the criterion that
  was met before it ran -- and the convergence message was printed
  regardless, leaving the caller holding a state the solver's own
  ``converged()`` denied. A spin-polarised iron atom printed
  "Converged to energy -1263.4350147951!" and the driver then reported
  that the SCF had not converged.
    - The message now says which it is, and names the gradient and the
      threshold when the cleanup has spent it.
    - Refusing the swap instead is the worse trade, and was measured to
      be: it is what made the spin-restricted atom report occupations
      0.13 outside the Fermi-level window as converged.
* ``atomtest`` gained ``--max-gradient``, which asserts that bound
  instead of requiring the solver's own convergence verdict.
    - The iron test at M = 0 uses it. Settling the occupations perturbs
      the orbitals, and winning that back is worth about g^2/2H --
      around 1e-12 there, three orders under what the Fock builder
      reproduces -- so no line search can pay for it and where the
      gradient lands is decided by the arithmetic. It does not
      reproduce between machines: 5.0e-7 and 1.1e-6 on the two kinds of
      core on one workstation, 2.26e-6 on CI, which is how a threshold
      chosen from the first two came to fail on the third while the
      occupations settled to 5.7e-11.
* ``atomtest`` tabulates its two-electron radial integrals on first
  use instead of recomputing them inside every Fock build. They depend
  only on the basis exponents and quantum numbers, so they are the
  same on every build, and evaluating them in the innermost loop of
  the density contraction meant 7.7 million ``tgamma`` calls per Fock
  matrix on a lanthanum atom in AHGBS-7. A profile put 94 per cent of
  that calculation inside ``GTOBasis::coulomb`` with four fifths of it
  in the gamma function.
    - Tabulated, the contraction is a matrix-vector product. The La+
      cation goes from 1243 to 62 seconds, a factor of twenty, and the
      table costs 21 MB for that basis.
    - The integrals are unchanged -- the first Fock build reproduces
      its energy to the last digit -- but the summation order is not,
      so trajectories shift exactly as they do between different
      kinds of CPU core.
    - This is test-driver work; the library does not have the
      integrals.

#### Enhancements (continued)
* The Aufbau cleanup no longer runs when there is nothing for it to
  do, and finding that out is now cheap.
    - Asking about the shape of the occupation polytope cost a full
      ODA step -- reference vertex, trial loop, Fock builds -- whose
      energy verdict was then discarded, because enumeration and
      evaluation were the same code path.
      ``optimal_damping_step_`` takes an enumerate-only flag and
      returns once the skeletons are in hand, before any density is
      evaluated, so the question costs one diagonalisation.
    - A polytope with zero parameters means no degenerate cluster is
      fractionally filled: the occupations are already the only Aufbau
      filling available and the relaxation cannot move them. The
      cleanup returns early instead of relaxing anyway. Measured on
      integer-occupied atoms (Ne, Ar, Kr, O at M = 3), this takes the
      cleanup from dozens of evaluations to zero.
    - The cleanup runs on a converged exit and on a stalled one, and
      not on the ``maximum_iterations`` exit. The useful distinction
      is proximity to a fixed point, not convergence: a run cut off at
      the iteration cap is nowhere near one, and the relaxation the
      cleanup performs is only meaningful near one. ``--maxiter 1`` on
      iron costs one Fock build.
* The Aufbau cleanup is adopted on the energy alone. A rise smaller
  than the convergence threshold is not evidence of a preference for
  the mixed density, so it no longer blocks the swap, and the gradient
  is not consulted at all.
    - Making the swap conditional on keeping ``converged()`` meant
      refusing an occupation state five orders better because it left
      the gradient eight per cent over its threshold. On a
      spin-restricted iron atom the cleanup reached a Fermi-level
      residual of 4.3e-7 at a gradient of 1.08e-6, and what was
      reported instead was the mixed density it started from, whose
      residual is 0.13. That is not a safer answer, it is a wrong one
      in the other variable, reported as converged.
    - The cost is that a run can finish with the occupations right and
      the gradient a little short, and say so. Settling the
      occupations perturbs the orbitals, and winning that back is
      worth about g^2/2H -- around 1e-12 at the gradients in question,
      three orders under what a Fock builder in double reproduces --
      so no line search can pay for it. Both iron tests are therefore
      held to a gradient of 1e-5 and a Fermi-level residual of 1e-5,
      which is what the cases can actually reach. Pinning the bound to
      the two trajectories a single machine produces does not survive
      a third: 2e-6, read off a pair of local runs at M = 0, was
      exceeded at 2.26e-6 by CI.
    - Attempting the cleanup *earlier*, while the SCF still has
      descent to spend on the repair, was tried and does not work. On
      four cases -- iron and a La+ cation, each spin-restricted and
      spin-polarised -- no trigger gradient succeeds everywhere, and
      several converge to different and higher minima: La+ restricted
      lands 2.6e-4 Eh high at one setting and La+ polarised 3.8e-2 Eh
      high at another. The conditions are written in orbital energies,
      and imposing them before the orbitals are converged imposes them
      on numbers that do not yet mean anything.
* The relaxed polytope search no longer walks by default. New setting
  ``relaxed_occupation_refinements`` (default 0, was effectively 4)
  says how many times it re-anchors its quadratic model and steps on
  it.
    - Measured on iron at M = 0 and M = 5 and on a La+ cation with
      PBE, the walk changes neither the energy nor the Fermi-level
      residual in any digit, and costs one, one and sixty Fock
      evaluations respectively. What settles the occupations is the
      initial projected sample, which fixes which orbitals are
      fractional, and the KKT refinement, which fixes where inside
      them the occupations sit; the walk between them arrives where
      both ends already are.
    - The measurement was only possible once the relaxation estimate
      feeding its curvature was gated: before that the walk never took
      a step on any case, so there was nothing to compare against.
    - Kept rather than removed. All three cases have a polytope of one
      or two dimensions, and a system with several coupled fractional
      shells may still need it; the setting makes that a discoverable
      regression rather than a silent one.
* The perturbative estimate of the orbital relaxation declines to
  answer where it cannot. It is second order in the rotation, so it
  is worth something only while the step it predicts stays inside the
  radius a quadratic describes; new setting
  ``perturbative_relaxation_max_step`` (default 0.1) is the largest
  rotation it will extrapolate over, and past that it returns zero and
  leaves the caller the uncorrected energy.
    - The polytope search asked it for an estimate at skeleton
      *vertices*, which sit up to a hartree above the relaxed point and
      are where the orbitals are furthest from stationary in the whole
      calculation. What came back was not a rough number but a
      meaningless one, and it was then differenced to build the
      curvature the search steps on.
    - The effect is visible: the refinement walk in
      ``relaxed_occupation_search_`` had never taken a step on any case
      in the test suite -- its quadratic programme returned the anchor
      it was given, so the loop broke on its first pass every time.
      With the estimate gated it walks, once on a spin-restricted iron
      atom and through all four of its refinements on a La+ cation.
    - It buys no accuracy on either: the answers are unchanged to the
      last digit, the occupation refinement having reached them
      anyway. It costs iron one Fock evaluation and La+ sixty, the
      latter being four relaxations the walk now pays for. Whether
      that walk earns its keep is a separate question, and one that
      could not be asked while it was not running.
* A stalled SCF stands on the best iterate it computed rather than
  the one it happened to be holding. The history is ordered by energy
  and the iterate is its head, but convergence is judged on the
  gradient, and the two part company exactly where the run stops: the
  step stalls because the energy has gone flat to the arithmetic, and
  among iterates whose energies are then indistinguishable the
  gradients still differ.
    - A spin-restricted La+ cation with PBE in the AHGBS-7 basis
      stalled at a head gradient of 1.06e-6 against a threshold of
      1e-6 -- six per cent short -- while holding an entry at 9.6e-7.
      It reported that it had not converged, and the driver exited
      non-zero, over a fraction of a microhartree of energy separating
      two iterates it had already paid for.
    - No step is taken and nothing is recomputed: the choice is among
      entries in hand, and only once the walk has stopped anyway, so a
      run that still had somewhere to go is unaffected.
    - The cleanup that follows a stall is now entered whether or not
      that choice recovered the convergence, with its guard set to
      whatever is being claimed. Gating it on having failed meant that
      recovering convergence silently skipped the occupation cleanup,
      which left the same case reporting success with its occupations
      unsettled.
* The KKT occupation refinement solves the stationarity conditions
  rather than creeping toward them. On the iron atoms it now reaches
  a Fermi-level residual of 2e-11 to 2e-7 where it used to stop at
  1e-5 to 1e-6, lands on the reference energy of the M = 5 atom to
  1e-10 from either of this machine's two kinds of CPU core, and
  costs 13 to 29 per cent fewer Fock evaluations.
    - The step is expressed in the occupations with the particle
      number carried by an explicit multiplier, not as transfers
      against one chosen active orbital. The chemical potential then
      comes out of the solve instead of being averaged over the
      fractional orbitals, and taking an orbital out of the active set
      is the deletion of a row and a column.
    - Which is what makes the bound handling expressible: a step that
      drives an occupation to zero or to its maximum is taken as far
      as the bound, that orbital is pinned there, and the reduced
      problem is solved again. The previous code scaled the whole step
      back and stopped just short, so an orbital the model wanted
      empty stayed in the active set indefinitely.
    - The curvature is the derivative of the orbital energies,
      ``d eps_k / d n_l``, which by Janak's theorem is the same matrix
      as the second derivative of the energy but is read from a first
      difference rather than a second -- an error of order delta/h
      instead of delta/h^2, three orders better at the probe size used
      here.
    - And it is corrected between sweeps by a symmetric rank-one
      update on the secant pair. The probes measure the curvature at
      fixed orbitals, while every step is judged after the orbitals
      have relaxed, and relaxation screens it: on the M = 5 atom the
      fixed-orbital value is four times too stiff, so every Newton
      step came out four times too short and the residual contracted
      by a constant 0.758 a sweep -- linear convergence, which ran out
      of sweeps at 1e-5 instead of arriving. Both halves of the pair
      are already computed, so the correction is free, and it also
      retires the per-sweep probes.
    - The trust radius shrinks below the step it rejected rather than
      by a blind factor. The Newton step is usually shorter than the
      radius, so dividing the radius alone left the next sweep
      proposing and rejecting the identical step: one trace repeated a
      step five times while the radius fell from 5e-3 to 2e-5, some
      seventy-five Fock evaluations spent re-deriving a verdict
      already in hand.
    - A step is allowed to climb, but only by less than the scale the
      solver treats as a real energy difference, and the budget is
      spent against where the refinement began rather than per step.
      The residual is first order in the occupation step where the
      energy is second, so near the solution the energy comparison is
      reading its own noise while the residual still carries thousands
      of times its own: one trace refused a step offering 2.9e-6 ->
      3.4e-8, eighty-five fold, over an energy rise of 1e-10.
    - The accepted iterate is handed back on its own. The history is
      ordered by energy and the iterate is its head, so an accepted
      uphill move was filed behind the entry it climbed away from and
      that entry became the iterate again -- the refinement undone by
      its own bookkeeping, and with it the whole cleanup, which the
      caller then refused for costing the convergence criterion.
    - Settled occupations are an orbital perturbation, so the
      orbitals are relaxed at them before the result is handed back.
* ``atomtest`` can carry the whole SCF -- the Fock builder and the
  solver together -- at a precision chosen on the command line:
  ``--precision double``, ``longdouble`` or ``quad``. The radial
  integrals, the orthogonaliser, the core Hamiltonian, the Coulomb
  build and the quadrature are all templated on the scalar type.
  Libxc alone stays in double, being a smooth function of a density
  rounded to double and so contributing a systematic 1e-15 rather
  than noise.
    - It exists to separate an algorithm from its arithmetic. In
      double the Fock builder reproduces its own answer only to about
      1e-9 -- the two kinds of CPU core in a hybrid processor differ
      by that much -- while the occupation refinement has to tell
      iterates apart by 1e-10. Raising the precision says what the
      algorithm does when nothing is masked, and it said two useful
      things: the spin-restricted iron atom was indeed blinded, its
      residual improving by two orders once the noise was lifted,
      while the M = 5 atom was not, giving the same 1e-5 in double, in
      long double and in quadruple precision. That pointed at the
      curvature rather than the arithmetic.
    - The orthogonaliser has to be templated with the rest. Built from
      a double overlap matrix it is orthonormal only to double, and
      that error enters every later matrix element as though it were
      noise.
* The occupation curvature used by the KKT refinement is taken from
  the energy at fixed orbitals, not from the energy plus the
  perturbative estimate of the orbital relaxation. Adding the
  estimate inverted it. The estimate vanishes at the centre of the
  probe, where the orbitals are already relaxed, and is negative at
  both displaced points, so the second difference measured almost
  nothing but that artefact: on a spin-restricted iron atom the
  4s-3d transfer came out at -0.54 where the fixed-orbital value is
  +0.31, and the fully relaxed energies say the positive one is
  right.
    - The wrong sign cost speed, not accuracy. A negative eigenvalue
      is floored onto the positive cone before the Newton solve,
      which makes the step unboundedly long, and it was then clipped
      to the trust radius. Every sweep therefore proposed the longest
      step permitted, overshot, was rejected on a worsened residual
      and quartered the radius, so the refinement crept in over the
      noise. Iron at M = 0 settles in 302 Fock evaluations rather
      than 327 on a performance core, and 239 rather than 258 on an
      efficiency one.
* The Aufbau cleanup prints an iteration history of its own at
  verbosity 5, in the same shape as the one ``run()`` prints: the
  running Fock count, the energy and its change, the
  particle-number, Aufbau and Fermi-level errors, the occupations of
  every block, and a line naming the fractionally occupied orbitals
  with their occupations and orbital energies. On a spin-restricted
  iron atom that last line is the interesting one -- it shows the 4s
  and 3d occupations settling with their orbital energies equal to
  six digits, which is the stationarity condition the cleanup is
  there to reach, and it was previously invisible.
* The Fock-evaluation counter no longer restarts inside the Aufbau
  cleanup. The cleanup relocates the iterate -- it samples trial
  occupations, relaxes the orbitals at them and moves back -- and
  each relocation went through ``initialize_with_orbitals``, which
  zeroes the counter. The cleanup therefore reported a fraction of
  the work it had done: on a spin-restricted iron atom it printed 71
  Fock builds where it had spent about 210. Since that phase is the
  larger part of the run on such a case, the number is worth having
  right.
* The Aufbau cleanup's relaxed occupation Hessian is estimated from
  perturbation theory rather than built by relaxing the orbitals at
  each polytope vertex. New setting
  ``perturbative_occupation_hessian`` (default 1; 0 restores the
  relaxations).
    - ``relaxed_occupation_search_`` needs the curvature of the energy
      *after* the orbitals have re-relaxed, and obtained it by
      performing that relaxation at every vertex -- tens of Fock
      builds each. The second-order estimate costs one build per
      point: a quadratic model along the rotations is minimised at
      ``d = -(H + sigma)^-1 g``, lowering the energy by
      ``-1/2 g^T (H + sigma)^-1 g``. With the Roothaan-Hall diagonal
      for ``H`` that is the familiar expression -- semicanonicalise,
      then sum the squared off-diagonal Fock elements over their
      orbital-energy denominators; the ARH curvature model is used in
      its place where the history has something to say.
    - It goes through energies rather than gradients, by second
      differences over the polytope axes.
      ``relaxed_occupation_gradient_`` reads the orbital energies of
      whatever iterate the solver stands on, so at an unrelaxed point
      it returns the *bare* occupation Hessian rather than the relaxed
      one, and the two differ by exactly the orbital response the
      search exists to account for. The energy correction carries that
      response; the gradient at an unrelaxed point does not.
    - The level shift is kept in the denominator, damping the estimate
      toward zero on the near-degenerate pairs where second-order
      perturbation theory is least reliable. That is the conservative
      direction: understating the gain costs a relaxation that would
      have paid, overstating it sends the search to a point that does
      not exist.
    - Only the Hessian is estimated. Candidate steps are still
      accepted on a measured relaxed energy, so the rule that a step
      is adopted only if it demonstrably lowers the energy is
      untouched.
    - Measured on iron at M = 5 with ``"ODA + ARH"`` in the AHGBS-9
      basis, spin-restricted so that the 4s and 3d shells share
      electrons fractionally: the cleanup takes 95 Fock builds against
      176, to the same energy to eleven digits, with the refinement
      walk needing no relaxations at all against four. On cc-pVDZ,
      where the cleanup is trivial, it changes nothing -- oxygen at
      M = 1 and M = 3 and iron at M = 5 are identical with it on and
      off.

* The occupations are settled onto their stationarity conditions at
  the end of the Aufbau cleanup, by root-finding on the residual
  rather than by minimising the energy. New diagnostic
  ``fermi_level_error`` and three settings,
  ``kkt_occupation_refinement_steps`` (12),
  ``kkt_occupation_threshold`` (1e-7) and
  ``kkt_occupation_trust_radius`` (5e-3).
    - Minimising the energy over the occupations subject to a fixed
      particle number and ``0 <= n <= n_max`` gives, with a chemical
      potential ``mu`` and Janak's ``dE/dn_k = eps_k``,
      ``eps_k = mu`` where the occupation is fractional, ``eps_k >= mu``
      where it is zero and ``eps_k <= mu`` where it is maximal.
      ``aufbau_error`` tests the two inequalities and exempts the
      Fermi-level cluster, fractional occupation being legitimate
      exactly there -- so the equality, which is what decides *where*
      inside that cluster the fractions sit, went unmeasured.
      ``fermi_level_error`` is the largest violation of any of the
      three, in energy units.
    - Why an energy criterion could not find this: near a stationary
      point the energy is quadratic in the occupation displacement
      while these residuals are linear. The optimal-damping step
      stopped on a spin-restricted iron atom with a predicted gain of
      9.9e-8, just under the 1e-7 below which it declines to churn at
      the arithmetic floor, and the residual left behind was 1.2e-4.
    - Measured on iron at M = 5 in cc-pVDZ, the same command
      reproducibly gave two answers: 224 Fock builds ending at
      -1263.4350125736 and 300 ending at -1263.4350145596, 2e-6 Eh
      apart, both reporting convergence, differing only in the beta
      4s/3d split (0.6501 against 0.6547 electrons). Their Fermi-level
      residuals were 2.7e-4 and 1.2e-4 -- the higher-energy answer the
      further from stationarity. With the refinement both paths reach
      -1263.435014795, agreeing to eleven digits.
    - The step is chosen by the residual and only checked against the
      energy: a step that raises it is rejected and the previous state
      restored, so the rule that nothing is adopted unless it lowers
      the energy is untouched.
    - Orbitals pinned full or empty enter through an active set. One
      on the wrong side of the chemical potential has a descent
      direction into the fractional region and joins the set; one on
      its own side is at a legitimate bound and is left alone.
    - The curvature is a second difference of perturbatively relaxed
      energies, which agree to nine decimals, so it is too noisy to
      set a step length; a trust region bounds the step instead. It
      doubles on success and quarters on failure. Tying it to the
      length actually accepted, so that it contracts as the problem
      does, measured worse on both counts -- the residual ended 6 to
      50 times higher with *more* rejections -- because the rejections
      are not about the length. A shorter step in a direction taken
      from noisy curvature is still a poor step.
    - Costs nothing where there is nothing to settle: oxygen at M = 1
      and M = 3 and krypton are identical in energy and Fock count
      with the refinement on and off.
    - The refinement saves and restores the orbital history, the DIIS
      caches and the rotation-step state around itself, handing back
      only the improved iterate as one new entry. It moves the iterate
      with ``initialize_with_orbitals``, which restarts all of that,
      and a routine that does so from inside an SCF leaves the state
      machine running on histories that no longer describe where the
      iterate has been -- measured, that segfaulted on the next
      rotation step.
    - Running it after every optimal-damping step rather than once at
      the end, gated on the residual being in a range where the
      orbital energies mean something, is safe with that in place and
      costs about three and a half times more: on iron at M = 5 it
      reached the same energy with 272 energy evaluations against 75.
      The refinement is cheap once, where the orbitals have converged
      and a single relaxation settles the fractions; run while they
      are still moving it re-relaxes repeatedly and the next rotation
      step discards the result.
    - ``initialize_with_orbitals`` no longer zeroes
      ``number_of_fock_evaluations`` when a caller is relocating the
      iterate rather than starting a calculation. It did, so the
      counter restarted every time the refinement moved to a trial
      point, and the cost of the in-loop experiment above read as 18
      Fock builds against the true 272.

#### Bug Fixes
* ``cleanup()`` no longer reads past the end of an empty vector when the
  orbital history holds a single entry, which segfaulted every SCF that
  reached it. It assumed at least two entries: it sized its
  ``density_differences`` vector as ``orbital_history_.size()-1`` -- an
  unsigned expression that wraps on an empty history -- and then
  indexed both that vector and the sorted-index vector at 0. The
  ``idx(0)`` read is the one that crashes, and it is unconditional, so
  the failure does not depend on the verbosity.
    - Reachable only without DIIS in the method set, i.e. exactly
      ``"ODA + CG"`` and ``"ODA + LBFGS"``, which is why the defaults
      never showed it. The stall exit is not the trigger; it ``break``s
      before ``cleanup()`` runs. A history falls to one entry in
      ``cleanup()``'s own erase loop, and the crash comes on the next
      iteration if the step taken there declines to evaluate any
      density -- ``add_entry`` appends before judging the energy, so any
      step that evaluates leaves at least two.
    - Pruning a history of fewer than two entries is a no-op by
      definition, so the guard changes no converging calculation.
      Reported against HelFEM's ``gensap``, where a hydrogen atom with
      ``--lmax=0 --nelem=3 --method=HF --scfmethods="ODA + CG"`` died at
      iteration 2 and now converges to -0.4999999941.
* ``run()`` no longer announces a convergence the Aufbau cleanup has
  spent. The cleanup is adopted on the energy and says nothing about
  the commutator, so it can settle the occupations at the cost of the
  gradient criterion that was met before it ran; the closing message
  was printed regardless, and a spin-polarised iron atom would report
  ``Converged to energy -1263.4350147951!`` and then be told by the
  driver that the SCF had not converged. The message now reports which
  of the two happened, and names the gradient it settled at against
  the threshold it missed.
    - Restoring the veto on the swap is the worse trade and was
      measured as such -- it is what makes the spin-restricted atom
      report occupations 0.13 outside the Fermi-level window as
      converged. What the cleanup buys is worth keeping; the claim
      about the gradient is what gives.

#### Misc.
-->


## v0.5.0 / (Unreleased)

#### New Features
* Convergence is now judged in occupation space as well as in the
  orbital gradient. Two diagnostics measure it, both re-evaluated on
  every read like ``converged``:
    - ``particle_number_error`` -- the largest ``|sum(n) - N|`` over
      the particle types. This is an invariant rather than a
      convergence measure: every step mixes densities that already
      carry the right particle number, so losing any means something
      discarded it, and iterating will not bring it back. It is
      therefore reported but deliberately does not gate convergence,
      since a solver that cannot finish is worse than one that tells
      you its answer is off.
    - ``aufbau_error`` -- the largest occupation sitting above the
      Fermi level or missing from below it, measured against the
      Aufbau filling of the current orbital energies. Orbitals inside
      the Fermi-level degenerate cluster are exempt, that being where
      fractional occupation is legitimate. This one *is* a convergence
      measure and bounds convergence through the new
      ``aufbau_convergence_threshold`` setting (default 1e-6, negative
      to switch the check off).
    - The order matters and is why the bound is a separate predicate,
      ``occupations_converged()``, rather than part of ``converged()``:
      occupation space is put right by the Aufbau cleanup step, which
      needs the gradient to have converged first. A single predicate
      demanding both would block on an error that nothing had yet been
      allowed to fix -- measured on iron, the Aufbau error sits at
      5e-6 and never falls until the cleanup runs. ``run()`` therefore
      orders them gradient, then cleanup, then occupations, and says
      so at the volume of the iteration line if the occupations are
      what is holding the SCF open.
    - Costs nothing on a healthy SCF. On oxygen the Aufbau error is
      already exactly 0 several iterations before the DIIS error
      reaches its threshold, and iron converges in the same 431
      iterations to the same energy with the check on or off.
* New ``"LCIIS"`` method token implementing the least-squares
  commutator in the iterative subspace of Li & Yaron, J. Chem.
  Theory Comput. **12**, 5322 (2016),
  doi:[10.1021/acs.jctc.6b00666](https://doi.org/10.1021/acs.jctc.6b00666).
  Where Pulay's CDIIS minimises ``|| sum_i c_i [F_i, D_i] ||_F^2``,
  LCIIS minimises the commutator between the *predicted* Fock matrix
  and the *predicted* density,
  ``|| [sum_i c_i F_i, sum_j c_j D_j] ||_F^2``, which is genuinely
  quartic in ``c`` and therefore needs the full ``M x M`` grid of
  mixed commutators ``[F_i, D_j]`` rather than just the diagonal.
  The quartic is minimised under ``sum_i c_i = 1`` by Newton's
  method on the Lagrangian, seeded from the CDIIS coefficients.
    - LCIIS is a variant of the extrapolation step, not a step of
      its own: the token implies ``"DIIS"``, so it slots into the
      existing state machine and the A/EDIIS bracketing continues to
      apply. Use it as ``"LCIIS"``, ``"LCIIS + ODA + CG"``, and so on.
      Because the two share the one extrapolation step, asking for
      both (``"DIIS + LCIIS"``) throws rather than silently
      discarding the DIIS request.
    - Any failure of the quartic solve -- singular KKT matrix,
      non-finite iterate, no convergence within the iteration cap, a
      negative target function -- falls back to the CDIIS
      coefficients rather than throwing. LCIIS is a convergence
      accelerator; a bad quartic solve is a reason to take the
      ordinary DIIS step, not to abort the SCF.
    - Three new settings: ``lciis_maximum_history`` (default 6),
      ``lciis_maximum_iterations`` (50) and
      ``lciis_convergence_threshold`` (1e-10). The history cap is
      separate from ``maximum_history_length`` because LCIIS holds
      the whole ``M x M`` commutator grid at once, so its memory
      grows as ``M^2 * n_basis^2`` and its commutator build as
      ``M^2`` rather than DIIS's ``M``.
    - On the oxygen atom with PBE/cc-pVDZ, LCIIS reaches the same
      energies as DIIS in fewer iterations (7 vs 9 at M=1, 8 vs 9 at
      M=3).
* ``atomtest`` gained a ``--methods`` flag that overrides the
  driver's default method mix, so any token combination can be
  exercised from the command line.
* ``atomtest`` can now fail. It gained ``--reference-energy`` with
  ``--energy-tolerance``, ``--max-fermi-level-error``, and a check
  that the SCF converged at all; any of them failing exits non-zero.
  Until now the regression tests asserted only that the driver did
  not crash, so a run could converge to the wrong answer, or not
  converge, and still be recorded as a pass. The two oxygen tests
  now carry reference energies.
* Two transition-metal regression tests, iron at M = 5 and at M = 0,
  the latter spin-restricted so that the valence is shared
  fractionally between the 4s and 3d shells. They run without DIIS,
  the extrapolation otherwise converging these cases on its own and
  leaving the orbital-rotation step barely exercised, and take three
  to five minutes each.
    - They are only assertable now. Before the occupations were held
      to their stationarity conditions the same command reproducibly
      gave two answers 2e-6 Eh apart, both reporting convergence, so
      there was no stable energy to compare against.
    - Each bounds the occupation stationarity residual as well as the
      energy, which is the check that matters here: the two answers
      the M = 5 case used to give differ by 2e-6 Eh, so an energy
      tolerance loose enough to be robust would have accepted either.
      The residuals differed by a factor of two and say plainly which
      one is converged.
    - The bounds are what the cases reach rather than round numbers.
      M = 5 settles at a residual of 2e-7 to 8e-7 and is held to
      1e-5; M = 0 stalls at 3e-6, an occupation there sitting against
      a bound where the equality condition no longer applies, and is
      held to 1e-4.
* New ``"ARH"`` method token implementing the augmented Roothaan-Hall
  step of Høst, Olsen, Jansík, Thøgersen, Jørgensen and Helgaker,
  J. Chem. Phys. **128**, 124106 (2008),
  doi:[10.1063/1.2884588](https://doi.org/10.1063/1.2884588),
  generalised to several particle types following Feldmann, Baiardi
  and Reiher, J. Chem. Theory Comput. **19**, 856 (2023),
  doi:[10.1021/acs.jctc.2c01035](https://doi.org/10.1021/acs.jctc.2c01035).
  ARH is a third way of taking the orbital-rotation step, alongside
  ``"CG"`` and ``"LBFGS"``, and like them exactly one may be
  requested.
    - The Hessian model is exact on the directions the density
      history spans and equal to the Roothaan-Hall diagonal on the
      orthogonal complement. That split is the point: the RH diagonal
      is accurate *except* between near-degenerate orbitals, and the
      density-difference history spans primarily those directions, so
      the subspace made exact is the subspace where the diagonal is
      worst.
    - The curvature comes from the quasi-Newton condition
      ``H (D_i - D_0) = 2 (F_i - F_0)``, read off matrices the
      history already holds: no additional Fock builds, and a
      rejected trial step still leaves its pair behind as free
      curvature. Exact for Hartree-Fock, where ``F`` is linear in
      ``D``; approximate for Kohn-Sham, with the line search
      absorbing the difference.
    - The two pieces of the Hessian add rather than replace. The
      energy Hessian in the rotation parameters is
      ``tr(dD/dx_a . dF/dD . dD/dx_b) + tr(F . d2D/dx_a dx_b)``, and
      the quasi-Newton condition measures only the first: a Fock
      difference reports how the density-space gradient responded and
      says nothing about the curvature of the parametrisation itself.
      The second piece is the Roothaan-Hall diagonal, exactly -- along
      one rotation a one-electron energy is
      ``const + (n_j - n_i)(eps_i - eps_j) sin^2 x``.
    - Solved by the Woodbury identity: one diagonal division, two
      ``n_par x 2k`` products and a ``2k x 2k`` solve, with no Krylov
      loop and nothing of the size of a Fock build. With an empty
      subspace it reduces exactly to the existing preconditioned
      steepest-descent direction, so it is a refinement of that step
      rather than a replacement for it.
    - The subspace is joint over particle types and symmetry blocks
      rather than one per particle. The Fock matrix of one particle
      type responds linearly to the density of every other, so the
      curvature coupling them exists only in the joint space -- one
      subspace, one level shift, one line search.
    - Carries no state between calls, the curvature being rebuilt from
      the orbital history each step. A history entry another method
      contributed counts as much as one the rotation step made, and
      nothing needs invalidating when an ODA or extrapolation step
      relocates the iterate.
    - Entries taken at different occupations count too, which on the
      ODA path is most of them. The curvature pairs carry occupation
      coordinates alongside the rotation ones: writing
      ``D = C n C^dag``, an occupation change is diagonal in the
      reference MO basis and a rotation is off-diagonal, so the two
      read off disjoint parts of the very same matrices -- the
      displacement is the diagonal of the density difference and the
      paired gradient change the diagonal of the Fock difference, the
      derivative of the energy with respect to an occupation being
      that orbital's energy (Janak), which is what the ODA occupation
      gradient already uses. Nothing new is computed.
    - Those coordinates are what make a pair *consistent* when the
      occupations moved. A Fock difference responds to the whole move,
      so pairing it with a displacement that described only the
      rotation would charge occupation-driven curvature to rotation
      directions. The check on the relative normalisation is that the
      extended inner product reproduces the density-space one,
      ``s . y = tr(dD dF)``, with off-diagonal pairs counted twice and
      the diagonal once -- which is where the factor of 2 on the
      rotation coordinates comes from. The test verifies this to
      round-off.
    - The step itself is still taken in the rotation coordinates
      alone: the model is orthonormalised and the secant imposed in
      the extended space, then restricted to the rotation rows of
      ``Q`` and ``W``, leaving ``B`` -- built from the full inner
      products, and so carrying what the occupation coordinates
      contributed -- alone. Occupations remain ODA's to move.
    - What this recovers is large. On iron at M = 5 with
      ``"ODA + ARH"``, the calls that found no usable curvature at all
      fall from 29 to 7, and the typical subspace goes from two
      directions to the full nine the history holds.
    - The occupation and rotation coordinates are measured in one
      Euclidean metric when the directions are orthonormalised, so
      their relative scale decides which combinations of history
      entries form the basis the model is fitted in. This is not a
      free choice: sweeping a fixed weight over four decades moves
      oxygen's Fock count by up to a factor of two, and the best fixed
      value is system-dependent -- 0.1 is the best constant on oxygen
      and among the worst on iron (317 Fock builds against 241 at 1).
    - So the scale is fitted per call from the curvature the pairs
      themselves report. For a block of coordinates,
      ``c = sum_i s_i . y_i / sum_i |s_i|^2`` is its average curvature
      in energy per squared displacement, and scaling the occupation
      coordinates by ``sqrt(c_occupation / c_rotation)`` leaves both
      blocks at the same average curvature, so the singular values
      being compared measure the same thing. It is a change of
      coordinates rather than a reweighting -- displacement multiplied,
      paired gradient change divided -- so ``s . y = tr(dD dF)`` and
      the rotation-rotation block the step is taken in are both
      untouched. It falls back to leaving the coordinates alone when
      either block reports non-positive curvature.
    - Measured against the best fixed weight, on ``"ODA + ARH"``:
      iron 202 Fock builds against 241, chromium at M = 7 45 against
      48, krypton 30 against 33, oxygen 24 against 21 at M = 1 and 26
      against 23 at M = 3. It wins on the three hard cases, by 39
      builds on iron, and costs three on each oxygen. The setting
      ``arh_occupation_scale`` pins the weight to a positive value
      instead, which is what the sweep used.
    - The displacement and the gradient change are weighted by the
      occupation differences of the *entry*, not of the reference.
      Writing ``U = C_0^dag C_i = exp(kappa)``, the density difference
      in the reference MO basis is ``(U n_i U^dag)_ij ~ kappa_ij
      (n_i,j - n_i,i)`` off the diagonal, and in the reference chart
      the derivative at the displaced point is
      ``tr(F_i . C_0 [T_a, n_i] C_0^dag)``: both carry the entry's
      occupations. Using the reference's instead scales every rotation
      coordinate by the ratio of the two, an error that does not
      vanish as the step shrinks -- measured at 90%, flat in the step
      length, against an error linear in it for the entry's. The two
      agree whenever the occupations do, which is why this only
      matters once entries taken at different occupations are
      admitted.
    - Note that ``s . y = tr(dD dF)`` consequently holds only at fixed
      occupations. It is a pairing in density space; the quasi-Newton
      condition needs the one in the chart the model works in, and the
      two coincide exactly while the map between them is the identity.
      The test checks the identity at fixed occupations and the
      displacement against a known ``kappa`` when they move.
    - Two new settings: ``arh_subspace_threshold`` (default 1e-6), the
      relative singular-value cutoff below which a history direction
      is dropped, and ``arh_occupation_scale`` (default 0, meaning
      fitted as above).
    - ``atomtest`` gained a ``--set key=value[,key=value]`` flag that
      applies arbitrary solver settings, dispatching on the type the
      solver's own catalog reports. Unknown keys throw, so a typo in a
      benchmark sweep fails loudly rather than silently measuring the
      default.
    - Measured on the DIIS-free path, where the rotation step does the
      optimising rather than assisting an extrapolation. Oxygen with
      PBE/cc-pVDZ converges in 21 Fock builds against 27 for both CG
      and L-BFGS at M = 1, and 23 against 50 and 61 at M = 3. Iron at
      M = 5, the case the occupation coordinates were built for, takes
      241 against 564 for both -- a factor of 2.3, and 315 with the
      rotation-only pairs, so a quarter of that gain is the occupation
      coordinates alone. Within the DIIS mixes the counts are equal or
      a few lower, the extrapolation having already done most of the
      work.
    - Krypton is the interesting case: ``"ODA + LBFGS"`` reaches the
      converged energy there but never satisfies the gradient
      criterion, its line search stalling with the DIIS error at
      3.0e-6 because every trial raises the energy by around 1e-10 --
      the energy signal has gone below the arithmetic noise while the
      gradient is still above threshold. ``"ODA + ARH"`` converges in
      31 Fock builds, taking the error from 4.0e-6 to 2.9e-7 on a step
      worth 1e-10 in energy. Curvature is what is left to steer by
      once the energy differences are noise.

#### Breaking Changes
* The default method mix is now ``"DIIS + ODA + LBFGS"`` rather than
  ``"DIIS + ODA + CG"``: the orbital-rotation step is taken by L-BFGS
  instead of preconditioned PR+ CG. Callers that relied on the old
  default can ask for it explicitly with
  ``set("methods", "DIIS + ODA + CG")``. The Armadillo shim's
  ``run()`` default argument and the Python binding's documented
  default follow suit.
* Requesting two methods that occupy the same slot now throws
  instead of silently resolving to one of them. This affects
  ``"CG + LBFGS"``, which previously ran L-BFGS and quietly ignored
  the ``CG`` request -- the two are alternative implementations of
  the single orbital-rotation step, not steps that can both run.
  The new ``"DIIS + LCIIS"`` combination is rejected for the same
  reason. Method strings naming only one of each pair are
  unaffected.

#### Enhancements
* Each setting is now a single self-describing object. An option used
  to be named in three places -- the ``options()`` catalog and one
  branch each in the ``set_*`` and ``get_*`` if-else chains -- so 31
  settings carried 88 copies of their key strings, and adding a knob
  meant editing three lists that nothing checked against each other.
  A catalog entry could disagree with what the dispatch actually
  accepted, silently.
    - A setting's value, key, documentation and validator now live
      together in one declaration: ``Setting<T>`` holds the value and
      derives from an abstract ``SettingBase`` carrying everything
      that does not depend on ``T``. ``options()``, ``set_*`` and
      ``get_*`` all read off the settings themselves, so a key string
      appears exactly once and the catalog cannot drift out of step
      with the dispatch.
    - Settings register themselves as they are constructed, so adding a
      knob is one declaration and nothing else -- there is no list to
      remember to append to, and a setting that is declared but not
      published is no longer possible to write. The registry records
      each setting's offset within the solver rather than its address,
      so the facade keeps working after the solver is moved.
    - ``Setting<T>`` converts implicitly to ``const T &`` and assigns
      from ``T``, so the ~195 uses of these members in the solver
      read as they did before.
    - A setting may carry a write validator or a read recompute, as a
      pointer to the member function implementing it. This covers the
      two settings that validate (``error_norm`` probes the norm,
      ``methods`` parses and canonicalises) and the five read-only
      diagnostics.
    - The typed getters and setters are one shared template body
      differing only in ``T``, replacing three near-identical
      if-else chains.
    - Setting a read-only diagnostic now reports it as read-only
      rather than as an unknown key.
* ODA now tries a "collapsed" polytope first: when the full
  skeleton enumeration would exceed ``n_particle_types``, each
  particle's skeleton set is replaced by its arithmetic mean --
  one smudged skeleton per particle, one lambda per particle.
  ``npars`` drops from the pathological "high-symmetry guess"
  case (e.g. 28 dimensions in the trace that motivated it) down
  to a handful, cutting vertex evaluations by an order of
  magnitude. If the collapsed pass fails to descend, the full
  skeleton set retries transparently -- trust-region refits and
  everything else apply only to the fallback path. Extra benefit:
  the smudged density has support in every degenerate orbital, so
  the SCF can't collapse onto a symmetry-broken saddle from the
  first-iteration guess.
* Convergence-time full-polytope check. When ``run()`` sees
  ``converged() == true``, ODA is allowed, and the last ODA step
  used the collapsed skeleton set, it invokes
  ``optimal_damping_step(force_full=true)`` before breaking the
  SCF loop. That confirms the converged iterate is stationary in
  the full skeleton set and not just in the collapsed one;
  sub-noise descents (below
  ``0.1 * max(convergence_threshold, K * noise_floor)``, the same
  tolerance the trust-region refit loop uses) are rejected so an
  already-converged SCF doesn't churn at the arithmetic floor.
  After a genuine descent the state machine picks the same
  ``{OrbitalRotation, DIIS, ODA}`` preference the normal post-ODA
  transition uses, so the next step relaxes at the new
  occupations before revisiting DIIS.
* The per-iteration orbital-occupations block in the verbose SCF
  trace now prints before the ``converged()`` check, so the
  occupations at the final iterate reach the log.
* Speed up the ODA quadratic-model construction and the whole
  DIIS / ADIIS / EDIIS extrapolation stack via a coordinated
  refactor of the ``C * diag(n) * C^dagger`` and
  ``(A * B).trace()`` idioms scattered through ``scfsolver.hpp``.
    - New ``build_density_block_`` is the single density-matrix
      construction path. Extracts the occupied natural orbitals
      first, so the outer product is O(k * n_basis^2) instead of
      the naive O(n_basis^3). Now used by ``get_density_matrix_block``
      (so every DIIS-side caller benefits), the ODA polytope-vertex
      cache, ``density_overlap``, the level-shift block, and
      ``interpolate_density``.
    - New ``tr_of_product_`` computes ``tr(A * B)`` in O(n_basis^2)
      via ``(A * B.transpose()).sum()`` instead of the O(n_basis^3)
      matmul-then-trace form. Replaces every ``(A * B).trace()``
      call in the file: ADIIS linear + quadratic, EDIIS quadratic,
      ``density_overlap``, and the ODA gradient / Hessian
      ``trace_diff``.
    - ``adiis_quadratic_term`` and ``ediis_quadratic_term`` now
      pre-cache the block densities so the doubly-nested history
      loop no longer re-materialises the same density
      ``nhist`` times.
    - ``optimal_damping_step`` pre-caches each axis-vertex density
      once via ``materialise_density``; the trace-diff loop then
      does one O(n_basis^2) elementwise trace per call rather than
      an O(n_basis^3) matmul that started from raw ``(C, n)``.
      Total polytope-setup cost drops from ``O(npars^2 * n_basis^3)``
      to ``O(npars * k * n_basis^2 + npars^2 * n_basis^2)`` -- a
      ~20x-30x speed-up already at ``npars = 28`` and increasing
      with basis size, which unlocks useful ODA behaviour on
      heavy-atom / large-basis systems where the older
      implementation appeared to hang after "Roothaan step in
      dimension N".
* Cache DIIS / ADIIS / EDIIS primitives across SCF iterations. Two
  mutable caches keyed on each history entry's stable iteration
  index:
    - ``diis_commutator_cache_`` holds the AO-basis commutator
      ``FP - PF`` per entry per block. Built via the rank-k route
      ``FP = F * (C_occ * diag(n_occ) * C_occ^dagger)`` in
      O(k * n_basis^2), with ``PF = (FP)^dagger`` skipping the
      second full matmul entirely. Dot products of AO-basis
      commutators are unitary-invariant, so the same cache backs
      both ``diis_error_matrix_element`` (no projection) and the
      per-iteration ``diis_residual`` used by ``converged()``.
    - ``trace_DF_cache_`` holds sum-over-blocks ``tr(D_a * F_b)``
      for every entry pair. Every ADIIS / EDIIS linear or
      quadratic entry is one of at most four such primitives, so
      the whole extrapolation-matrix build reduces to map lookups
      once the primitives are populated.
    - ``diis_matrix_cache_`` holds the final DIIS error-matrix
      element ``B(i, j) = -Re(tr(X_i * X_j))`` summed over blocks,
      keyed by the sorted pair of iteration indices. Once cached,
      each subsequent ``B(i, j)`` access is one map lookup with no
      per-block work at all -- the doubly-nested DIIS matrix loop
      only ever visits the row / column of the newest entry.
    - ``density_diff_cache_`` holds the sum-over-blocks Frobenius
      distance ``||D_i - D_j||`` from ``density_matrix_difference``,
      keyed the same way so the ``cleanup()`` sweep runs at
      steady-state cost.
    - Caches are cleared by ``initialize_with_*`` and
      ``reset_history()``. Stale entries otherwise linger, which
      is cheap: one ``Tbase`` per scalar cache and O(n_basis^2)
      per commutator block.
    Per-iteration cost of the DIIS matrix build drops from
    O(nhist^2 * n_basis^3) naive to O(n_basis^2) steady-state
    (only the new entry's row / column populates).
* ODA trust-region refinement: after each ``optimal_damping_step``
  accepts a trial candidate, re-anchor the polytope quadratic
  model at the accepted iterate using the gradient observed there
  (free from the already-computed Fock matrix), re-solve the QP,
  and accept the refined point if it lowers the energy. Exact for
  Hartree-Fock along any linear ray; for DFT, catches the
  residual non-quadraticity the initial axis-vertex data missed.
  Number of refits is capped by the new ``max_oda_refits``
  setting (default 3, 0 disables). Refits pre-check the model's
  predicted improvement against a fraction of the SCF convergence
  threshold and skip the Fock build entirely when the prediction
  is below noise, so idle refits cost nothing on well-conditioned
  problems.
* ODA candidate selection is now model-first. The polytope
  quadratic-model minimum and the 1D cubic Hermite probes (one per
  axis, one per pair-diagonal edge) are ranked by their
  polynomial-predicted energy, and the trial loop evaluates them in
  that order. Along a linear ray in density space the Hartree-Fock
  energy is exactly quadratic and the polynomial fits are exact for
  HF and closely accurate for DFT near convergence, so the sorted
  first candidate is normally the accepted step -- one Fock build
  per ODA call in place of the worst-case
  O(candidates × backoff scales) sweep.
* The 1D probes emit only interior *minima* of the fitted cubic
  (the second root of the derivative is a maximum and is never a
  useful trial step), and de-duplicate the doubled root returned
  when the cubic degenerates to a quadratic. Axis and edge probes
  now share one fitting routine.
* The orbital rotation ``exp(K)`` is taken from Eigen's matrix
  exponential -- scaling and squaring with a diagonal Pade
  approximant -- instead of being assembled from an eigen-
  decomposition. Diagonalising an anti-Hermitian ``K`` needs
  eigenvalues that are real, which for a real anti-symmetric ``K``
  means promoting the whole problem to complex arithmetic to
  diagonalise ``iK``. The Pade route is a handful of matrix
  multiplications and one LU solve and stays in real arithmetic
  throughout, and a diagonal Pade approximant of an anti-Hermitian
  argument is unitary exactly, not just to the approximation
  error: with ``r(K) = D(K)^-1 N(K)`` and ``N(K)^dagger = N(-K) =
  D(K)``, the numerator and denominator commute as polynomials in
  ``K``. Measured 3.8x to 9.8x faster over matrix sizes 100 to
  1000, and up to 16.6x at the small rotation norms an SCF
  approaching convergence actually produces. Orthogonality of the
  result holds to 0-20 machine epsilons across every instantiated
  scalar type.
    - Eigen carries the Pade path only for the scalar types it
      names, routing everything else through a complex Schur
      decomposition. For ``_Float128`` that is both slower than
      diagonalising ``iK`` directly (49 s against 8.5 s at
      n = 150) and reintroduces the complex promotion the Pade
      path exists to avoid, so the choice is gated on Eigen's own
      ``is_exp_known_type`` and the eigendecomposition is kept as
      the fallback. Naming Eigen's trait rather than copying its
      list means the gate fails to compile if the trait goes away
      and picks up any type Eigen adds, instead of silently
      reverting to Schur.
    - This makes the library depend on Eigen's ``unsupported/``
      module tree, ``unsupported/Eigen/MatrixFunctions``.
* The trust radius for the line search along ``exp(tK)`` no longer
  forms eigenvectors it does not use. It is set by the largest
  ``|eigenvalue|`` of ``-iK`` alone, so the decomposition runs with
  ``Eigen::EigenvaluesOnly``: the tridiagonal reduction is kept and
  the back-transformation dropped, for the same eigenvalues to the
  bit. Measured 1.7x to 5.9x faster over block sizes 20 to 800.
  Per orbital-rotation step this runs once, next to the exponential
  above.

#### New Features (continued)
* Route all library log output through a caller-installable
  callback. `SCFSolver::logger(std::function<void(int level, const
  std::string & msg)>)` registers a sink; the library builds each
  formatted message via `vsnprintf` and either hands it to the sink
  (when set) or writes it to stdout (the pre-callback default). The
  `level` argument is the minimum ``verbosity`` at which the
  message would print, so callers can route different severities to
  different destinations. Python bindings expose it as
  `solver.logger(callback)` / `solver.logger(None)` /
  `solver.has_logger()`.
* Every `printf` call in `openorbitaloptimizer/scfsolver.hpp`
  (60 sites, excluding commented-out debug prints) migrated to
  `log_(level, fmt, ...)`. Every `std::cout << ...` matrix / vector
  dump (12 live sites) migrated to `log_stream_(level) << ...`,
  an ostream-style companion proxy that RAII-flushes into the same
  sink. The internal levels reflect the pre-existing
  `if(verbosity_ >= N)` gates; unconditional diagnostic warnings
  and the print_history dump use level 0.


#### Breaking Changes
* The per-option typed getters and setters on `SCFSolver` have been
  removed. Use the string-keyed façade instead:
  `solver.set(key, value)`, `solver.get_real / get_int /
  get_string(key)`, `SCFSolver::options()`. The Armadillo compatibility
  shim keeps its typed forwarders and now routes them through the
  façade.
* `SCFSolver::run()` no longer takes a method-mix argument; it
  consumes the `methods` setting, which is normalised to canonical
  uppercase on `set()`. Migrate `solver.run("ODA + CG")` to
  `solver.set("methods", "ODA + CG"); solver.run()`.
* Python bindings dropped the per-option typed methods (`verbosity`,
  `convergence_threshold`, `maximum_iterations`, ...). Use
  `solver.set(key, value)` or the attribute-style
  `solver.settings.<key> = value`. `SCFSolver` and `OptionInfo` are
  now importable from the `openorbital` package top level.

#### New Features
* SCF convergence threshold is now clamped to an arithmetic-precision
  floor: the effective threshold is
  `max(convergence_threshold, K * noise_floor)`, where `noise_floor`
  is a per-run estimate of the roundoff floor of the DIIS residual
  `C^dagger [F, P] C` frozen from the initial Fock. `K` defaults to
  10 and is tunable via `noise_safety_factor`. `__float128` runs are
  unaffected because their epsilon is tiny; the clamp mainly rescues
  low-precision runs from spinning below what the arithmetic can
  resolve. Callback-driven convergence
  (`callback_convergence_function`) is untouched.
* Introduce a string-keyed settings façade on `SCFSolver`:
  `set(key, value)`, `get_real/get_int/get_string(key)`, and a
  static `options()` catalog listing every knob and read-only
  diagnostic with its type and one-line description. Downstream
  callers now only need to know this triple to reach any setting,
  making JSON/dict-shaped configuration pipe-throughs trivial.
* Add `openorbital.Settings`: an attribute-style proxy that dispatches
  through the C++ catalog, so
  `solver.settings.convergence_threshold = 1e-9` is equivalent to
  `solver.set("convergence_threshold", 1e-9)` with the same catalog
  validation, `dir()` completion, and read-only diagnostic guards.
* Add `SCFSolver::print_settings()` which dumps every catalog entry
  with its current value. Read-only diagnostics that aren't yet
  computable (e.g. `converged` before the first `initialize_with_*`)
  print as `n/a` instead of throwing. Python bindings expose it as
  `solver.print_settings()` and the same output backs
  `str(solver.settings)`.
* Add `SCFSolver::citation()` and `SCFSolver::print_citation()` so
  downstream drivers can echo the canonical reference (Lehtola &
  Burns, J. Phys. Chem. A **129**, 5651 (2025);
  doi:10.1021/acs.jpca.5c02110) without hardcoding it, and add a
  `CITATION.cff` at the repository root so GitHub's "Cite this
  repository" button and citation-tracking tools resolve it
  automatically.
* The three PySCF drivers grew an `options=None` dict argument on
  `kernel()`; entries are forwarded via `solver.set(key, value)`
  before the run.

#### Enhancements
* `L-BFGS` history depth is now controlled by the shared
  `maximum_history_length` setting; the private
  `lbfgs_history_size_` knob has been folded away.
* `brute_force_search_for_lowest_configuration` now saves and
  restores `verbosity` and `frozen_occupations`, so calling it no
  longer permanently silences and thaws the parent solver.
* Removed the deprecated `run_optimal_damping()` alias on the
  Armadillo compatibility shim.

#### Bug Fixes
* The Aufbau cleanup is no longer rejected on atoms with a fractionally
  occupied Fermi level. It swapped the occupation vector but judged it
  on orbitals that were relaxed for the *old* occupations, so it lost
  to the converged mixed density every time and was discarded --
  silently, its return value being dropped. The occupations reported
  stayed the mixed ones, Rydberg tail and all.
    - The step now relaxes the orbitals at the occupations it chose,
      to stationarity, and compares only the relaxed energy. On iron
      (PBE/cc-pVDZ) that alone took the gap from 1.64e-2 Eh to
      9.74e-4.
    - The skeleton set is established once and held fixed across the
      cleanup's passes. Which orbitals are degenerate is a property of
      the solution being refined, and re-deriving it from each relaxed
      iterate destroyed it: the relaxation moves the cluster apart by
      more than ``optimal_damping_degeneracy_threshold``, after which
      the walk no longer recognises it and hands back a simplex of
      dimension zero.
    - What was left was a curvature the model could not see. ODA
      expands the energy in the occupations at *fixed* orbitals, so it
      misses the response term in
      ``H_relaxed = H_ll - H_lk H_kk^-1 H_kl``, and being too stiff it
      stops short: on iron it put the beta 4s/3d split at 0.826/1.174
      where the SCF finds 0.657/1.343 -- both on the same
      one-parameter line, so this was never a missing degree of
      freedom. Alternating the two halves does not recover it either;
      each is exact given the other, so the alternation has a fixed
      point and sits down at it.
    - Where the Fermi level is a single degenerate pair, the cleanup
      now models the coupling. It relaxes at the sampled points and
      fits a cubic through relaxed data rather than forming the Schur
      complement, which would need orbital response equations the
      solver does not have. The endpoint slopes are free: at an
      orbital-stationary point the orbital-response contribution to
      ``dE/dlambda`` carries a factor ``dE/dkappa = 0``, leaving
      Hellmann-Feynman.
    - Iron's minimum now lands at lambda = 0.6596, against the SCF's
      0.6567, and reaches 1.3e-6 Eh *below* the converged mixed
      density, so the cleanup is accepted and the reported occupations
      are Aufbau. Oxygen accepts at M = 1, 3 and 5 across three method
      mixes with the reference energies unchanged.
    - This stays out of the SCF loop deliberately: it cost about 100
      Fock builds on iron against 576 for the whole run, the loop
      already performs the alternation across its iterations, and the
      free-slope argument holds only at a stationary point.
    - Fermi levels spanning more than two orbitals still fall back to
      the alternation and may still be rejected.
    - The search starts at the polytope point whose occupations best
      reproduce the ones the solver is standing on, found by least
      squares on the same simplex the energy model is minimised over.
      Starting at a skeleton vertex instead put the first sample 0.06
      to 1.0 Eh above the converged density, since for a fractionally
      occupied Fermi level the promoted vertex is a plain Aufbau fill
      that dumps the whole remainder onto one orbital. Everything after
      that was a climb back, and a climb that stops short reads as the
      Aufbau state losing on energy when it has merely not arrived --
      lanthanides and actinides recovered 69% to 98% of it and were
      rejected by 7 to 27 mEh, while the occupations they were rejected
      in favour of were already Aufbau to 2.2e-7, worth about 1e-6 Eh.
    - The cleanup runs on every way out of the SCF loop, not only the
      converged one. Missing the threshold by a few percent used to
      cost the entire Aufbau structure: a run stalling at 1.06e-7
      against 1e-7 kept the raw mixed density and its whole Rydberg
      tail, some fifty orbitals down to 1e-12. It is adopted only if it
      does not raise the energy, so it cannot make the reported answer
      worse.
    - Acceptance is judged against the descent the solver already calls
      noise rather than demanding a strict improvement, which had been
      discarding exactly-Aufbau occupations over gaps as small as
      1.8e-11 Eh.
    - With these, all 118 spherically averaged neutral atoms come out
      exactly Aufbau.
* ``atomtest`` gained ``--M 0``, which runs the spin-restricted code
  whatever the shell structure and gives an open shell a fractionally
  occupied Fermi level rather than refusing it. The ``restricted``
  flag it supersedes is removed; it was computed and never read.
* ``atomtest`` accepts functional id 0. Libxc numbers its functionals
  from one, so a non-positive id means "nothing to add here" -- 0 as
  written by someone wanting exchange only, -1 as returned by
  ``xc_functional_get_number`` for a name it does not know. Only -1 was
  honoured, and only in two of the three builders, so ``--xfunc 0``
  threw and NEO without an electron-proton correlation functional could
  not run at all.
* The occupations reported at convergence are now Aufbau: full below
  the Fermi level, exactly zero above it, fractional only inside the
  degenerate cluster at it. What was reported before was the natural
  occupation vector of a *mixed* density, and a mixture of densities
  carrying different orbitals is not idempotent shell by shell, so a
  nominally full shell came out at ``max_occ - epsilon`` and orbitals
  well above the Fermi level carried ``epsilon``.
    - At convergence the SCF takes one more ODA step with the current
      density left out of the polytope. That is the whole fix, because
      the mixing is the whole problem: every skeleton is an Aufbau
      filling of one common set of orbitals, so a combination of
      skeletons alone has those orbitals as its natural orbitals and
      the combined occupation vector as its occupations, exactly. The
      Aufbau structure is inherited rather than imposed, and the
      Fermi-level fractions come from minimising the energy over the
      skeleton simplex rather than from a filling rule.
    - Leaving the reference out is a reparametrisation rather than a
      new code path: one skeleton is promoted to the ``lambda = 0``
      vertex and dropped from the axes, so ``n`` skeletons are
      described by ``n-1`` parameters -- the simplex they span. The
      polytope is still ``{lambda >= 0, sum(lambda) <= 1}``, so the
      QP, the cubic rays and the backoff scaling are untouched.
    - Largest occupation above the Fermi level at convergence, oxygen
      with PBE/cc-pVDZ: 6.6e-12 to 0 (M=3) and 2.0e-12 to 0 (M=1),
      with the deviation of full shells from ``max_occ`` following.
      Reference energies are unchanged, M=3 by a digit in its favour.
* ``interpolate_density`` no longer hands back a density with
  default-constructed blocks for a particle that has no trial
  occupations; it copies the reference across instead.
* ODA no longer loses particle number. The mixed density's natural
  occupations were snapped to exactly zero below
  ``sqrt(eps) * max_occ`` -- 3.0e-8 for an s block, 2.1e-7 for an f
  block -- with nothing to put the discarded occupation back. Because
  the snapped density is fed forward as the next iterate, the loss
  compounded rather than staying an output artifact, and a converged
  SCF could report occupations summing to less than the requested
  particle number by ~1e-6 electrons on a block with a Rydberg tail.
  Tightening ``convergence_threshold`` did not help: the shortfall is
  a fixed truncation, not an unconverged iterate.
    - The mixed density is a convex combination of same-basis
      densities, so it is positive semidefinite and carries the trace
      its ingredients carried. Both are now enforced -- negative
      occupations are clamped, then the block is rescaled to the
      trace its inputs sum to -- rather than left to the accuracy of
      the eigendecomposition. Nothing is discarded silently.
    - The ``sqrt(eps)`` tolerance was justified by the conditioning
      of a density "projected between basis sets and then mixed", but
      this call site mixes densities built in the same basis in that
      very iteration, so it carries ordinary elementwise roundoff and
      not ``cond * eps``. Measured on the oxygen ODA steps, the
      eigendecomposition's own error is 6.2e-15 while the snapping
      discarded up to 8.1e-9 from a single block.
    - The abort guard is unchanged and still fires below
      ``-eps^(1/4)``; it is checked on the raw occupations, before
      the clean-up could hide anything from it.
    - Measured end to end, the converged occupation sum minus the
      requested particle number goes from -2.5e-12 to 0 (O, M=1),
      -6.2e-12 to 0 (O, M=3), and -6.6e-10 to -1.1e-14 (Fe, M=5).
      ``atomtest`` now checks this, so the existing ``run1`` / ``run2``
      ctest cases fail if it regresses.
* L-BFGS no longer stalls when combined with ODA. On open-shell
  oxygen (PBE/cc-pVDZ, M = 3) ``"ODA + LBFGS"`` used to exhaust
  1000 iterations at −74.1255 Eh, 0.85 Eh above the answer that
  ``"ODA + CG"`` reached in 24; it now converges in 26 to the same
  −74.9722875549 Eh. Three defects contributed, each fixed:
    - The curvature pair recorded the search *direction* ``d`` as
      its ``s``, but the step taken is ``exp(t K)``, a displacement
      of ``t*d``. The paired ``y = g_new - g_old`` was therefore
      measured across a different displacement than ``s``
      described. Since the line search routinely accepts ``t``
      orders of magnitude away from 1 -- 1.4e-8 in the stalling
      run -- the stored pairs were inconsistent with each other by
      the same orders of magnitude, and the resulting two-loop
      direction exceeded the preconditioned-SD one in length by up
      to a factor of 5.7e6.
    - Neither the L-BFGS history nor the PR+ CG direction was
      cleared after an accepted extrapolation, so the recorded
      gradient belonged to the pre-extrapolation iterate and the
      next ``y`` spanned the DIIS jump as well as the rotation.
      ODA accepts already cleared it; DIIS accepts now do too.
      This is what left ``"DIIS + LBFGS"`` needing 200 iterations
      at M = 3, now 15.
    - The two-loop direction replaced the preconditioned-SD one on
      the strength of ``d.g < 0`` alone. Descending is not the same
      as being worth taking: the direction it displaced routinely
      descended several times faster per unit length. The
      replacement now has to retain at least half that rate.
* ODA no longer aborts with "Negative natural occupation numbers"
  on an SCF started from a density projected between two
  different basis sets. Two independent problems fed the same
  crash, and both are fixed.
    - The polytope parameters are now projected onto their
      simplex ``{lambda >= 0, sum(lambda) <= 1}`` before the
      mixed density is built. The bound-constrained solve
      enforces that simplex only to the accuracy of its
      constrained linear solve, so an ill-conditioned reduced
      Hessian -- exactly what a cross-basis projection produces
      -- can leave ``sum(lambda)`` above one by of order
      ``cond * eps``. Any overshoot gives the reference density a
      negative weight, so the mixed density is no longer positive
      semidefinite. The trial loop only ever scales candidates
      *down*, so nothing in the algorithm wanted a step outside
      the simplex in the first place.
    - The natural-occupation noise tolerances are now derived
      from powers of the machine epsilon and scaled by the
      block's maximum occupation: values below ``sqrt(eps)`` are
      clamped to zero, and the abort fires below
      ``-eps^(1/4)``. The old absolute ``10 * eps`` / ``100 * eps``
      cutoffs were right for a freshly built density but far too
      tight for a projected-and-mixed one, whose eigendecomposition
      carries error of order ``cond * eps``. The guard now
      triggers only on occupations negative enough to mean a
      genuinely corrupt density, and the thresholds still tighten
      automatically in ``__float128``.
* ``initialize_with_orbitals`` now resets ``last_oda_via_collapsed_``
  and ``last_polytope_dimension_`` along with the rest of the
  per-run state. Left set from an earlier run on the same solver,
  the former made the convergence-time full-polytope check fire on
  a run that never took a collapsed ODA step.
* The degenerate-cluster walk had two implementations with opposite
  boundaries: the ODA skeleton enumeration included an orbital
  exactly one threshold away from the cluster start, the
  active-rotation count excluded it. The latter sizes the post-ODA
  CG burst *for the clusters the former creates*, and its docstring
  already claimed the two definitions were identical. Both now go
  through one ``degenerate_cluster_end_``, on the ODA boundary.
* Corrected the ADIIS linear-term docstring: the loop actually
  computes ``2 * <D_i - D_0 | F_0>`` (the standard ADIIS model of
  Hu & Yang, JCP 132, 054109), not ``2 * <D_i - D_0 | F_i - F_0>``
  as the old comment claimed. Computation itself is unchanged.

#### Misc.
* Gave each duplicated concept in ``scfsolver.hpp`` a single
  definition. No behaviour change; energies are bit-identical on
  both the ODA and the DIIS paths.
    - The effective convergence threshold had three spellings, two
      of them the same quantity written ``Tbase(0.1)`` and
      ``Tbase(1)/Tbase(10)``. Now
      ``effective_convergence_threshold_()`` and
      ``minimum_useful_descent_()``, which also retires the last
      bare fractional literal in the file.
    - The natural-orbital re-diagonalisation had two copies that had
      already drifted. The drift turns out to be *correct* and is
      now documented rather than removed: ODA mixes convexly, so its
      result must be positive semidefinite, while DIIS extrapolates
      affinely with weights that may be negative, so negative
      occupations are a normal outcome there. ``natural_orbitals_``
      shares the mechanics and takes the noise tolerance as an
      argument; ``require_nonnegative_occupations_`` is the opt-in
      guard. Its doc comment also claimed the sign flip yields
      *increasing* occupations — it yields decreasing ones, which is
      both the convention and what the code wants.
    - The active-natural-orbital extraction (count, allocate, fill,
      tolerance ``10 * max_occ * eps``) was open-coded in both the
      rank-k density build and the rank-k commutator build. Now
      ``active_natural_orbitals_``.
    - ``trace_diff`` open-coded the O(N^2) elementwise trace
      ``tr_of_product_`` already provides, and ``materialise_density``
      was an ODA-local lambda doing per block what
      ``build_density_block_`` does; it is now the member
      ``build_density_blocks_``.
    - The 449-line body of the ODA ``attempts`` loop was never
      re-indented when the loop was wrapped around it. Whitespace
      only — ``git diff -w`` on that commit is empty.
* Bare ``double`` literals in ``scfsolver.hpp`` arithmetic sites
  (0.0, 1.0, 2.0, 10.0, 100.0, -1.0; 45 occurrences) wrapped in
  ``Tbase(N)``, so ``float`` and ``__float128`` instantiations no
  longer silently promote through ``double``. Bare 0.5 sites
  rewritten as ``Tbase(1)/Tbase(2)`` (7 occurrences), and
  scientific-notation / bare decimal defaults such as
  ``1e-4`` / ``0.02`` wrapped in ``Tbase(...)`` (13 sites) for the
  same reason.


## v0.4.0 / 2026-07-20

#### Breaking Changes
* The linear-algebra backend switched from Armadillo to Eigen 3.4. The public
  API keeps the `SCFSolver<Torb, Tbase>` template signature via a compat shim,
  but callers that reached into Armadillo types directly must migrate to the
  new `OpenOrbitalOptimizer::Matrix<T>` / `Vector<T>` / `IndexVector` aliases.

#### New Features
* [\#38](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/38) Port the
  library and `atomtest` from Armadillo to Eigen 3, unlocking arbitrary-precision
  scalar types. Adds a `_Float128` (libquadmath) instantiation of `SCFSolver`
  via `openorbitaloptimizer/quad_support.hpp`.
* [\#10](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/10) Bi-level
  optimal-damping + preconditioned CG state machine with a skeleton-density-matrix
  polytope for degenerate shells, orbital-rotation bursts after ODA, and an
  L-BFGS phase. `SCFSolver::run("DIIS + ODA + CG")` is the new default; the
  method mix is user-selectable via `run("DIIS")`, `run("ODA + CG")`, etc.
* [\#10](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/10) Optional
  batched Fock-builder callback (`set_batched_fock_builder`) that receives a
  list of trial densities in one call, letting integral / grid setup amortise
  across the ODA polytope axis-vertex sweep.
* [\#10](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/10) New
  `--oda` / `--odadegthresh` / `--maxiter` flags in the `atomtest` driver.

#### Enhancements
* Python bindings expose the new API surface: `run(methods=...)`,
  `set_batched_fock_builder`, `has_batched_fock_builder`,
  `clear_batched_fock_builder`, `optimal_damping_degeneracy_threshold`,
  `orbital_rotation_steps_after_oda`, `last_polytope_dimension`,
  `last_active_rotation_count`, `number_of_fock_evaluations`, `converged`.
* The three PySCF drivers (`atomic`, `molecular`, `diatomic`) register
  batched Fock builders automatically and forward `methods` through
  `kernel(methods=...)`.

#### Misc.
* Armadillo is no longer a dependency of the header-only library. `atomtest`
  and the compat shim still require Eigen 3.4+; Armadillo is only pulled in
  by the compat shim's `<-> arma::Mat` conversion helpers when a caller
  explicitly opts in.


## v0.3.0 / 2026-05-21

#### Breaking Changes

#### New Features
* [\#34](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/34) Add Python interface!
  This covers the most common `SCFSolver<double, double>` case; others added upon request.
* [\#34](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/34) Add a PySCF integration layer
  with three usage-aware drivers, exposed as the `openorbital` package.
* [\#35](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/35) Add option `oda_restart_steps`
  to set the number of steps with no DIIS energy improvement after which to use ODA independently of
  DIIS history length. Previously used `maximum_history_length`/2

#### Enhancements

* [\#35](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/35) Explicitly symmetrize matices
  for DIIS error calculation to avoid numerical issues.

#### Bug Fixes
* Fix four correctness bugs in SCF solver and CG optimizer
  - get_energy: bounds check used > instead of >=, allowing out-of-bounds access when
    `ihist == orbital_history_.size()`.
  - matricise(vec, dim): the per-block offset was never advanced, so every block read from offset 0
    of the input vector.
  - steepest_descent line search: the "decrease step" branch used std::max, which actually grew the
    step whenever the parabolic prediction was larger than step/20, stalling the search.
  - CG Polak-Ribière: divided by dot(g_prev, g_prev) without guarding against a zero previous gradient,
    propagating NaN into the search direction. Fall back to steepest descent in that case.

#### Misc.
* [\#30](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/30) `max(idx)` has been deprecated
  in Armadillo in favor of `index_max()` so switching to the new syntax.
* [\#36](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/36) Use modern FindPython with pybind11 module.
* Added a `CLAUDE.md` file to aid agents.



## v0.2.0 / 2025-08-12

#### Breaking Changes

#### New Features

#### Enhancements
 * [\#26](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/26) add callback so caller can
   apply its own convergence criteria.
 * MESA should use Aufbau occupations not MOM
 * Undo minimum error criterion by Garza and Scuseria to avoid penalizing large steps leading to a decrease in energy
 * Increase `pure_ediis_factor`
 * Implement callback functionality

#### Bug Fixes
 * [\#27](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/27) fix a `Col::subvec()` error
   with minimal basis sets.
 * [\#27](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/27) fix UHF with frozen
   occupations by disabling ODA.

#### Misc.
 * Added user guide to readme
 * Deploy docs site
 * Fix GCC 15 warnings


## v0.1.0 / 2025-03-30

#### New Features
 * [\#20](https://github.com/SusiLehtola/OpenOrbitalOptimizer/pull/20) Intf -- allow OOO and
   IntegratorXX to work on Windows.
 * All start-up functionality making library operational.

