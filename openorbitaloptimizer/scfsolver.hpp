/*
 Copyright (C) 2023- Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once
#include "types.hpp"
#include "eigen_compat.hpp"

#include <algorithm>
#include <array>
#include <any>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace OpenOrbitalOptimizer {

  namespace HelperRoutines {
    /// Fit cubic polynomial f(x) = a0 + a1*x + a2*x^2 + a3*x^3 to the
    /// data {f(0)=E0, f'(0)=dE0, f(x1)=E1, f'(x1)=dE1}.
    template<typename T>
    std::tuple<T,T,T,T> fit_cubic_polynomial_with_derivatives(T E0, T dE0, T x1, T E1, T dE1) {
      T a0 = E0;
      T a1 = dE0;
      T x1sq = x1*x1;
      T x1cu = x1sq*x1;
      T a3 = (dE1 + dE0)/x1sq - 2*(E1 - E0)/x1cu;
      T a2 = 3*(E1 - E0)/x1sq - (2*dE0 + dE1)/x1;
      return std::make_tuple(a0, a1, a2, a3);
    }

    /// Return the (real) zeros of the derivative f'(x) = a1 + 2*a2*x + 3*a3*x^2
    /// of the cubic polynomial f(x) = a0 + a1*x + a2*x^2 + a3*x^3,
    /// i.e. the candidate extrema of f. Throws if no real roots exist.
    template<typename T>
    std::pair<T,T> cubic_polynomial_zeros(T a0, T a1, T a2, T a3) {
      (void) a0;
      // Solve 3*a3*x^2 + 2*a2*x + a1 = 0
      T eps = std::numeric_limits<T>::epsilon();
      if(std::abs(a3) <= eps) {
        if(std::abs(a2) <= eps)
          throw std::logic_error("Cubic derivative is constant; no extrema");
        T x = -a1/(2*a2);
        return std::make_pair(x, x);
      }
      T disc = 4*a2*a2 - 12*a3*a1;
      if(disc < 0)
        throw std::logic_error("Cubic derivative has no real roots");
      T sq = std::sqrt(disc);
      T x1 = (-2*a2 - sq)/(6*a3);
      T x2 = (-2*a2 + sq)/(6*a3);
      return std::make_pair(x1, x2);
    }

    /// Project ``v`` onto the simplex {v_i >= 0, sum(v) <= 1}:
    /// negative entries are clamped to zero, and if the sum then
    /// still exceeds one the whole vector is rescaled by 1/sum.
    ///
    /// Rescaling rather than merely capping the sum matters wherever
    /// the individual entries are used as weights in their own right:
    /// capping the sum alone would leave the entries summing to more
    /// than one while the complementary weight (1 - sum) went to
    /// zero, which shifts the total.
    ///
    /// Used on the ODA polytope parameters, where the QP solver
    /// enforces the simplex only to the accuracy of its constrained
    /// linear solve. An ill-conditioned reduced Hessian -- what a
    /// density projected between two different basis sets produces --
    /// can leave the sum above one by of order cond * eps, which
    /// hands the reference density a negative weight and yields a
    /// non-positive-semidefinite mixed density.
    template<typename T>
    void project_onto_unit_simplex(Vector<T> & v) {
      for(Index i = 0; i < v.size(); i++)
        if(v(i) < T(0)) v(i) = T(0);
      T s = v.sum();
      if(s > T(1))
        v /= s;
    }

    /// Evaluate a polynomial with the given coefficients (index i =
    /// coefficient of x^i) at ``x`` via Horner's scheme.
    template<typename T, size_t N>
    T evaluate_polynomial(const std::array<T, N> & coeffs, T x) {
      T r = coeffs[N - 1];
      for (size_t i = N - 1; i-- > 0;)
        r = r * x + coeffs[i];
      return r;
    }

  }

  /// SCF solver class
  template<typename Torb, typename Tbase> class SCFSolver {
    static_assert(std::is_same_v<Torb, Tbase> ||
                  std::is_same_v<Torb, std::complex<Tbase>>,
                  "SCFSolver<Torb, Tbase>: Torb must be either Tbase or "
                  "std::complex<Tbase>");
    /* Input data section */
    /// The number of orbital blocks per particle type (length ntypes)
    IndexVector number_of_blocks_per_particle_type_;
    /// The maximal capacity of each orbital block
    Vector<Tbase> maximum_occupation_;
    /// The number of particles of each class in total (length ntypes, used to determine Aufbau occupations)
    Vector<Tbase> number_of_particles_;
    /// The Fock builder used to evaluate energies and Fock matrices
    FockBuilder<Torb, Tbase> fock_builder_;
    /// Optional batched Fock builder; if unset, evaluate_batch_ loops
    /// over fock_builder_ instead.
    BatchedFockBuilder<Torb, Tbase> batched_fock_builder_;
    /// Descriptions of the blocks
    std::vector<std::string> block_descriptions_;
    /// Callback function
    std::function<void(const std::map<std::string,std::any> & data)> callback_function_;
    /// Callback function to allow calling program to judge convergence (trumps convergence_threshold_)
    std::function<bool(const std::map<std::string,std::any> & data)> callback_convergence_function_;
    /// Sink for library log messages. If set, log_() calls the sink
    /// with ``(level, formatted-message)`` instead of writing to
    /// stdout. The library still gates on ``verbosity_`` before
    /// invoking the sink, so callers filter further only if they want
    /// to route different levels to different destinations.
    std::function<void(int level, const std::string & msg)> logger_;

    /** (Optional) fixed number of particles in each symmetry, affects
        the way occupations are assigned in Aufbau. These are used if
        the array has the expected size.
    */
    Vector<Tbase> fixed_number_of_particles_per_block_;

    /* Settings
     *
     * A setting is an object that owns its value and knows its own
     * name, so the storage, the key, the default, the documentation
     * and the writability are all declared in one place, at the
     * member declaration. The string facade (set/get by key,
     * options(), print_settings) walks the settings through the
     * SettingBase interface; the solver body reads and writes them
     * like plain members, via the implicit conversion and operator=.
     */

    /// Everything about a setting that does not depend on its type.
    class SettingBase {
    public:
      SettingBase(const char * key, const char * doc, bool writable)
        : key_(key), doc_(doc), writable_(writable) {}
      virtual ~SettingBase() = default;

      const char * key() const { return key_; }
      const char * doc() const { return doc_; }
      /// False for read-only diagnostics, which the facade refuses to set.
      bool writable() const        { return writable_; }

      /// "real", "int" or "string" -- which typed facade reaches this.
      virtual const char * type() const = 0;
      /// Used by print_settings, which does not know the value type.
      virtual void print_value(std::ostream & os) const = 0;

    private:
      const char * key_;
      const char * doc_;
      bool writable_;
    };

    /// Collects the solver's settings as they are constructed, so that
    /// declaring a setting is all it takes to publish it through the
    /// string facade -- there is no second list to keep in step.
    ///
    /// It records each setting's offset from the registry rather than
    /// its address. An address would belong to the object the setting
    /// was constructed in, and SCFSolver is copyable and movable, so a
    /// registry of addresses would point into the source object.
    /// Offsets describe the class layout instead, which is by
    /// definition the same in the destination object, so the
    /// implicitly generated copy and move stay correct and need no
    /// fixup.
    class SettingRegistry {
    public:
      /// Called by each Setting's constructor. The conversion to
      /// SettingBase * happens in the caller, so what is recorded is
      /// the offset of the base subobject -- exactly what settings()
      /// hands back.
      void add(const SettingBase * setting) {
        offsets_.push_back(reinterpret_cast<const char *>(setting) -
                           reinterpret_cast<const char *>(this));
      }

      std::vector<SettingBase *> settings() {
        std::vector<SettingBase *> out;
        out.reserve(offsets_.size());
        for(std::ptrdiff_t offset : offsets_)
          out.push_back(reinterpret_cast<SettingBase *>(
              reinterpret_cast<char *>(this) + offset));
        return out;
      }

      std::vector<const SettingBase *> settings() const {
        std::vector<const SettingBase *> out;
        out.reserve(offsets_.size());
        for(std::ptrdiff_t offset : offsets_)
          out.push_back(reinterpret_cast<const SettingBase *>(
              reinterpret_cast<const char *>(this) + offset));
        return out;
      }

    private:
      std::vector<std::ptrdiff_t> offsets_;
    };

    /// A setting holding a value of type T.
    ///
    /// The implicit conversion means the solver body can keep using a
    /// setting exactly like the plain member it replaced --
    /// ``if(verbosity_ >= 5)``, ``x * diis_epsilon_`` -- and
    /// operator= keeps internal writes (which are not subject to the
    /// writable flag) equally plain.
    template<typename T>
    class Setting : public SettingBase {
    public:
      /// Optional validator / canonicaliser, applied before a write
      /// that comes in through the string facade. It is a member
      /// function of the solver, so it can consult solver state
      /// without the setting holding a back-pointer -- which would
      /// dangle the first time the solver was moved. Null means
      /// "store the value as given".
      using Hook = T (SCFSolver::*)(const T &) const;

      /// Optional source. When set, the setting has no meaningful
      /// stored value: it is recomputed on every read. Used by the
      /// diagnostics that must reflect the solver's state right now
      /// rather than whatever it was when they were last written.
      using Source = T (SCFSolver::*)() const;

      /// Registers itself, so that a setting is published to the
      /// string facade by the act of declaring it.
      Setting(SettingRegistry & registry, const char * key, const char * doc,
              T value, bool writable = true, Hook hook = nullptr,
              Source source = nullptr)
        : SettingBase(key, doc, writable),
          value_(std::move(value)), hook_(hook), source_(source) {
        registry.add(this);
      }

      operator const T & () const { return value_; }
      const T & get() const       { return value_; }
      Hook hook() const           { return hook_; }
      Source source() const       { return source_; }

      /// Internal writes. Not gated on writable(): that flag governs
      /// only what the string facade will accept from a caller.
      Setting & operator=(T v) { value_ = std::move(v); return *this; }
      Setting & operator+=(const T & v) { value_ += v; return *this; }

      const char * type() const override {
        if constexpr (std::is_same_v<T, std::string>) return "string";
        else if constexpr (std::is_integral_v<T>)     return "int";
        else                                          return "real";
      }
      void print_value(std::ostream & os) const override { os << value_; }

    private:
      T value_;
      Hook hook_;
      Source source_;
    };

    /// Postfix increment for the Fock-evaluation counter, so the call
    /// sites stay ``number_of_fock_evaluations_++``.
    friend int operator++(Setting<int> & s, int) {
      int old = s.get();
      s = old + 1;
      return old;
    }


    /// Declared before every setting, so that it is alive by the time
    /// their constructors register with it: members are initialised in
    /// declaration order.
    SettingRegistry settings_;

    /// Convergence threshold for orbital gradient
    Setting<Tbase> convergence_threshold_{
        settings_, "convergence_threshold",
        "DIIS-error convergence threshold", Tbase(1e-7)};

    /// Occupation-space convergence threshold: how much occupation is
    /// allowed to sit outside the Fermi-level window before the SCF
    /// refuses to call itself converged. See ``aufbau_error``.
    ///
    /// A converged fractional-occupation solution is Aufbau as a
    /// matter of theory -- the minimum over the occupation simplex
    /// fills every orbital below the chemical potential, empties every
    /// orbital above it, and leaves fractional occupation only where
    /// the orbital energies are equal to it -- so the honest value of
    /// this bound is zero and the default only leaves room for
    /// arithmetic. Measured on oxygen, the error is exactly 0 several
    /// iterations before the DIIS error reaches its own threshold, so
    /// the bound does not delay a healthy SCF; it is here to stop an
    /// unhealthy one reporting occupations that the energy functional
    /// it just minimised does not actually support.
    ///
    /// Set to a negative value to switch the bound off and convergence
    /// back to the DIIS error alone.
    Setting<Tbase> aufbau_convergence_threshold_{
        settings_, "aufbau_convergence_threshold",
        "occupation outside the Fermi-level window allowed at convergence; "
        "negative disables the check",
        Tbase(1e-6)};

    /// Safety factor K for the arithmetic-precision clamp on the
    /// effective convergence threshold: the SCF is considered
    /// converged when the DIIS error drops below
    /// max(convergence_threshold_, K * noise_floor_). K = 0 disables
    /// the clamp. K > 0 keeps low-precision runs (float, and
    /// eventually MPFR at reduced precision) from spinning below
    /// what the arithmetic can resolve, while __float128 users see
    /// no change because their epsilon is tiny.
    Setting<Tbase> noise_safety_factor_{
        settings_, "noise_safety_factor",
        "K in effective threshold max(convergence_threshold, K * noise_floor)",
        Tbase(10)};

    /// Norm to use by default: root-mean-square error
    Setting<std::string> error_norm_{
        settings_, "error_norm",
        "DIIS error norm; one of rms, fro, inf, 1, 2",
        "rms",
        true,
        &SCFSolver::canonicalise_error_norm_};

    /// SCF method mix consumed by run(). Stored in canonical
    /// (uppercase) form: parse and validate on set("methods", ...).
    /// Supported tokens: "DIIS", "LCIIS", "ODA", "CG", "LBFGS",
    /// "ARH", joined with " + ".
    Setting<std::string> methods_{
        settings_, "methods",
        "SCF method mix consumed by run(); e.g. \"DIIS + ODA + LBFGS\", \"DIIS\", \"LCIIS + ODA + CG\", \"ODA + CG\", \"DIIS + ODA + CG\"",
        "DIIS + ODA + LBFGS",
        true,
        &SCFSolver::canonicalise_methods_};

    /// Start to mix in DIIS at this error threshold (Garza and Scuseria, 2012)
    Setting<Tbase> diis_epsilon_{
        settings_, "diis_epsilon",
        "pure-DIIS blend cutoff", Tbase(1e-1)};

    /// Threshold for pure DIIS (Garza and Scuseria, 2012)
    Setting<Tbase> diis_threshold_{
        settings_, "diis_threshold",
        "A/EDIIS blend cutoff (Garza-Scuseria)", Tbase(1e-4)};

    /// Damping factor for DIIS diagonal (Hamilton and Pulay, 1986)
    Setting<Tbase> diis_diagonal_damping_{
        settings_, "diis_diagonal_damping",
        "DIIS matrix diagonal damping", Tbase(0.02)};

    /// DIIS restart criterion (Chupin et al, 2021)
    Setting<Tbase> diis_restart_factor_{
        settings_, "diis_restart_factor",
        "DIIS history restart factor", Tbase(1e-4)};

    /// Number of history entries LCIIS extrapolates over. Capped
    /// separately from maximum_history_length_ because LCIIS holds
    /// the full M x M grid of mixed commutators [F_i, D_j] at once,
    /// so its memory grows as M^2 * n_basis^2 and its commutator
    /// build as M^2 rather than DIIS's M. 0 means "no separate cap".
    Setting<int> lciis_maximum_history_{
        settings_, "lciis_maximum_history",
        "history entries LCIIS extrapolates over (0 = no separate cap)", 6};

    /// Maximum Newton iterations in the LCIIS quartic minimisation.
    Setting<int> lciis_maximum_iterations_{
        settings_, "lciis_maximum_iterations",
        "max Newton iterations in the LCIIS quartic minimisation", 50};

    /// Convergence threshold on the LCIIS Newton step norm.
    Setting<Tbase> lciis_convergence_threshold_{
        settings_, "lciis_convergence_threshold",
        "convergence threshold on the LCIIS Newton step norm", Tbase(1e-10)};

    /// Criterion for max error for which to use optimal damping
    Setting<Tbase> optimal_damping_threshold_{
        settings_, "optimal_damping_threshold",
        "DIIS error above which ODA takes over", Tbase(1)};

    /// Energy gap below which orbitals are treated as degenerate when
    /// enumerating skeleton density matrices in optimal damping. The
    /// default of 0.01 Eh (~0.27 eV) is loose enough to group orbitals
    /// that are degenerate by molecular symmetry but split by numerical
    /// noise during convergence (typical of transition-metal d-shells
    /// under PBE + UHF/UKS) and tight enough to leave the
    /// well-separated valence levels of main-group molecules alone.
    /// Override through set("optimal_damping_degeneracy_threshold", eps).
    Setting<Tbase> optimal_damping_degeneracy_threshold_{
        settings_, "optimal_damping_degeneracy_threshold",
        "ODA orbital-degeneracy window (Eh)", Tbase(1e-2)};

    /// Maximum number of trust-region refit iterations run inside
    /// each optimal_damping_step after the initial descent step is
    /// accepted. Each refit re-anchors the polytope quadratic model
    /// at the current accepted iterate using the observed gradient
    /// there and re-solves the QP. Exact for Hartree-Fock along
    /// any linear ray; each refit costs one Fock build. Default 3;
    /// 0 disables refinement (falls back to the accepted step from
    /// the ranked trial loop).
    Setting<int> max_oda_refits_{
        settings_, "max_oda_refits",
        "max trust-region refits inside optimal_damping_step after acceptance (0 disables)",
        3};

    /// Maximum number of iterations
    Setting<int> maximum_iterations_{
        settings_, "maximum_iterations",
        "outer SCF iteration cap", 128};

    /// History length
    Setting<int> maximum_history_length_{
        settings_, "maximum_history_length",
        "DIIS and L-BFGS history depth", 10};

    /// Steps with no DIIS energy improvement after which to use ODA. Previously maximum_history_length_/2
    Setting<int> oda_restart_steps_{
        settings_, "oda_restart_steps",
        "steps of no DIIS progress before switching to ODA", 5};

    /// Number of orbital-rotation steps to take after each ODA step (when CG is the
    /// next state at all -- ODA accept with integer occupations still
    /// skips CG and hands directly to DIIS). The orbital-rotation steps relax the
    /// orbital rotations at the ODA-chosen occupations before DIIS
    /// gets its turn. Default value 0 means "use the active-rotation
    /// count from the most recent ODA call" -- the number of orbital-
    /// rotation pairs that lie inside a degenerate group and have
    /// different occupations, summed across particles and blocks
    /// (see last_active_rotation_count_). That count is at most one
    /// per Krylov dimension of the badly-preconditioned subspace, so
    /// the orbital-rotation burst naturally scales with the hard-to-precondition
    /// part of the active space; 1 orbital-rotation step is taken as a floor.
    /// Set to an explicit positive value to override.
    Setting<int> orbital_rotation_steps_after_oda_{
        settings_, "orbital_rotation_steps_after_oda",
        "orbital-rotation steps after each ODA (0 = use last_active_rotation_count)",
        0};

    /// Minimal normalized projection of preconditioned search direction onto gradient
    Setting<Tbase> minimal_gradient_projection_{
        settings_, "minimal_gradient_projection",
        "minimum preconditioned-CG projection on gradient", Tbase(1e-4)};

    /// Initial level shift used as the floor in the orbital-rotation
    /// preconditioner:
    ///     d_alpha = -g_alpha / (sigma + max(0, h_alpha)),
    /// starting from sigma = initial_level_shift_ on trial 1 of the
    /// sigma-line-search. A small floor keeps the well-conditioned
    /// DOFs in their near-Newton regime (where h_alpha dominates the
    /// denominator) instead of damping them at the constant 1 Eh that
    /// the previous default forced; the line search bumps sigma up
    /// adaptively when wrong-sign or near-degenerate DOFs cause the
    /// trial step to overshoot.
    Setting<Tbase> initial_level_shift_{
        settings_, "initial_level_shift",
        "orbital-rotation preconditioner floor", Tbase(1e-3)};

    /// Relative singular-value cutoff for the ARH curvature subspace.
    /// A history direction is dropped when its singular value falls
    /// below this fraction of the largest, on the same reasoning as
    /// the DIIS history mask: a direction the history barely explores
    /// carries a curvature estimate dominated by the noise in the
    /// density difference it was extracted from, and the ARH model
    /// would then impose that noise as exact curvature.
    ///
    /// The extraction divides a density difference by an occupation
    /// difference, which is bounded below only by
    /// occupation_change_threshold_, so nearly-equally-occupied pairs
    /// enter with an amplified displacement. This cutoff is what
    /// keeps such a direction from dominating the subspace.
    Setting<Tbase> arh_subspace_threshold_{
        settings_, "arh_subspace_threshold",
        "ARH curvature-subspace relative singular-value cutoff",
        Tbase(1e-6)};

    /// Relative weight of the occupation coordinates against the
    /// rotation ones when the ARH curvature directions are
    /// orthonormalised.
    ///
    /// The two kinds of coordinate are measured in one Euclidean
    /// metric, so their relative scale decides which combinations of
    /// history entries form the orthonormal basis the model is fitted
    /// in, and which directions the singular-value cutoff discards.
    /// Zero or negative selects the scale fitted from the curvature
    /// the pairs themselves report, which is the default and what
    /// ``rescale_occupation_coordinates_`` documents; a positive value
    /// pins it instead, which is what the sweep that motivated the
    /// fit used.
    /// Whether the relaxed occupation Hessian is built from second
    /// differences of perturbatively relaxed energies (nonzero) or by
    /// relaxing the orbitals at each polytope vertex (zero).
    ///
    /// The estimate costs one Fock build per point against tens for a
    /// relaxation. What it gives up is exactness: the correction is
    /// second order in the orbital gradient, so a vertex far from
    /// orbital stationarity is described worse than a relaxation
    /// describes it.
    Setting<int> perturbative_occupation_hessian_{
        settings_, "perturbative_occupation_hessian",
        "relaxed occupation Hessian from perturbation theory (0 = relax)",
        1};

    /// Sweeps of the KKT occupation refinement to attempt after the
    /// Aufbau cleanup. Zero switches it off.
    Setting<int> kkt_occupation_refinement_steps_{
        settings_, "kkt_occupation_refinement_steps",
        "sweeps of KKT occupation refinement after the cleanup (0 = off)",
        12};

    /// Residual below which the occupations are taken to satisfy the
    /// stationarity conditions. In energy units, those residuals being
    /// orbital-energy differences.
    Setting<Tbase> kkt_occupation_threshold_{
        settings_, "kkt_occupation_threshold",
        "orbital-energy residual accepted as occupation stationarity",
        Tbase(1e-7)};

    /// Occupation displacement used to probe the curvature by second
    /// differences. Large enough that the differences clear round-off
    /// in the energy, small enough to stay in the quadratic region.
    Setting<Tbase> kkt_occupation_probe_{
        settings_, "kkt_occupation_probe",
        "occupation step used to probe the curvature by finite differences",
        Tbase(1e-3)};

    /// Largest occupation change a single refinement step may make.
    ///
    /// The curvature it divides by is a second difference of energies
    /// that differ in the ninth decimal, so it is noisy, and a raw
    /// Newton step on a curvature that comes out too small is enormous
    /// -- measured, one such step moved 0.1 electrons, raised the
    /// energy by 4e-3 Eh and the residual by a factor of 140. The
    /// trust radius bounds the damage without needing the curvature to
    /// be right; it is halved after a rejected step and the sweep
    /// retried.
    Setting<Tbase> kkt_occupation_trust_radius_{
        settings_, "kkt_occupation_trust_radius",
        "largest occupation change a KKT refinement step may make",
        Tbase(5e-3)};

    Setting<Tbase> arh_occupation_scale_{
        settings_, "arh_occupation_scale",
        "ARH occupation-coordinate weight (<= 0 fits it from the history)",
        Tbase(0)};

    /// Level shift diminution factor
    Setting<Tbase> level_shift_factor_{
        settings_, "level_shift_factor",
        "level-shift diminution factor", Tbase(2)};

    /// Threshold for detection of occupied orbitals
    Setting<Tbase> occupied_threshold_{
        settings_, "occupied_threshold",
        "occupied-orbital detection cutoff", Tbase(1e-6)};

    /// Norm-squared tolerance for deduplicating skeleton occupations
    Setting<Tbase> occupation_change_threshold_{
        settings_, "occupation_change_threshold",
        "occupation-equality tolerance", Tbase(1e-6)};

    /// History cleanup criterion: keep only those density matrices that satisfy delta ||P0-Pi|| < min_{j>0} ||P0-Pj||
    Setting<Tbase> density_restart_factor_{
        settings_, "density_restart_factor",
        "history density-diff restart factor", Tbase(1e-4)};

    /// (Optional) freeze occupations altogether to their previous values
    Setting<int> frozen_occupations_{
        settings_, "frozen_occupations",
        "pin occupations across SCF (0 or 1)", 0};

    /// Verbosity level: 0 for silent, higher values for more info
    Setting<int> verbosity_{
        settings_, "verbosity",
        "0..30", 5};

    /// Noise floor of the DIIS error, frozen at the start of run()
    /// from the initial Fock. The one-electron part dominates basis
    /// conditioning, so refreshing this per iteration would be
    /// noise itself; freezing keeps the effective threshold stable
    /// across the run.
    Setting<Tbase> noise_floor_{
        settings_, "noise_floor",
        "frozen roundoff floor of DIIS error, populated by run()",
        Tbase(0),
        false};

    /// How many times the relaxed polytope search re-anchors its
    /// quadratic model and steps on it.
    ///
    /// Zero by default, because the walk has been measured and does
    /// not pay. It did nothing at all until the relaxation estimate
    /// feeding its curvature was gated -- its quadratic programme
    /// returned the anchor it was given and the loop broke on the
    /// first pass -- and once it began walking, the answers did not
    /// move: iron at M = 0 and M = 5 and a La+ cation with PBE all
    /// reproduce their energies and their Fermi-level residuals to the
    /// last digit with the walk switched off, for one, one and sixty
    /// fewer Fock evaluations. The initial projected sample settles
    /// which orbitals are fractional and the occupation refinement
    /// settles where inside them the occupations sit; the walk between
    /// them re-derives a point both ends already reach.
    ///
    /// It is kept rather than removed because all three cases have a
    /// polytope of one or two dimensions. A system with several
    /// coupled fractional shells may yet need it, and leaving the
    /// setting makes that a discoverable regression instead of a
    /// silent one.
    Setting<int> relaxed_occupation_refinements_{
        settings_, "relaxed_occupation_refinements",
        "refinement steps taken by the relaxed polytope search",
        0};

    /// Largest rotation the perturbative relaxation estimate will
    /// extrapolate over. The estimate is second order, so it is worth
    /// having only while the step it predicts stays inside the radius
    /// a quadratic describes; past that it is not a rough answer but a
    /// wrong one.
    Setting<Tbase> perturbative_relaxation_max_step_{
        settings_, "perturbative_relaxation_max_step",
        "largest predicted rotation the relaxation estimate will trust",
        Tbase(0.1)};

    /// Number of Fock matrix evaluations
    Setting<int> number_of_fock_evaluations_{
        settings_, "number_of_fock_evaluations",
        "Fock-evaluation counter (reset on initialize_with_*)", 0, false};

    /// Number of skeleton dimensions of the most recent ODA call (the
    /// N_par variable inside optimal_damping_step). Read by the run()
    /// state machine to size the orbital-rotation burst when orbital_rotation_steps_after_oda_ is
    /// left at its default.
    Setting<int> last_polytope_dimension_{
        settings_, "last_polytope_dimension",
        "ODA polytope dimension of the most recent optimal_damping_step",
        0,
        false};

    /// Number of orbital-rotation DOFs in degenerate groups at the
    /// iterate that emerged from the most recent ODA call:
    ///
    ///     sum_p sum_b sum_g  #pairs(i,j) in g with n_i != n_j
    ///
    /// where g runs over the maximal clusters of orbitals within block
    /// b of particle p that have orbital energies within
    /// optimal_damping_degeneracy_threshold_ of each other. This is
    /// the rotation-DOF count that the diagonal Hessian preconditioner
    /// handles poorly; preconditioned CG needs at most this many steps
    /// to relax the active subspace at fixed occupations (for a
    /// quadratic functional, exactly this many; for DFT, generally
    /// fewer with the cubic-Hermite line search). Used as the default
    /// orbital-rotation burst length when orbital_rotation_steps_after_oda_ is left at zero.
    Setting<int> last_active_rotation_count_{
        settings_, "last_active_rotation_count",
        "active rotations counted by the most recent ODA step", 0, false};

    /// Whether the SCF is converged, exposed through the string facade
    /// as 0 or 1. Carries no stored value: reading it re-evaluates
    /// ``converged()`` against the current history, so it reflects the
    /// solver's state now rather than whatever it was when the flag was
    /// last written.
    Setting<int> converged_{
        settings_, "converged",
        "0 or 1 -- re-evaluates the convergence rule now",
        0,
        false,
        nullptr,
        &SCFSolver::converged_as_int_};

    /// Occupation-space diagnostics. Like ``converged`` they carry no
    /// stored value and re-measure the current iterate on every read,
    /// so a caller can ask what the occupations look like at any point
    /// rather than only at the end. See ``particle_number_error`` and
    /// ``aufbau_error`` for what each measures and why only one of
    /// them gates convergence.
    Setting<Tbase> particle_number_error_{
        settings_, "particle_number_error",
        "largest |sum(n) - N| over the particle types -- re-measured now",
        Tbase(0),
        false,
        nullptr,
        &SCFSolver::particle_number_error};

    Setting<Tbase> aufbau_error_{
        settings_, "aufbau_error",
        "largest occupation outside the Fermi-level window -- re-measured now",
        Tbase(0),
        false,
        nullptr,
        &SCFSolver::aufbau_error};

    Setting<Tbase> fermi_level_error_{
        settings_, "fermi_level_error",
        "spread of orbital energies over the fractionally occupied"
        " orbitals -- re-measured now",
        Tbase(0),
        false,
        nullptr,
        &SCFSolver::fermi_level_error};

    /// Every setting, in declaration order. Supplied by the registry
    /// the settings added themselves to as they were constructed.
    std::vector<SettingBase *> all_settings_() {
      return settings_.settings();
    }

    std::vector<const SettingBase *> all_settings_() const {
      return settings_.settings();
    }


    /* Internal data section */
    /// The number of blocks
    size_t number_of_blocks_;
    /// The orbital history used for convergence acceleration
    OrbitalHistory<Torb, Tbase> orbital_history_;
    /// Orbital energies, updated each iteration from the lowest-energy solution
    OrbitalOccupations<Tbase> orbital_occupations_;

    /// Monotonically increasing stamp handed to each new history
    /// entry by ``make_history_entry``. Per solver, never reused, and
    /// relied upon by the DIIS caches below as a unique key.
    mutable size_t next_history_index_ = 0;

    /// Cache: AO-basis DIIS commutator ``FP - PF`` per history entry,
    /// keyed by the entry's stable iteration index (returned by
    /// ``get_index()``) and populated lazily. Dot products of these
    /// commutators are invariant under the ``C^dagger ... C`` projection
    /// used by ``diis_residual`` (full natural-orbital basis is
    /// unitary), so the same cache serves ``diis_error_matrix_element``.
    /// Pruned by ``prune_diis_caches_()`` whenever a history entry is
    /// dropped, and emptied by ``clear_diis_caches_()`` on a full
    /// history reset. The pruning is not optional: each retained
    /// index costs ``number_of_blocks_`` dense n_basis x n_basis
    /// matrices, and because the keys are monotone the cache would
    /// otherwise grow with the total number of Fock builds in the
    /// run rather than with the history depth.
    mutable std::map<size_t, std::vector<Matrix<Torb>>> diis_commutator_cache_;

    /// Cache: sum-over-blocks ``tr(D_a * F_b)``, keyed by the two
    /// entries' stable iteration indices. Every ADIIS / EDIIS matrix
    /// element is a linear combination of at most four of these
    /// primitives, so caching them collapses the ``nhist^2`` per-block
    /// rebuild of the linear/quadratic-term matrices into ``O(nhist)``
    /// new fills per iteration.
    mutable std::map<std::pair<size_t, size_t>, Tbase> trace_DF_cache_;
    /// Cache: DIIS error matrix element ``dot(e_i, e_j) = -Re(tr(X_i X_j))``,
    /// keyed by the two entries' stable iteration indices in
    /// ``(min, max)`` order (the matrix is symmetric).
    mutable std::map<std::pair<size_t, size_t>, Tbase> diis_matrix_cache_;
    /// Cache: sum-over-blocks Frobenius-norm-vectorised distance
    /// between the density matrices of two history entries, keyed by
    /// their stable iteration indices in ``(min, max)`` order (also
    /// symmetric). Populated on demand by
    /// ``density_matrix_difference``.
    mutable std::map<std::pair<size_t, size_t>, Tbase> density_diff_cache_;

    /// Set by ``optimal_damping_step`` when the collapsed skeleton
    /// pass accepted the step; cleared when the fallback full pass
    /// runs. ``run()``'s convergence-time full-polytope check
    /// consults this to skip verifying already-full ODA steps.
    bool last_oda_via_collapsed_ = false;

    /// Internal holder for computing deltaE
    Tbase old_energy_ = Tbase(0);

    /// Polak-Ribière conjugate-gradient state retained between
    /// scaled_steepest_descent_step calls so that we can layer CG on
    /// top of the preconditioned steepest-descent step. Reset whenever
    /// the orbital basis changes globally (e.g. after a successful ODA
    /// step) or whenever the rotation degrees of freedom change.
    Vector<Tbase> previous_orbital_gradient_;
    Vector<Tbase> previous_orbital_direction_;
    std::vector<OrbitalRotation> previous_orbital_dofs_;

    /// Limited-memory BFGS state retained between lbfgs_step calls.
    /// ``s``, ``y``, ``rho`` store the last maximum_history_length_
    /// triples that drive the two-loop recursion (same cap as DIIS --
    /// the two share one history-depth knob). ``pending_s``, ``pending_g``
    /// hold the (s, g) recorded at the previous accepted step so the
    /// y = g_new - g_old pair can be formed on entry to the next call.
    /// All members are cleared whenever the orbital basis changes
    /// globally (ODA accept), the line search fails, or the DOF set
    /// changes. An empty state costs six empty container headers and
    /// no heap allocation, so it is held by value: a pointer would buy
    /// nothing but a null check at every use and the risk of a
    /// dangling reference when the state is reset mid-step.
    struct LBFGSState {
      std::deque<Vector<Tbase>> s;
      std::deque<Vector<Tbase>> y;
      std::deque<Tbase> rho;
      Vector<Tbase> pending_s;
      Vector<Tbase> pending_g;
      std::vector<OrbitalRotation> history_dofs;
    };
    LBFGSState lbfgs_;

    /// Method-mix flags parsed from methods_. Shared by run() and the
    /// validator in set("methods", ...).
    /// The ODA skeleton set: for each particle type, a list of trial
    /// occupation vectors, one entry per block of that particle. Each
    /// trial is an extreme integer filling of the degenerate orbital
    /// groups at the Fock matrix it was enumerated from, and the
    /// polytope ODA searches is their convex hull.
    using SkeletonOccupations =
        std::vector<std::vector<std::vector<Vector<Tbase>>>>;

    struct AllowedMethods {
      bool diis = false, oda = false, cg = false, lbfgs = false, arh = false;
      /// LCIIS is a variant of the extrapolation step, not a step of
      /// its own: the "LCIIS" token sets ``diis`` as well, so every
      /// existing ``allowed.diis`` gate keeps working and this flag
      /// only selects which coefficients the extrapolation uses.
      bool lciis = false;
      bool orbital_rotation() const { return cg || lbfgs || arh; }
      bool any() const { return diis || oda || orbital_rotation(); }
    };

    /// Parse a method-mix string ("DIIS + ODA + CG") into flags.
    /// Case-insensitive. Throws std::logic_error on unknown tokens,
    /// an empty parse, or a request for two methods that occupy the
    /// same slot (DIIS with LCIIS, or CG with LBFGS).
    static AllowedMethods parse_method_string(const std::string & methods) {
      AllowedMethods allowed;
      std::string s = methods;
      std::transform(s.begin(), s.end(), s.begin(),
                     [](unsigned char c){ return std::tolower(c); });
      std::istringstream iss(s);
      std::string token;
      // Tracked separately from allowed.diis: the LCIIS token also
      // enables the DIIS step, so the two have to be distinguishable
      // afterwards to catch a request for both.
      bool diis_token = false, lciis_token = false;
      while(std::getline(iss, token, '+')) {
        while(!token.empty() && std::isspace((unsigned char)token.front()))
          token.erase(token.begin());
        while(!token.empty() && std::isspace((unsigned char)token.back()))
          token.pop_back();
        if(token.empty()) continue;
        if(token == "diis") diis_token = true;
        else if(token == "lciis") lciis_token = true;
        else if(token == "oda") allowed.oda = true;
        else if(token == "cg") allowed.cg = true;
        else if(token == "lbfgs") allowed.lbfgs = true;
        else if(token == "arh") allowed.arh = true;
        else throw std::logic_error("Unknown method '" + token
            + "' in methods string '" + methods
            + "' (allowed: DIIS, LCIIS, ODA, CG, LBFGS, ARH)");
      }
      // LCIIS is not a step of its own: it substitutes its quartic
      // solve for Pulay's CDIIS solve inside the one extrapolation
      // step. Asking for both would therefore silently discard the
      // DIIS request, so refuse it instead.
      if(diis_token && lciis_token)
        throw std::logic_error("Both DIIS and LCIIS requested in methods"
            " string '" + methods + "': LCIIS replaces the CDIIS"
            " coefficients within the same extrapolation step rather than"
            " adding a step, so the combination is ambiguous. Ask for"
            " exactly one of them.");
      allowed.lciis = lciis_token;
      allowed.diis = diis_token || lciis_token;

      // Same story one level down: CG, L-BFGS and ARH are three ways
      // to take the one orbital-rotation step, and the step picks a
      // single one, so listing two would silently drop whichever lost.
      {
        std::vector<std::string> rotation_tokens;
        if(allowed.cg) rotation_tokens.push_back("CG");
        if(allowed.lbfgs) rotation_tokens.push_back("LBFGS");
        if(allowed.arh) rotation_tokens.push_back("ARH");
        if(rotation_tokens.size() > 1) {
          std::string names = rotation_tokens[0];
          for(size_t i = 1; i + 1 < rotation_tokens.size(); i++)
            names += ", " + rotation_tokens[i];
          names += " and " + rotation_tokens.back();
          throw std::logic_error(names + " all requested in methods"
              " string '" + methods + "': they are alternative"
              " implementations of the one orbital-rotation step, not"
              " steps that can both run, so the combination is ambiguous."
              " Ask for exactly one of them.");
        }
      }

      if(!allowed.any())
        throw std::logic_error("No methods enabled in '" + methods + "'");
      return allowed;
    }

    /// Uppercase a copy of s (used to canonicalise the stored
    /// methods_ string).
    static std::string to_upper_copy(const std::string & s) {
      std::string out = s;
      std::transform(out.begin(), out.end(), out.begin(),
                     [](unsigned char c){ return std::toupper(c); });
      return out;
    }

    /* Internal functions */
    /// Log a printf-formatted message at the given verbosity ``level``.
    /// Drops the message when ``verbosity_ < level``. When a caller
    /// has registered a logger via ``logger()``, the finished string
    /// (with its embedded newlines) is delivered to that callback.
    /// Otherwise the message goes to stdout, matching the pre-callback
    /// default. Marked ``__attribute__((format(printf, ...)))`` so GCC
    /// / Clang validate every call site.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    void log_(int level, const char * fmt, ...) const {
      if(verbosity_ < level) return;
      va_list args1;
      va_start(args1, fmt);
      va_list args2;
      va_copy(args2, args1);
      int n = std::vsnprintf(nullptr, 0, fmt, args1);
      va_end(args1);
      if(n < 0) { va_end(args2); return; }
      std::string msg((size_t) n + 1, '\0');
      std::vsnprintf(msg.data(), (size_t) n + 1, fmt, args2);
      va_end(args2);
      msg.resize((size_t) n);  // trim trailing NUL
      if(logger_) logger_(level, msg);
      else       std::fputs(msg.c_str(), stdout);
    }

    /// Force any pending log output to become visible. When no logger
    /// is installed this flushes stdout (matching the historical
    /// ``fflush(stdout)`` calls scattered through the file); when a
    /// caller-supplied sink is installed, buffering is that sink's
    /// concern so the call is a no-op.
    void log_flush_() const {
      if(!logger_) std::fflush(stdout);
    }

    /// Ostream-style companion to log_(). Constructed via
    /// ``log_stream_(level)``; accumulates output through
    /// ``operator<<`` and flushes to the logger / stdout on
    /// destruction. If ``verbosity_ < level`` the object is inert
    /// (its ``operator<<`` is a no-op), so the caller writes
    /// ``log_stream_(N) << ...`` in place of
    /// ``if(verbosity_>=N) std::cout << ...`` without a manual gate.
    class LogStream {
    public:
      LogStream(const SCFSolver * s, int level)
        : solver_(s), level_(level), enabled_(s && s->verbosity_ >= level) {}
      LogStream(const LogStream &) = delete;
      LogStream & operator=(const LogStream &) = delete;
      LogStream(LogStream && o) noexcept
        : solver_(o.solver_), level_(o.level_),
          enabled_(o.enabled_), oss_(std::move(o.oss_)) {
        o.enabled_ = false;  // moved-from doesn't flush
      }
      ~LogStream() {
        if(!enabled_) return;
        std::string s = oss_.str();
        if(solver_->logger_) solver_->logger_(level_, s);
        else                 std::fputs(s.c_str(), stdout);
      }
      template <class T>
      LogStream & operator<<(const T & v) {
        if(enabled_) oss_ << v;
        return *this;
      }
      // Handle io manipulators such as std::endl / std::flush.
      LogStream & operator<<(std::ostream & (*manip)(std::ostream &)) {
        if(enabled_) oss_ << manip;
        return *this;
      }
      /// True iff the log gate opened and writes will be flushed.
      /// Use to short-circuit expensive formatting when the sink
      /// would discard it anyway.
      bool enabled() const { return enabled_; }
      /// Direct handle on the underlying std::ostream buffer, for
      /// helpers that write via a std::ostream& argument (e.g.
      /// print_settings). Writes when the gate is closed still land
      /// in oss_ but are dropped when the LogStream destructs.
      std::ostream & stream() { return oss_; }
    private:
      const SCFSolver * solver_;
      int level_;
      bool enabled_;
      std::ostringstream oss_;
    };

    /// Return an ostream-style log proxy at verbosity ``level``.
    /// Cheap when the gate rejects (no allocation, no format).
    LogStream log_stream_(int level) const {
      return LogStream(this, level);
    }

    /// Is the block empty?
    bool empty_block(size_t iblock) const {
      // Check if Fock matrix has zero dimension
      if(iblock>=std::get<0>(orbital_history_[0]).first.size())
        throw std::logic_error("Trying to check empty block for nonexistent index!\n");
      return std::get<1>(orbital_history_[0]).second[iblock].size() == 0;
    }

    /// Materialise every block of a density matrix from its
    /// natural-orbital / occupation representation. Used to hoist the
    /// C * diag(n) * C^dagger builds out of the O(npars^2) ODA
    /// Hessian loop so each vertex density is constructed once;
    /// combined with ``tr_of_product_`` this collapses the axis-
    /// Hessian construction from O(npars^2 * N^3) to
    /// O(npars * N^3 + npars^2 * N^2).
    std::vector<Matrix<Torb>> build_density_blocks_(
        const DensityMatrix<Torb,Tbase> & dm) const {
      const auto & orbitals = dm.first;
      const auto & occupations = dm.second;
      std::vector<Matrix<Torb>> blocks(orbitals.size());
      for(size_t iblock=0; iblock<orbitals.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        blocks[iblock] = build_density_block_(
            orbitals[iblock], occupations[iblock],
            maximum_occupation_(iblock));
      }
      return blocks;
    }

    /// Get a block of the density matrix for the ihist:th entry
    Matrix<Torb> get_density_matrix_block(size_t ihist, size_t iblock) const {
      return build_density_block_(
          get_orbital_block(ihist, iblock),
          get_orbital_occupation_block(ihist, iblock),
          maximum_occupation_(iblock));
    }

    /// Trace of ``A * B``, evaluated as
    ///   sum_{ij} A(i,j) * B(j,i) = (A .* B.transpose()).sum()
    /// -- O(N^2) instead of the O(N^3) matmul-then-trace form.
    /// The real part is returned; callers rely on the trace being
    /// real (both A and B are Hermitian in every current use site).
    Tbase tr_of_product_(const Matrix<Torb> & A,
                         const Matrix<Torb> & B) const {
      return std::real((A.array() * B.transpose().array()).sum());
    }

    /// Extract the natural orbitals carrying a non-negligible
    /// occupation: the columns of ``C`` whose occupation exceeds
    /// ``10 * max_occ * epsilon`` in magnitude, together with those
    /// occupations. Restricting the rank-k density and commutator
    /// builds to these ``k`` columns is what makes them
    /// O(k * n_basis^2) instead of O(n_basis^3), which is the whole
    /// point whenever ``k << n_basis`` -- atoms, small molecules,
    /// ODA polytope vertices, brute-force search trial occupations.
    ///
    /// The outputs are resized to match; ``n_active.size() == 0``
    /// means the block carries no density at all, which callers must
    /// special-case since the rank-k products are then empty.
    void active_natural_orbitals_(
        const OrbitalBlock<Torb> & C,
        const OrbitalBlockOccupations<Tbase> & n,
        Tbase max_occ,
        Matrix<Torb> & C_active,
        Vector<Tbase> & n_active) const {
      const Tbase tol =
          Tbase(10) * max_occ * std::numeric_limits<Tbase>::epsilon();
      Index count = 0;
      for(Index k = 0; k < n.size(); k++)
        if(std::abs(n(k)) > tol) ++count;
      C_active.resize(C.rows(), count);
      n_active.resize(count);
      Index col = 0;
      for(Index k = 0; k < n.size(); k++) {
        if(std::abs(n(k)) > tol) {
          C_active.col(col) = C.col(k);
          n_active(col) = n(k);
          ++col;
        }
      }
    }

    /// Build a density-matrix block C * diag(n) * C^dagger from the
    /// natural-orbital / occupation representation, dropping natural
    /// orbitals with a zero occupation before the outer product. The
    /// restricted product is O(k * n_basis^2) with ``k`` the number
    /// of populated natural orbitals, versus O(n_basis^3) for the
    /// unrestricted form -- worth doing whenever ``k << n_basis``
    /// (atoms and small molecules, ODA polytope vertices, brute-
    /// force search trial occupations, ...).
    Matrix<Torb> build_density_block_(
        const OrbitalBlock<Torb> & C,
        const OrbitalBlockOccupations<Tbase> & n,
        Tbase max_occ) const {
      Matrix<Torb> C_active;
      Vector<Tbase> n_active;
      active_natural_orbitals_(C, n, max_occ, C_active, n_active);
      if(n_active.size() == 0)
        return Matrix<Torb>::Zero(C.rows(), C.rows());
      return C_active * n_active.asDiagonal() * C_active.adjoint();
    }

    /// Get a block of the orbital occupations for the ihist:th entry
    OrbitalBlock<Torb> get_orbital_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbitals for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbitals for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).first[iblock];
    }

    /// Get a block of the orbital occupations for the ihist:th entry
    OrbitalBlockOccupations<Tbase> get_orbital_occupation_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).second[iblock];
    }

    /// Get lowest energy after the given reference index
    Tbase get_lowest_energy_after_index(size_t index=0) const {
      bool initialized = false;
      Tbase lowest_energy;
      for(size_t i=0;i<orbital_history_.size();i++) {
        if(get_index(i) > index) {
          if(not initialized) {
            initialized=true;
            lowest_energy = get_energy(i);
          } else {
            lowest_energy = std::min(lowest_energy, get_energy(i));
          }
        }
      }
      if(initialized)
        return lowest_energy;
      else {
        print_history();
        log_flush_();
        std::ostringstream oss;
        oss << "Did not find any entries with index greater than " << index << "!\n";
        throw std::logic_error(oss.str());
      }
    }

    /// Get the energy for the entry
    size_t get_index(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access index for nonexistent history member!\n");
      return std::get<2>(orbital_history_[ihist]);
    }

    /// Get largest index
    size_t largest_index() const {
      size_t index = get_index(0);
      for(size_t i=1;i<orbital_history_.size();i++) {
        index = std::max(index, get_index(i));
      }
      return index;
    }

    /// Matrix dimensions
    IndexVector matrix_dimension() const {
      const auto & fock = std::get<1>(orbital_history_[0]).second;
      IndexVector dim(fock.size());
      for(size_t i=0;i<fock.size();i++)
        dim(i) = fock[i].cols();
      return dim;
    }

    /// Get a block of the Fock matrix for the ihist:th entry
    Matrix<Torb> get_fock_matrix_block(size_t ihist, size_t iblock) const {
      return std::get<1>(orbital_history_[ihist]).second[iblock];
    }

    /// Vectorise
    Vector<Tbase> vectorise(const Matrix<Torb> & mat) const {
      return vectorise_real_imag(mat);
    }

    /// Vectorise
    Vector<Tbase> vectorise(const std::vector<Matrix<Torb>> & mat) const {
      std::vector<Vector<Tbase>> vectors(mat.size());
      for(size_t iblock=0;iblock<mat.size();iblock++) {
        if(mat[iblock].size()==0)
          continue;
        vectors[iblock]=vectorise(mat[iblock]);
      }
      // join_columns skips zero-length parts, so empty blocks need no
      // special-casing in the concatenation.
      return join_columns(vectors);
    }

    Matrix<Torb> matricise(const Vector<Tbase> & vec, size_t nrows, size_t ncols) const {
      if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
        if(vec.size() != (Index)(nrows*ncols)) {
          std::ostringstream oss;
          oss << "Matricise error: expected " << nrows*ncols << " elements for " << nrows << " x " << ncols << " real matrix, but got " << vec.size() << " instead!\n";
          throw std::logic_error(oss.str());
        }
        return Eigen::Map<const Matrix<Torb>>(vec.data(), nrows, ncols);
      } else {
        if(vec.size() != (Index)(2*nrows*ncols)) {
          std::ostringstream oss;
          oss << "Matricise error: expected " << 2*nrows*ncols << " elements for " << nrows << " x " << ncols << " complex matrix, but got " << vec.size() << " instead!\n";
          throw std::logic_error(oss.str());
        }

        Eigen::Map<const Matrix<Tbase>> real_part(vec.data(), nrows, ncols);
        Eigen::Map<const Matrix<Tbase>> imag_part(vec.data()+nrows*ncols, nrows, ncols);
        Matrix<Torb> mat = real_part.template cast<Torb>()
                        + imag_part.template cast<Torb>() * std::complex<Tbase>(Tbase{0}, Tbase{1});
        return mat;
      }
    }

    std::vector<Matrix<Torb>> matricise(const Vector<Tbase> & vec, const IndexVector & dim) const {
      std::vector<Matrix<Torb>> mat(dim.size());
      size_t ioff = 0;
      for(Index iblock=0; iblock<dim.size(); iblock++) {
        if(dim(iblock)==0)
          continue;
        size_t sz = (size_t)dim(iblock) * (size_t)dim(iblock);
        if constexpr (Eigen::NumTraits<Torb>::IsComplex) {
          sz *= 2;
        }
        mat[iblock] = matricise(vec.segment(ioff, sz), dim(iblock), dim(iblock));
        ioff += sz;
      }
      return mat;
    }

    /// Empty every history-derived cache. Called on any operation
    /// that discards or replaces history entries wholesale
    /// (initialize_with_*, reset_history).
    void clear_diis_caches_() const {
      diis_commutator_cache_.clear();
      trace_DF_cache_.clear();
      diis_matrix_cache_.clear();
      density_diff_cache_.clear();
    }

    /// Drop cache entries belonging to history entries that no longer
    /// exist. Safe and final: the keys are the monotone per-solver
    /// history indices, so an index that is not currently live can
    /// never become live again.
    ///
    /// This must run after every operation that removes a history
    /// entry (``add_entry``'s length-capping pop, ``cleanup``'s
    /// density-difference erase). Without it the caches are only ever
    /// emptied by ``initialize_with_*`` / ``reset_history``, neither
    /// of which fires during an SCF, so ``diis_commutator_cache_``
    /// would grow with the total number of Fock builds in the run
    /// rather than with the history depth -- it holds
    /// ``number_of_blocks_`` dense n_basis x n_basis matrices per
    /// retained index, so a long run in a large basis leaks
    /// gigabytes.
    void prune_diis_caches_() const {
      std::vector<size_t> live;
      live.reserve(orbital_history_.size());
      for(size_t i = 0; i < orbital_history_.size(); i++)
        live.push_back(get_index(i));
      std::sort(live.begin(), live.end());
      auto is_live = [&live](size_t idx) {
        return std::binary_search(live.begin(), live.end(), idx);
      };
      for(auto it = diis_commutator_cache_.begin(); it != diis_commutator_cache_.end(); )
        it = is_live(it->first) ? std::next(it) : diis_commutator_cache_.erase(it);
      auto prune_pair_keyed = [&is_live](auto & cache) {
        for(auto it = cache.begin(); it != cache.end(); )
          it = (is_live(it->first.first) && is_live(it->first.second))
                 ? std::next(it) : cache.erase(it);
      };
      prune_pair_keyed(trace_DF_cache_);
      prune_pair_keyed(diis_matrix_cache_);
      prune_pair_keyed(density_diff_cache_);
    }

    /// Ordered index pair for symmetric caches, so lookups agree
    /// regardless of the caller's (i, j) ordering.
    static std::pair<size_t, size_t> sorted_pair_(size_t a, size_t b) {
      return a <= b ? std::make_pair(a, b) : std::make_pair(b, a);
    }

    /// AO-basis commutator ``[F_a, D_b] = F_a D_b - D_b F_a`` for one
    /// block, taking the Fock matrix from history entry ``ihist_fock``
    /// and the density from ``ihist_density``. Built via the rank-k
    /// route
    ///   FP = F * (C_occ * diag(n_occ) * C_occ^dagger)
    ///   PF = (FP)^dagger      (F, D Hermitian)
    /// so the build is ``O(k * n_basis^2)`` in place of the naive
    /// ``O(n_basis^3)`` two-matmul form.
    ///
    /// DIIS only ever needs the diagonal ``a == b``; LCIIS needs the
    /// full ``M x M`` grid of mixed commutators, which is why this
    /// takes two independent entry indices.
    Matrix<Torb> commutator_block_(size_t ihist_fock, size_t ihist_density,
                                   size_t iblock) const {
      auto F = get_fock_matrix_block(ihist_fock, iblock);
      Matrix<Torb> F_sym = (Tbase(1)/Tbase(2)) * (F + F.adjoint());

      Matrix<Torb> C_active;
      Vector<Tbase> n_active;
      active_natural_orbitals_(get_orbital_block(ihist_density, iblock),
                               get_orbital_occupation_block(ihist_density, iblock),
                               maximum_occupation_(iblock),
                               C_active, n_active);
      if(n_active.size() == 0)
        return Matrix<Torb>::Zero(F_sym.rows(), F_sym.cols());

      Matrix<Torb> FC = F_sym * C_active;                  // O(N^2 * k)
      Matrix<Torb> FP = FC * n_active.asDiagonal()
                     * C_active.adjoint();                 // O(N^2 * k)
      // PF = (FP)^dagger since F and D are Hermitian.
      return FP - FP.adjoint();
    }

    /// The same-entry commutator ``[F_i, D_i]`` -- the DIIS error
    /// matrix -- cached by the entry's stable iteration index, so
    /// subsequent DIIS-matrix accesses at the same entry cost nothing.
    const Matrix<Torb> &
    diis_commutator_cached_(size_t ihist, size_t iblock) const {
      const size_t idx = get_index(ihist);
      auto & blocks = diis_commutator_cache_[idx];
      if(blocks.empty())
        blocks.resize(number_of_blocks_);
      if(blocks[iblock].size() != 0)
        return blocks[iblock];
      blocks[iblock] = commutator_block_(ihist, ihist, iblock);
      return blocks[iblock];
    }

    /// Sum over blocks of ``tr(D_a * F_b)`` for the given history
    /// entries, cached by (idx_a, idx_b).
    Tbase trace_DF_cached_(size_t ihist_a, size_t ihist_b) const {
      const size_t idx_a = get_index(ihist_a);
      const size_t idx_b = get_index(ihist_b);
      const auto key = std::make_pair(idx_a, idx_b);
      auto it = trace_DF_cache_.find(key);
      if(it != trace_DF_cache_.end()) return it->second;
      Tbase tr = Tbase(0);
      for(size_t iblock = 0; iblock < number_of_blocks_; iblock++) {
        if(empty_block(iblock)) continue;
        tr += tr_of_product_(
            get_density_matrix_block(ihist_a, iblock),
            get_fock_matrix_block(ihist_b, iblock));
      }
      trace_DF_cache_[key] = tr;
      return tr;
    }

    /// Compute DIIS residual
    Matrix<Torb> diis_residual(size_t ihist, size_t iblock) const {
      // FPS - SPF (S = I) commutator lives on the cache with the
      // ``FP - PF`` sign convention. Project into the current
      // reference natural orbitals so the L^infty norm stays
      // basis-independent, matching the pre-cache semantics.
      const Matrix<Torb> & X = diis_commutator_cached_(ihist, iblock);
      auto C = get_orbital_block(0, iblock);
      return C.adjoint() * X * C;
    }

    /// Compute DIIS residual
    std::vector<Matrix<Torb>> diis_residual(size_t ihist) const {
      std::vector<Matrix<Torb>> residuals(number_of_blocks_);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        residuals[iblock] = diis_residual(ihist, iblock);
      }
      return residuals;
    }

    /// Form DIIS error vector for ihist:th entry
    Vector<Tbase> diis_error_vector(size_t ihist, size_t iblock) const {
      return vectorise(diis_residual(ihist, iblock));
    }

    /// Form DIIS error vector for ihist:th entry
    Vector<Tbase> diis_error_vector(size_t ihist) const {
      // Form error vectors
      std::vector<Vector<Tbase>> error_vectors(number_of_blocks_);
      for(size_t iblock = 0; iblock<number_of_blocks_;iblock++) {
        error_vectors[iblock] = diis_error_vector(ihist, iblock);
        log_(20, "ihist %i block %i error vector norm %e\n", (int) ihist, (int) iblock, (double) (norm(error_vectors[iblock])));
        log_stream_(30) << error_vectors[iblock] << std::endl;
      }

      // Compound error vector. join_columns skips zero-length parts
      // and gets the offset bookkeeping right by construction, so the
      // hand-rolled offset loop and its "Indexing error!" assertion
      // are no longer needed.
      return join_columns(error_vectors);
    }

    /// Estimate the roundoff noise floor of the DIIS error vector
    /// from the current Fock. Each entry of the projected commutator
    /// C^dagger [F_sym, P_sym] C carries a per-element roundoff bound
    /// of order eps * ||F||_F (C is unitary, ||P|| <= 1). Assemble a
    /// mock error vector at that bound and reduce with the active
    /// error norm so the returned value is directly comparable to
    /// diis_error_norm().
    Tbase compute_noise_floor() const {
      const Tbase eps = std::numeric_limits<Tbase>::epsilon();
      std::vector<Matrix<Torb>> mock(number_of_blocks_);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        auto F = get_fock_matrix_block(0, iblock);
        Index n = F.rows();
        Tbase per_elem = eps * F.norm();
        Torb seed;
        if constexpr (Eigen::NumTraits<Torb>::IsComplex)
          // Load both real and imag halves of vectorise_real_imag so
          // the clamp is not looser for complex than for real.
          seed = Torb(per_elem, per_elem);
        else
          seed = Torb(per_elem);
        mock[iblock] = Matrix<Torb>::Constant(n, n, seed);
      }
      return norm(vectorise(mock));
    }

    /// The convergence threshold actually in force: the user's
    /// ``convergence_threshold_``, raised to
    /// ``noise_safety_factor_ * noise_floor_`` whenever the
    /// arithmetic noise floor sits above it, so the SCF never chases
    /// a threshold below what the working precision can resolve.
    Tbase effective_convergence_threshold_() const {
      return std::max<Tbase>(convergence_threshold_,
                             noise_safety_factor_ * noise_floor_);
    }

    /// Smallest energy descent that can still change the outer SCF's
    /// stopping decision: one tenth of the effective convergence
    /// threshold. Steps that squeeze out less than this are rejected
    /// as noise rather than accepted as progress.
    Tbase minimum_useful_descent_() const {
      return effective_convergence_threshold_() / Tbase(10);
    }

    /// Compute element of DIIS error matrix
    Tbase diis_error_matrix_element(size_t ihist, size_t jhist) const {
      // dot(e_i, e_j) = Re(tr((C^dagger X_i C)^dagger (C^dagger X_j C))).
      // C is unitary (full natural-orbital basis) so this reduces to
      //   Re(tr(X_i^dagger X_j)) = -Re(tr(X_i X_j))
      // using the anti-Hermitian property X^dagger = -X. Both factors
      // and the resulting scalar are cached by the entries' stable
      // iteration indices; a matrix element only re-touches the
      // commutator cache when it hasn't been seen before.
      const auto key = sorted_pair_(get_index(ihist), get_index(jhist));
      auto it = diis_matrix_cache_.find(key);
      if(it != diis_matrix_cache_.end()) return it->second;
      Tbase el=Tbase(0);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        const Matrix<Torb> & Xi = diis_commutator_cached_(ihist, iblock);
        const Matrix<Torb> & Xj = diis_commutator_cached_(jhist, iblock);
        el -= tr_of_product_(Xi, Xj);
      }
      diis_matrix_cache_[key] = el;
      return el;
    }

    /// Form DIIS error matrix
    Matrix<Tbase> diis_error_matrix(const std::vector<size_t> & mask) const {
      // The error matrix is given by the orbital gradient dot products
      const size_t N=mask.size();
      Matrix<Tbase> B = Matrix<Tbase>::Zero(N,N);

      for(size_t ihist=0; ihist<N; ihist++) {
        for(size_t jhist=0; jhist<=ihist; jhist++) {
          B(ihist, jhist) = B(jhist, ihist) = diis_error_matrix_element(mask[ihist], mask[jhist]);
        }
      }
      return B;
    }

    Vector<Tbase> diis_error_matrix_diagonal() const {
      Vector<Tbase> B = Vector<Tbase>::Zero(orbital_history_.size());
      for(Index ihist=0; ihist<B.size(); ihist++) {
        B(ihist) = diis_error_matrix_element(ihist, ihist);
      }
      return B;
    }

    Matrix<Tbase> diis_error_matrix() const {
      std::vector<size_t> mask(orbital_history_.size());
      for(size_t i=0;i<mask.size();i++)
        mask[i]=i;
      return diis_error_matrix(mask);
    }

    /// Calculate DIIS weights
    Vector<Tbase> diis_weights() const {
      // Only use reference points with error residuals that are sufficiently small
      std::vector<size_t> history_mask(orbital_history_.size());
      for(size_t i=0;i<history_mask.size();i++)
        history_mask[i]=i;
      Vector<Tbase> residuals(history_mask.size());
      for(Index i=0;i<residuals.size();i++)
        residuals(i) = diis_error_matrix_element(history_mask[i], history_mask[i]);
      Tbase min_residual = residuals.minCoeff();
      for(size_t i=history_mask.size()-1;i<history_mask.size();i--)
        // Criterion from Chupin et al, 2012
        if(residuals(i)*diis_restart_factor_ > min_residual)
          history_mask.erase(history_mask.begin()+i);
      size_t nrestart = orbital_history_.size()-history_mask.size();
      if(nrestart>0)
        log_(10, "Removed %i entries corresponding to large DIIS errors\n", (int) nrestart);

      // Set up the DIIS error matrix
      const size_t N=history_mask.size();
      Matrix<Tbase> B = Matrix<Tbase>::Constant(N+1, N+1, Tbase(-Tbase(1)));
      B.block(0, 0, N, N) = diis_error_matrix(history_mask);
      B(N,N)=Tbase(0);

      // Apply the diagonal damping
      B.block(0, 0, N, N).diagonal() *= Tbase(1)+diis_diagonal_damping_;

      // To improve numerical conditioning, scale entries of error
      // matrix such that the last diagonal element is one; Eckert et
      // al, J. Comput. Chem 18. 1473-1483 (1997)
      Vector<Tbase> Bdiag(B.diagonal());
      Tbase diagmin = Bdiag.head(N).minCoeff();
      if(diagmin != Tbase(0))
        B.block(0, 0, N, N) /= diagmin;

      // Right-hand side of equation is
      Vector<Tbase> rh = Vector<Tbase>::Zero(N+1);
      rh(N)=-Tbase(1);

      // Solve the equation
      Vector<Tbase> sol = B.colPivHouseholderQr().solve(rh);
      Vector<Tbase> diis_weights = sol.head(N);

      // Pad to full space
      Vector<Tbase> diis_weights_full = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t i=0;i<history_mask.size();i++)
        diis_weights_full[history_mask[i]] = diis_weights[i];

      return diis_weights_full;
    }
    /// Number of history entries LCIIS extrapolates over: the
    /// requested cap, clamped to what the history actually holds.
    size_t lciis_subspace_size_() const {
      size_t M = orbital_history_.size();
      if(lciis_maximum_history_ > 0)
        M = std::min(M, (size_t) lciis_maximum_history_);
      return M;
    }

    /// Build the fully symmetric LCIIS quartic tensor over the ``M``
    /// leading history entries, returned flattened in row-major
    /// ``((i*M + j)*M + k)*M + l`` order.
    ///
    /// LCIIS minimises the Frobenius norm of the commutator between
    /// the predicted Fock matrix and the predicted density,
    ///   f(c) = || [sum_i c_i F_i, sum_j c_j D_j] ||_F^2
    ///        = sum_ijkl c_i c_j c_k c_l T_ijkl,
    ///   T_ijkl = tr([F_i, D_j]^dagger [F_k, D_l]),
    /// which is Li & Yaron, JCTC 12, 5322 (2016), eqs 4-6. Note this
    /// is genuinely quartic in ``c``: unlike CDIIS, which minimises
    /// ``|| sum_i c_i [F_i, D_i] ||_F^2`` and so only ever touches
    /// the diagonal commutators, LCIIS needs the full M x M grid of
    /// mixed commutators ``[F_i, D_j]``.
    ///
    /// The returned tensor is symmetrised over all 24 index
    /// permutations. Only the fully symmetric part of ``T``
    /// contributes to the quartic form, and symmetrising up front
    /// makes the gradient and Hessian the textbook
    ///   g_i  = 4 sum_jkl S_ijkl c_j c_k c_l
    ///   H_ij = 12 sum_kl  S_ijkl c_k c_l
    /// rather than the six- and three-term index gymnastics the
    /// unsymmetrised form requires. The symmetrisation costs
    /// ``O(M^4)`` on a tensor with ``M <= 20``, i.e. nothing next to
    /// the ``O(M^2 N^3)`` commutator builds.
    std::vector<Tbase> lciis_tensor_(size_t M) const {
      // Mixed commutators A[i*M + j] = [F_i, D_j], per block. This is
      // the memory high-water mark of LCIIS: M^2 blocks of n_basis^2,
      // which is why lciis_maximum_history exists as a separate cap
      // from maximum_history_length.
      std::vector<std::vector<Matrix<Torb>>> A(M * M);
      for(size_t i = 0; i < M; i++)
        for(size_t j = 0; j < M; j++) {
          auto & Aij = A[i * M + j];
          Aij.resize(number_of_blocks_);
          for(size_t iblock = 0; iblock < number_of_blocks_; iblock++) {
            if(empty_block(iblock))
              continue;
            Aij[iblock] = commutator_block_(i, j, iblock);
          }
        }

      // T_ijkl = tr(A_ij^dagger A_kl). The commutators are
      // anti-Hermitian, so A^dagger = -A and the Frobenius inner
      // product is -Re tr(A_ij A_kl) -- the same identity
      // diis_error_matrix_element uses on the diagonal.
      //
      // T_ijkl = T_klij because swapping the pairs conjugates the
      // trace and we keep only the real part, so half the pair
      // combinations come for free.
      const size_t M2 = M * M;
      std::vector<Tbase> T(M2 * M2, Tbase(0));
      for(size_t ij = 0; ij < M2; ij++)
        for(size_t kl = ij; kl < M2; kl++) {
          Tbase el = Tbase(0);
          for(size_t iblock = 0; iblock < number_of_blocks_; iblock++) {
            if(empty_block(iblock))
              continue;
            el -= tr_of_product_(A[ij][iblock], A[kl][iblock]);
          }
          T[ij * M2 + kl] = el;
          T[kl * M2 + ij] = el;
        }

      // Symmetrise over all 24 permutations of (i,j,k,l).
      static const int perm[24][4] = {
        {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
        {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
        {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
        {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}};
      std::vector<Tbase> S(T.size(), Tbase(0));
      size_t idx[4];
      for(idx[0] = 0; idx[0] < M; idx[0]++)
        for(idx[1] = 0; idx[1] < M; idx[1]++)
          for(idx[2] = 0; idx[2] < M; idx[2]++)
            for(idx[3] = 0; idx[3] < M; idx[3]++) {
              Tbase acc = Tbase(0);
              for(const auto & p : perm)
                acc += T[((idx[p[0]] * M + idx[p[1]]) * M
                          + idx[p[2]]) * M + idx[p[3]]];
              S[((idx[0] * M + idx[1]) * M + idx[2]) * M + idx[3]]
                  = acc / Tbase(24);
            }
      return S;
    }

    /// LCIIS extrapolation weights: Li & Yaron, JCTC 12, 5322 (2016).
    ///
    /// Minimises the quartic ``f(c) = sum_ijkl S_ijkl c_i c_j c_k c_l``
    /// -- the squared Frobenius norm of the commutator between the
    /// predicted Fock matrix and the predicted density -- subject to
    /// ``sum_i c_i = 1``, by Newton's method on the Lagrangian
    ///   L(c, lambda) = f(c) - lambda (1 - sum_i c_i).
    /// The Newton step solves the bordered KKT system
    ///   [ H   1 ] [ dc     ]     [ g + lambda 1 ]
    ///   [ 1^T 0 ] [ dlambda ] = -[ sum_i c_i - 1 ]
    /// and is taken in *step* form rather than the paper's equivalent
    /// direct form, so a starting guess that violates the constraint
    /// by roundoff is pulled back onto it instead of being assumed
    /// exact.
    ///
    /// The iteration starts from the CDIIS coefficients, as the paper
    /// recommends. Failure at any point -- a singular KKT matrix, a
    /// non-finite iterate, no convergence within the iteration cap --
    /// falls back to those CDIIS coefficients rather than throwing:
    /// LCIIS is a convergence accelerator, and a bad quartic solve is
    /// a reason to take the ordinary DIIS step, not to abort the SCF.
    Vector<Tbase> lciis_weights() const {
      const Vector<Tbase> cdiis = diis_weights();
      const size_t M = lciis_subspace_size_();
      // With one entry the constraint fixes c entirely.
      if(M < 2)
        return cdiis;

      const std::vector<Tbase> S = lciis_tensor_(M);
      const size_t M2 = M * M;

      // Seed from CDIIS restricted to the leading M entries,
      // renormalised so the constraint holds exactly at the start.
      Vector<Tbase> c = cdiis.head(M);
      const Tbase csum = c.sum();
      if(!(std::abs(csum) > std::numeric_limits<Tbase>::epsilon()))
        return cdiis;
      c /= csum;

      auto quartic_value = [&](const Vector<Tbase> & x) {
        Tbase f = Tbase(0);
        for(size_t i = 0; i < M; i++)
          for(size_t j = 0; j < M; j++)
            for(size_t k = 0; k < M; k++)
              for(size_t l = 0; l < M; l++)
                f += S[((i * M + j) * M + k) * M + l]
                     * x(i) * x(j) * x(k) * x(l);
        return f;
      };

      Tbase lambda = Tbase(0);
      Tbase f_current = quartic_value(c);
      bool converged = false;

      for(int iter = 0; iter < lciis_maximum_iterations_; iter++) {
        // Contract S with c twice to get the Hessian, then once more
        // for the gradient: H_ij = 12 sum_kl S_ijkl c_k c_l and
        // g_i = (1/3) sum_j H_ij c_j, using Euler's identity for the
        // homogeneous quartic (H c = 3 g) so the gradient costs
        // O(M^2) rather than a third O(M^3) contraction.
        Matrix<Tbase> H = Matrix<Tbase>::Zero(M, M);
        for(size_t i = 0; i < M; i++)
          for(size_t j = 0; j < M; j++) {
            Tbase acc = Tbase(0);
            for(size_t k = 0; k < M; k++)
              for(size_t l = 0; l < M; l++)
                acc += S[((i * M + j) * M + k) * M + l] * c(k) * c(l);
            H(i, j) = Tbase(12) * acc;
          }
        const Vector<Tbase> g = (H * c) / Tbase(3);

        // Bordered KKT system.
        Matrix<Tbase> K = Matrix<Tbase>::Zero(M + 1, M + 1);
        K.block(0, 0, M, M) = H;
        K.block(0, M, M, 1) = Vector<Tbase>::Ones(M);
        K.block(M, 0, 1, M) = Vector<Tbase>::Ones(M).transpose();
        Vector<Tbase> rhs(M + 1);
        rhs.head(M) = -(g + lambda * Vector<Tbase>::Ones(M));
        rhs(M) = -(c.sum() - Tbase(1));

        Eigen::ColPivHouseholderQR<Matrix<Tbase>> qr(K);
        if(qr.rank() < (Index) (M + 1))
          break;                       // singular KKT: keep CDIIS
        const Vector<Tbase> step = qr.solve(rhs);
        if(has_nan(step) || has_inf(step))
          break;

        c += step.head(M);
        lambda += step(M);

        const Tbase f_new = quartic_value(c);
        const Tbase step_norm = step.head(M).norm();
        log_(10, "LCIIS iteration %i: f = %e, step norm %e\n",
             (int) iter, (double) f_new, (double) step_norm);
        if(step_norm <= lciis_convergence_threshold_) {
          converged = true;
          f_current = f_new;
          break;
        }
        f_current = f_new;
      }

      if(!converged) {
        log_(5, "LCIIS did not converge in %i iterations; "
                "falling back to CDIIS weights.\n",
             (int) lciis_maximum_iterations_);
        return cdiis;
      }
      if(has_nan(c) || has_inf(c)) {
        log_(5, "LCIIS produced a non-finite solution; "
                "falling back to CDIIS weights.\n");
        return cdiis;
      }
      // A negative minimum is numerically impossible for a squared
      // norm and signals that the quartic model has been corrupted.
      if(f_current < Tbase(0)) {
        log_(5, "LCIIS target function went negative (%e); "
                "falling back to CDIIS weights.\n", (double) f_current);
        return cdiis;
      }

      // Pad back to the full history length: entries beyond the LCIIS
      // subspace get zero weight.
      Vector<Tbase> weights = Vector<Tbase>::Zero(orbital_history_.size());
      weights.head(M) = c;
      return weights;
    }

    /// Calculate ADIIS weights by minimizing quadratic form
    Vector<Tbase> aediis_weights(const Vector<Tbase> & b, const Matrix<Tbase> & A) const {
      if(b.size()==1) {
        // Nothing to optimize
        return Vector<Tbase>::Ones(b.size());
      }

      // Parameters
      const size_t max_iter = 1000000;
      const Tbase df_tol = Tbase(1e-8);

      // Function to evaluate function value
      std::function<Tbase(const Vector<Tbase> & x)> fx = [b, A](const Vector<Tbase> & x) {
        return Tbase(1)/Tbase(2)*(x.transpose()*A*x).value() + b.dot(x);
      };

      // Function to determine optimal step
      std::function<Tbase(const Vector<Tbase> &, const Vector<Tbase> &)> optimal_step = [b, A](const Vector<Tbase> & current_direction, const Vector<Tbase> & x) {
        return -((current_direction.transpose()*A*x).value() + b.dot(current_direction))
               / (current_direction.transpose()*A*current_direction).value();
      };

      /// Make initial guesses for parameters
      std::vector<Vector<Tbase>> xguess;
      // Center point
      xguess.push_back(Vector<Tbase>::Constant(b.size(), Tbase(1)/b.size()));
      // "Gauss" points
      for(Index i=0;i<b.size();i++) {
        Vector<Tbase> xtr = Vector<Tbase>::Constant(b.size(), Tbase(1)/(b.size()+2));
        xtr(i) *= 3;
        xguess.push_back(xtr);
      }
      // End points
      for(Index i=0;i<b.size();i++) {
        Vector<Tbase> xtr = Vector<Tbase>::Zero(b.size());
        xtr(i) = Tbase(1);
        xguess.push_back(xtr);
      }

      // Find minimum
      Vector<Tbase> yguess(xguess.size());
      for(size_t i=0;i<xguess.size();i++)
        yguess[i] = fx(xguess[i]);

      IndexVector idx(sort_index_ascending(yguess));
      Vector<Tbase> x = xguess[idx[0]];

      /// Matrix of search directions
      Matrix<Tbase> search_directions = Matrix<Tbase>::Identity(b.size(), b.size());

      /// Evaluate initial point
      auto current_point = fx(x);
      auto old_x = x;

      // Powell algorithm
      for(size_t imacro=0; imacro<max_iter; imacro++) {
        Tbase curval(current_point);

        for(Index i=0; i<b.size(); i++) {
          Vector<Tbase> c_i(search_directions.col(i));
          // x -> (1-step)*x + step*c_i = x + step*(c_i-x)
          Tbase step = optimal_step(c_i-x, x);
          if(!std::isnormal(step))
            continue;
          //printf("Direction %i: optimal step %e\n",i,step);
          if(step > Tbase(0) and step <= Tbase(1)) {
            auto new_point = fx(x+step*(c_i-x));
            //printf("Direction %i: optimal step changes energy by %e\n",(int) i,new_point.first - current_point.first);
            if(new_point < current_point) {
              x += step*(c_i-x);
              current_point = new_point;
            }
          }
        }

        Tbase dE = current_point - curval;
        //printf("Macroiteration %i changed energy by %e\n", imacro, dE);

        // Update in x
        Vector<Tbase> dx = x - old_x;

        // Repeat line search along this direction
        Tbase step = optimal_step(dx, x);
        if(std::isnormal(step) and step > Tbase(0) and step <= Tbase(1)) {
          auto new_point = fx(x+step*dx);
          if(new_point < current_point) {
            x += step*dx;
            //printf("Line search along dx changes energy by %e\n", new_point-current_point);
            current_point = new_point;
            dE = current_point - curval;
          }
        }
        old_x = x;

        if(dE > -df_tol) {
          log_(10, "A/EDIIS weights converged in %i macroiterations\n",(int) imacro);
          break;
        } else if(imacro==max_iter-1) {
          log_(10, "A/EDIIS weights did not converge in %i macroiterations, dE=%e\n", (int) imacro, (double) (dE));
        }

      }

      // Handle the edge case where the last matrix has zero norm
      if(x(0)==Tbase(0)) {
        x.setZero();
        x(0)=Tbase(1);
        // Reset search directions
        search_directions.setIdentity();
        for(Index i=0; i<b.size(); i++) {
          Vector<Tbase> c_i(search_directions.col(i));
          // x -> (1-step)*x + step*c_i = x + step*(c_i-x)
          Tbase step = optimal_step(c_i-x, x);
          if(!std::isnormal(step))
            continue;
          if(step > Tbase(0) and step < Tbase(1)) {
            auto new_point = fx(x+step*(c_i-x));
            if(new_point < current_point) {
              x += step*(c_i-x);
              current_point = new_point;
            }
          }
        }
      }

      //printf("Current energy %e\n",current_point);
      //throw std::logic_error("Stop");

      return x;
    }

    /// ADIIS linear term: <D_i - D_0 | F_0>
    Vector<Tbase> adiis_linear_term() const {
      // 2 * (tr(D_i F_0) - tr(D_0 F_0)) per history entry -- both
      // primitives live in trace_DF_cache_ and are shared with the
      // ADIIS/EDIIS quadratic-term builders.
      const size_t nhist = orbital_history_.size();
      const Tbase tD0_F0 = trace_DF_cached_(0, 0);
      Vector<Tbase> ret(nhist);
      for(size_t ihist = 0; ihist < nhist; ihist++)
        ret(ihist) = Tbase(2) * (trace_DF_cached_(ihist, 0) - tD0_F0);
      return ret;
    }

    /// EDIIS linear term: list of energies
    Vector<Tbase> ediis_linear_term() const {
      Vector<Tbase> ret = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
        ret(ihist) = get_energy(ihist);
      }
      return ret;
    }

    /// ADIIS quadratic term: <D_i - D_0 | F_j - F_0>
    Matrix<Tbase> adiis_quadratic_term() const {
      // Expand: tr((D_i - D_0)(F_j - F_0)) = tr(D_i F_j) - tr(D_i F_0)
      //                                    - tr(D_0 F_j) + tr(D_0 F_0)
      // All four primitives are cached via trace_DF_cached_.
      const size_t nhist = orbital_history_.size();
      const Tbase tD0_F0 = trace_DF_cached_(0, 0);
      Matrix<Tbase> ret(nhist, nhist);
      for(size_t ihist=0; ihist<nhist; ihist++) {
        const Tbase tDi_F0 = trace_DF_cached_(ihist, 0);
        for(size_t jhist=0; jhist<nhist; jhist++) {
          const Tbase tD0_Fj = trace_DF_cached_(0, jhist);
          ret(ihist, jhist) = trace_DF_cached_(ihist, jhist)
                            - tDi_F0 - tD0_Fj + tD0_F0;
        }
      }
      // Only the symmetric part matters; we also multiply by two
      // since we define the quadratic model as 0.5*x^T A x + b x
      return ret+ret.adjoint();
    }

    /// EDIIS quadratic term: -0.5*<D_i - D_j | F_i - F_j>
    Matrix<Tbase> ediis_quadratic_term() const {
      // Expand: tr((D_i - D_j)(F_i - F_j)) = tr(D_i F_i) - tr(D_i F_j)
      //                                    - tr(D_j F_i) + tr(D_j F_j)
      // Every primitive is cached via trace_DF_cached_ and shared
      // with the ADIIS terms.
      const size_t nhist = orbital_history_.size();
      Matrix<Tbase> ret(nhist, nhist);
      for(size_t ihist=0; ihist<nhist; ihist++) {
        const Tbase tDi_Fi = trace_DF_cached_(ihist, ihist);
        for(size_t jhist=0; jhist<nhist; jhist++) {
          const Tbase tDj_Fj = trace_DF_cached_(jhist, jhist);
          const Tbase tDi_Fj = trace_DF_cached_(ihist, jhist);
          const Tbase tDj_Fi = trace_DF_cached_(jhist, ihist);
          ret(ihist, jhist) = -(tDi_Fi - tDi_Fj - tDj_Fi + tDj_Fj);
        }
      }
      // Only the symmetric part matters; the factor 0.5 already
      // exists in the base model
      return (Tbase(1)/Tbase(2))*(ret+ret.adjoint());
    }

    /// Calculate ADIIS weights
    Vector<Tbase> adiis_weights() const {
      return aediis_weights(adiis_linear_term(), adiis_quadratic_term());
    }

    /// Calculate EDIIS weights
    Vector<Tbase> ediis_weights() const {
      return aediis_weights(ediis_linear_term(), ediis_quadratic_term());
    }

    /** Minimal Error Sampling Algorithm (MESA), doi:10.14288/1.0372885 */
    std::tuple<Vector<Tbase>,std::string> minimal_error_sampling_algorithm_weights(Tbase aediis_coeff) const {
      // Form the linear-extrapolation weights. LCIIS replaces Pulay's
      // CDIIS coefficients with the minimiser of the commutator norm
      // between the predicted Fock matrix and the predicted density;
      // everything downstream (the A/EDIIS mixing, the extrapolated
      // Fock build) is unchanged.
      const bool use_lciis = parse_method_string(methods_).lciis;
      Vector<Tbase> diis_w(use_lciis ? lciis_weights() : diis_weights());
      const std::string linear_step = use_lciis ? "LCIIS" : "DIIS";
      log_stream_(10) << linear_step << " weights: "
                      << diis_w.transpose() << std::endl;
      if(aediis_coeff == Tbase(0)) {
        return std::make_tuple(diis_w, linear_step);
      }

      // Get various extrapolation weights
      const size_t N = orbital_history_.size();
      Vector<Tbase> adiis_w(adiis_weights());
      log_stream_(10) << "ADIIS weights: " << adiis_w.transpose() << std::endl;
      Vector<Tbase> ediis_w(ediis_weights());
      log_stream_(10) << "EDIIS weights: " << ediis_w.transpose() << std::endl;

      // Candidates
      Matrix<Tbase> candidate_w = Matrix<Tbase>::Zero(N, 2);
      size_t icol=0;
      candidate_w.col(icol++) = adiis_w;
      candidate_w.col(icol++) = ediis_w;
      const std::vector<std::string> weight_legend({"ADIIS", "EDIIS"});
      std::string step;

      Vector<Tbase> density_projections = Vector<Tbase>::Zero(candidate_w.cols());
      for(Index iw=0;iw<candidate_w.cols();iw++) {
        density_projections(iw) = density_projection(Vector<Tbase>(candidate_w.col(iw)));
      }
      log_stream_(10) << "Density projections: " << density_projections.transpose() << std::endl;

      Index idx;
      density_projections.maxCoeff(&idx);
      log_(10, "Max density projection %e with %s weights\n",(double) (density_projections(idx)),weight_legend[idx].c_str());

      Vector<Tbase> aediis_w = candidate_w.col(idx);
      Vector<Tbase> weights(aediis_coeff * aediis_w + (Tbase(1) - aediis_coeff) * diis_w);
      if(aediis_coeff == Tbase(1)) {
        step = weight_legend[idx];
      } else {
        step = weight_legend[idx] + "+" + linear_step;
      }

      return std::make_tuple(weights,step);
    }

    /// Compute density change with given weights
    Tbase density_projection(const Vector<Tbase> & weights) const {
      // Get the extrapolated Fock matrix
      auto fock(extrapolate_fock(weights));

      // Reference calculation
      const auto reference_orbitals = get_orbitals();
      const auto reference_occupations = get_orbital_occupations();

      // Diagonalize the extrapolated Fock matrix
      auto diagonalized_fock = compute_orbitals(fock);
      auto & new_orbitals = diagonalized_fock.first;
      auto & new_orbital_energies = diagonalized_fock.second;

      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      return density_overlap(new_orbitals, new_occupations, reference_orbitals, reference_occupations);
    }

    /// Computes the difference between orbital occupations
    Tbase occupation_difference(const OrbitalOccupations<Tbase> & old_occ, const OrbitalOccupations<Tbase> & new_occ) const {
      Tbase diff = Tbase(0);
      for(size_t iblock = 0; iblock<old_occ.size(); iblock++) {
        if(old_occ[iblock].size()==0)
          continue;
        Index n = std::min(new_occ[iblock].size(), old_occ[iblock].size());
        diff += (new_occ[iblock].head(n) - old_occ[iblock].head(n)).array().abs().sum();
        if(new_occ[iblock].size()>n)
          diff += new_occ[iblock].tail(new_occ[iblock].size()-n).array().abs().sum();
        else if(old_occ[iblock].size()>n)
          diff += old_occ[iblock].tail(old_occ[iblock].size()-n).array().abs().sum();
      }

      return diff;
    }

    /// Perform DIIS extrapolation of Fock matrix
    FockMatrix<Torb> extrapolate_fock(const Vector<Tbase> & weights) const {
      if(weights.size() != (Index)orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.size() << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated Fock matrix
      FockMatrix<Torb> extrapolated_fock(number_of_blocks_);
      for(size_t iblock = 0; iblock < extrapolated_fock.size(); iblock++) {
        if(empty_block(iblock))
          continue;
        // Apply the DIIS weight
        for(size_t ihist = 0; ihist < orbital_history_.size(); ihist++) {
          Matrix<Torb> block = weights(ihist) * get_fock_matrix_block(ihist, iblock);
          if(ihist==0) {
            extrapolated_fock[iblock] = block;
          } else {
            extrapolated_fock[iblock] += block;
          }
        }
      }

      return extrapolated_fock;
    }

    /// Re-diagonalise a Hermitian density-matrix block into its
    /// natural-orbital representation. The sign is flipped before the
    /// eigendecomposition because Eigen returns eigenvalues in
    /// ascending order, so diagonalising ``-D`` and negating back
    /// yields the occupations in descending order -- the conventional
    /// natural-orbital ordering, with the populated orbitals first.
    ///
    /// Occupations whose magnitude is at or below ``zero_tol`` are
    /// snapped to exactly zero. The caller supplies that tolerance
    /// because the right value depends on the provenance of the
    /// density: a freshly built one carries only elementwise
    /// roundoff, while one that has been projected between basis
    /// sets and then mixed carries error of order ``cond * eps``.
    ///
    /// No positivity check is made here, and deliberately so. A
    /// convex combination of densities must come out positive
    /// semidefinite, but the DIIS extrapolation is an *affine*
    /// combination whose weights may be negative, so negative
    /// natural occupations are a normal outcome there. Callers that
    /// do require a genuine density check with
    /// ``require_nonnegative_occupations_``.
    void natural_orbitals_(const Matrix<Torb> & dm_block,
                           Tbase zero_tol,
                           Matrix<Torb> & orbitals,
                           Vector<Tbase> & occupations) const {
      Matrix<Torb> neg_dm = -dm_block;
      Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(neg_dm);
      occupations = es.eigenvalues();
      orbitals = es.eigenvectors();
      occupations *= Tbase{-1};
      for(Index k=0; k<occupations.size(); k++)
        if(std::abs(occupations(k)) <= zero_tol)
          occupations(k) = Tbase{0};
    }

    /// Throw unless every occupation in the block is non-negative to
    /// within ``fail_tol``. Used where the density is a convex
    /// combination and therefore must be positive semidefinite; a
    /// violation there means the mixing coefficients left their
    /// simplex, not that the eigensolver was noisy.
    void require_nonnegative_occupations_(
        const Vector<Tbase> & occupations, size_t iblock,
        Tbase fail_tol) const {
      if(occupations.minCoeff() < -fail_tol) {
        std::ostringstream oss;
        oss << "Negative natural occupation numbers in block " << iblock
            << "!\n" << occupations;
        throw std::logic_error(oss.str());
      }
    }

    /// Make the natural occupations of a convex combination of
    /// densities non-negative and particle-conserving: clamp away
    /// negative occupations, then rescale what is left so the block
    /// sums to ``target``, the trace the mixed density actually has.
    ///
    /// Both properties are invariants of the mixing rather than
    /// things to hope the eigensolver delivered. Every ingredient
    /// density is positive semidefinite with a known trace and the
    /// mixing coefficients lie on their simplex, so the mixture must
    /// be too; any negative occupation is noise, and any departure of
    /// the sum from ``target`` is the eigendecomposition's roundoff.
    ///
    /// Rescaling is what keeps the step conservative, and nothing here
    /// may discard occupation instead. Snapping values below a
    /// tolerance to zero would throw away up to that tolerance per
    /// orbital with nothing to put it back, and since the result is
    /// fed forward as the next iterate the loss would compound rather
    /// than stay an output artifact -- a shortfall that no tightening
    /// of ``convergence_threshold`` can reach, being a truncation and
    /// not an unconverged iterate.
    void conserve_block_occupations_(Vector<Tbase> & occupations,
                                     Tbase target) const {
      for(Index k = 0; k < occupations.size(); k++)
        if(occupations(k) < Tbase(0))
          occupations(k) = Tbase(0);
      Tbase sum = occupations.sum();
      if(sum > std::numeric_limits<Tbase>::min())
        occupations *= target / sum;
    }

    /// Perform DIIS extrapolation of density matrix
    DensityMatrix<Torb, Tbase> extrapolate_density(const Vector<Tbase> & weights) const {
      if(weights.size() != (Index)orbital_history_.size()) {
        std::ostringstream oss;
        oss << "Inconsistent weights: " << weights.size() << " elements vs orbital history of size " << orbital_history_.size() << "!\n";
        throw std::logic_error(oss.str());
      }

      // Form DIIS extrapolated density matrix
      std::vector<Matrix<Torb>> orbitals(number_of_blocks_);
      std::vector<Vector<Tbase>> occupations(number_of_blocks_);
      for(size_t iblock = 0; iblock < number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;

        Matrix<Torb> dm_block;
        for(size_t ihist = 0; ihist < orbital_history_.size(); ihist++) {
          Matrix<Torb> block = weights(ihist) * get_density_matrix_block(ihist, iblock);
          if(ihist==0) {
            dm_block = block;
          } else {
            dm_block += block;
          }
        }

        // The extrapolation weights may be negative, so this is an
        // affine and not a convex combination: negative natural
        // occupations are legitimate here and are left alone.
        const Tbase zero_tol = Tbase(10) * maximum_occupation_(iblock)
                             * std::numeric_limits<Tbase>::epsilon();
        natural_orbitals_(dm_block, zero_tol,
                          orbitals[iblock], occupations[iblock]);
      }

      return std::make_pair(orbitals,occupations);
    }

    /// Compute density overlap between two sets of orbitals and occupations
    Tbase density_overlap(const Orbitals<Torb> & lorb, const OrbitalOccupations<Tbase> & locc, const Orbitals<Torb> & rorb, const OrbitalOccupations<Tbase> & rocc) const {
      if(lorb.size() != rorb.size() or lorb.size() != locc.size() or lorb.size() != rocc.size())
        throw std::logic_error("Inconsistent orbitals!\n");

      Tbase ovl=Tbase(0);
      for(size_t iblock=0; iblock<lorb.size(); iblock++) {
        if(lorb[iblock].size()==0)
          continue;
        Matrix<Torb> Pl = build_density_block_(
            lorb[iblock], locc[iblock], maximum_occupation_(iblock));
        Matrix<Torb> Pr = build_density_block_(
            rorb[iblock], rocc[iblock], maximum_occupation_(iblock));
        ovl += tr_of_product_(Pl, Pr);
      }
      return ovl;
    }

    /// Attempt extrapolation with given weights
    bool attempt_extrapolation(const Vector<Tbase> & weights, bool density=false) {
      // Get the extrapolated Fock matrix
      if(not density) {
        auto fock(extrapolate_fock(weights));
        return attempt_fock(fock);
      } else {
        auto dm(extrapolate_density(weights));
        return add_entry(std::make_pair(dm.first, dm.second));
      }
    }

    /// See if given Fock matrix reduces the energy
    bool attempt_fock(const FockMatrix<Torb> & fock) {
      // Diagonalize the Fock matrix
      auto diagonalized_fock = compute_orbitals(fock);
      auto new_orbitals = diagonalized_fock.first;
      auto new_orbital_energies = diagonalized_fock.second;

      // Determine new occupations
      auto new_occupations = update_occupations(new_orbital_energies);

      // Try out the new occupations
      return add_entry(std::make_pair(new_orbitals, new_occupations));
    }

    /// Minimise ½ lam^T H lam + g^T lam on the product-of-simplices
    /// polytope { lam : lam_i >= 0, sum_{i in particle p} lam_i <= 1 }
    /// via a primal active-set QP. Working set is the indices of axes
    /// pinned at zero plus the particles whose sum-cap is binding;
    /// each iteration solves an equality-constrained KKT system for
    /// the step from the current iterate, walks the longest feasible
    /// step in that direction, and either drops the most-negative
    /// Lagrange multiplier or adds the blocking constraint to the
    /// working set. The cost is polynomial in the polytope dimension,
    /// which matters because enumerating faces instead would cost
    /// 2^(n_p+1)-1 per particle -- intractable for the npars in the
    /// tens that show up when several differently-occupied orbitals
    /// straddle the Aufbau Fermi level in a large degenerate group.
    /// H is symmetric but not necessarily
    /// positive-definite; the algorithm still terminates because of
    /// the iteration cap and falls back to lam = 0 if the KKT system
    /// is singular.
    std::pair<Vector<Tbase>, Tbase> solve_polytope_qp_(
        const Matrix<Tbase> & H,
        const Vector<Tbase> & g,
        Tbase E_orig,
        const std::vector<size_t> & particle_off,
        const std::vector<size_t> & particle_len) const {
      const size_t npars = g.size();
      const size_t nparts = particle_off.size();
      const Tbase eps = std::numeric_limits<Tbase>::epsilon();
      const Tbase tol = 100 * eps;

      auto model_value = [&](const Vector<Tbase> & lam) {
        return E_orig + g.dot(lam)
                      + Tbase(1)/Tbase(2) * (lam.transpose() * H * lam).value();
      };

      // Per-axis -> particle map.
      std::vector<size_t> particle_of(npars, nparts);
      for(size_t p = 0; p < nparts; p++)
        for(size_t i = particle_off[p]; i < particle_off[p] + particle_len[p]; i++)
          particle_of[i] = p;

      // Initial state: lam = 0, all non-neg constraints active, no
      // sum-caps active. Iteratively drop / add constraints until KKT.
      Vector<Tbase> lam = Vector<Tbase>::Zero(npars);
      std::vector<bool> at_zero(npars, true);
      std::vector<bool> sum_active(nparts, false);
      Vector<Tbase> nu = Vector<Tbase>::Zero(nparts);  // sum-cap multipliers

      const int max_iter = 8 * int(npars + nparts) + 16;
      for(int iter = 0; iter < max_iter; iter++) {
        std::vector<size_t> free_axes;
        for(size_t i = 0; i < npars; i++)
          if(!at_zero[i]) free_axes.push_back(i);
        const size_t n_free = free_axes.size();
        std::vector<size_t> active_sum_p;
        for(size_t p = 0; p < nparts; p++)
          if(sum_active[p]) active_sum_p.push_back(p);
        const size_t n_eq = active_sum_p.size();

        Vector<Tbase> p_step = Vector<Tbase>::Zero(npars);
        nu.setZero();
        bool solved = true;

        if(n_free > 0) {
          Matrix<Tbase> H_red(n_free, n_free);
          for(size_t r = 0; r < n_free; r++)
            for(size_t c = 0; c < n_free; c++)
              H_red(r, c) = H(free_axes[r], free_axes[c]);
          Vector<Tbase> g_red(n_free);
          Vector<Tbase> lam_red(n_free);
          for(size_t k = 0; k < n_free; k++) {
            g_red(k) = g(free_axes[k]);
            lam_red(k) = lam(free_axes[k]);
          }
          // Gradient of objective at lam, restricted to free axes
          // (at-zero axes contribute nothing since lam_i = 0 there).
          Vector<Tbase> grad_free = H_red * lam_red + g_red;

          Vector<Tbase> step_free;
          if(n_eq == 0) {
            try {
              step_free = H_red.colPivHouseholderQr().solve(-grad_free);
            } catch(...) { solved = false; }
            if(solved && !step_free.allFinite()) solved = false;
          } else {
            // Map each particle p with active sum-cap to its free-axis
            // local positions (rows of the constraint matrix).
            std::vector<std::vector<size_t>> particle_free(nparts);
            for(size_t k = 0; k < n_free; k++)
              particle_free[particle_of[free_axes[k]]].push_back(k);

            Matrix<Tbase> KKT = Matrix<Tbase>::Zero(n_free + n_eq, n_free + n_eq);
            Vector<Tbase> rhs = Vector<Tbase>::Zero(n_free + n_eq);
            KKT.block(0, 0, n_free, n_free) = H_red;
            rhs.segment(0, n_free) = -grad_free;
            for(size_t c = 0; c < n_eq; c++) {
              for(size_t local : particle_free[active_sum_p[c]]) {
                KKT(n_free + c, local) = 1;
                KKT(local, n_free + c) = 1;
              }
              // Step must preserve the active sum-cap equality, so
              // RHS for this constraint is 0 (not 1).
            }
            Vector<Tbase> sol;
            try {
              sol = KKT.colPivHouseholderQr().solve(rhs);
            } catch(...) { solved = false; }
            if(solved && !sol.allFinite()) solved = false;
            if(solved) {
              step_free = sol.segment(0, n_free);
              for(size_t c = 0; c < n_eq; c++)
                nu(active_sum_p[c]) = sol(n_free + c);
            }
          }
          if(solved)
            for(size_t k = 0; k < n_free; k++)
              p_step(free_axes[k]) = step_free(k);
        }

        if(!solved) break;  // singular KKT; keep best lam so far

        if(p_step.template lpNorm<Eigen::Infinity>() < tol) {
          // Step is zero -> we're at the QP optimum on the current
          // working set. Check Lagrange multipliers; drop the most-
          // negative active inequality, or stop if all multipliers are
          // non-negative.
          Vector<Tbase> grad_at_lam = g + H * lam;
          int    worst_kind = 0;   // 0 none, 1 non-neg axis, 2 sum-cap
          size_t worst_axis = npars;
          size_t worst_p    = nparts;
          Tbase  worst_val  = -tol;
          for(size_t i = 0; i < npars; i++) {
            if(!at_zero[i]) continue;
            Tbase mu = grad_at_lam(i)
                     + (sum_active[particle_of[i]]
                          ? nu(particle_of[i]) : Tbase(0));
            if(mu < worst_val) { worst_val = mu; worst_axis = i; worst_kind = 1; }
          }
          for(size_t p = 0; p < nparts; p++) {
            if(!sum_active[p]) continue;
            if(nu(p) < worst_val) { worst_val = nu(p); worst_p = p; worst_kind = 2; }
          }
          if(worst_kind == 0) break;  // KKT satisfied
          if(worst_kind == 1) at_zero[worst_axis] = false;
          else                sum_active[worst_p] = false;
          continue;
        }

        // Take the longest step alpha in [0, 1] that keeps lam feasible.
        Tbase alpha_max = std::numeric_limits<Tbase>::infinity();
        int    block_kind = 0;
        size_t block_axis = npars;
        size_t block_p    = nparts;
        for(size_t i = 0; i < npars; i++) {
          if(at_zero[i]) continue;
          if(p_step(i) < -tol) {
            Tbase a = -lam(i) / p_step(i);
            if(a < alpha_max) {
              alpha_max = a; block_axis = i; block_kind = 1;
            }
          }
        }
        for(size_t p = 0; p < nparts; p++) {
          if(sum_active[p]) continue;
          Tbase sum_lam = 0, sum_step = 0;
          for(size_t i = particle_off[p]; i < particle_off[p] + particle_len[p]; i++) {
            sum_lam += lam(i);
            sum_step += p_step(i);
          }
          if(sum_step > tol) {
            Tbase a = (Tbase(1) - sum_lam) / sum_step;
            if(a < alpha_max) {
              alpha_max = a; block_p = p; block_kind = 2;
            }
          }
        }

        Tbase alpha = std::min(alpha_max, Tbase(1));
        if(alpha < 0) alpha = 0;
        lam += alpha * p_step;
        for(size_t i = 0; i < npars; i++)
          if(lam(i) < 0 && lam(i) > -tol) lam(i) = 0;
        if(alpha < Tbase(1) - tol && block_kind != 0) {
          if(block_kind == 1) at_zero[block_axis] = true;
          else                sum_active[block_p] = true;
        }
      }
      return std::make_pair(lam, model_value(lam));
    }

    /// One ODA step, enumerating its own skeleton set and keeping
    /// nothing from it. This is the form the SCF state machine uses;
    /// the Aufbau cleanup drives ``optimal_damping_step_`` directly,
    /// because it needs to hold the skeleton set fixed across calls
    /// and to see the best density built even when it is rejected.
    bool optimal_damping_step(bool force_full = false) {
      std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>
        best_evaluated;
      SkeletonOccupations skeletons;
      return optimal_damping_step_(force_full, /*exclude_reference=*/false,
                                   best_evaluated, skeletons);
    }

    /// Optimal damping algorithm step. For each particle type, builds
    /// a list of skeleton density matrices corresponding to the
    /// extreme integer fillings of every degenerate orbital group at
    /// the current Fock matrix; then minimizes the energy on the
    /// product of per-particle simplices spanned by (P_reference,
    /// P_skel_1, ..., P_skel_n) using analytic-gradient cubic line
    /// searches along axes and pairwise vertex-vertex diagonals.
    /// Returns true if an entry strictly below the reference was
    /// added to the orbital history.
    /// One ODA step. ``force_full`` skips the cheap "collapsed
    /// skeleton" pass and goes straight to the full polytope search;
    /// used by ``run()`` for the one-shot check that ``converged()``
    /// really is a stationary point of the full skeleton set and not
    /// of the collapsed one.
    ///
    /// ``exclude_reference`` replaces the current density at the
    /// polytope's lambda = 0 vertex with the plain Aufbau filling of
    /// the current Fock matrix, so the search runs over the skeletons
    /// alone. Every skeleton is an Aufbau filling of one common set of
    /// orbitals, so every point of the polytope becomes a convex
    /// combination of such fillings -- full below the Fermi level,
    /// zero above it, fractional only inside the degenerate cluster
    /// the Fermi level lands in. See ``aufbau_cleanup_step``.
    /// ``best_evaluated`` receives the lowest-energy density this step
    /// built, whether or not it beat the reference and so whether or
    /// not the step reports success. Only filled under
    /// ``exclude_reference``, the Aufbau cleanup being the one caller
    /// that needs it: the occupations it wants are the simplex
    /// optimum, which loses to the converged density until the
    /// orbitals have been relaxed at them, so they would otherwise be
    /// discarded before the relaxation that justifies them could run.
    ///
    /// ``skeletons`` fixes the skeleton set across calls: empty on the
    /// way in it is filled with the set this call enumerated, and
    /// non-empty it is used in place of enumerating one. Which
    /// orbitals form the degenerate cluster, and which integer
    /// fillings of it to span, is a property of the solution being
    /// refined, not of each intermediate iterate. Re-deriving it every
    /// call makes it evaporate: relaxing the orbitals at a fixed
    /// fractional filling pushes the cluster apart by more than
    /// ``optimal_damping_degeneracy_threshold_``, after which the walk
    /// no longer recognises it and hands back a simplex of dimension
    /// zero with nothing left to optimise.
    /// With ``enumerate_only`` the call stops once ``skeletons`` has
    /// been filled and reports failure without having built a single
    /// Fock matrix. Enumeration needs the orbital energies and nothing
    /// else, so asking what the polytope looks like costs one
    /// diagonalisation; running a whole step to find out and throwing
    /// the result away costs a great deal more.
    bool optimal_damping_step_(bool force_full,
                               bool exclude_reference,
                               std::pair<DensityMatrix<Torb, Tbase>,
                                         FockBuilderReturn<Torb, Tbase>>
                                 & best_evaluated,
                               SkeletonOccupations & skeletons,
                               bool enumerate_only = false) {
      auto particles_left = [](Tbase n) {
        return n >= 10*std::numeric_limits<Tbase>::epsilon();
      };

      auto reference_orbitals = get_orbitals();
      auto reference_occupations = get_orbital_occupations();
      auto reference_fock = get_fock_matrix();
      auto reference_energy = get_energy();

      // Roothaan step: diagonalize current Fock matrix
      auto diagonalized_fock = compute_orbitals(reference_fock);
      auto new_orbitals = diagonalized_fock.first;
      auto new_orbital_energies = diagonalized_fock.second;

      if(exclude_reference)
        reference_orbitals = new_orbitals;

      // Skeleton occupations per particle type: [iparticle][itrial][iblock_within_particle]
      SkeletonOccupations trial_occupations_per_particle(number_of_blocks_per_particle_type_.size());

      const bool reuse_skeletons = !skeletons.empty();
      if(reuse_skeletons)
        trial_occupations_per_particle = skeletons;

      for(Index iparticle=0; !reuse_skeletons && iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
        size_t iblock_start = particle_block_offset(iparticle);
        size_t nblocks_iparticle = number_of_blocks_per_particle_type_(iparticle);

        std::vector<Vector<Tbase>> particle_occupations(nblocks_iparticle);
        for(size_t iblock=0; iblock<nblocks_iparticle; iblock++)
          particle_occupations[iblock] = Vector<Tbase>::Zero(new_orbital_energies[iblock_start+iblock].size());
        IndexVector orbital_index = IndexVector::Zero(nblocks_iparticle);

        Tbase num_left = number_of_particles_(iparticle);
        auto all_energies = order_orbitals_by_energy(new_orbital_energies, iparticle);

        size_t ifill=0;
        while(particles_left(num_left)) {
          // Find end of this degenerate group
          const size_t jfill = degenerate_cluster_end_(
              ifill, all_energies.size(),
              [&](size_t k) { return std::get<0>(all_energies[k]); });

          // Total capacity of the degenerate group
          Tbase maximum_capacity = Tbase(0);
          for(size_t iorb=ifill; iorb<jfill; iorb++)
            maximum_capacity += maximum_occupation_(std::get<1>(all_energies[iorb]));

          if(num_left >= maximum_capacity or jfill-ifill==1) {
            // Group is fully filled or only one orbital — single skeleton
            for(size_t iorb=ifill; iorb<jfill; iorb++) {
              auto block_index = std::get<1>(all_energies[iorb]);
              auto capacity = maximum_occupation_(block_index);
              auto fill = std::min(capacity, num_left);
              particle_occupations[block_index-iblock_start](orbital_index(block_index-iblock_start)++) = fill;
              num_left -= fill;
              if(not particles_left(num_left))
                break;
            }
            if(not particles_left(num_left))
              trial_occupations_per_particle[iparticle].push_back(particle_occupations);
          } else {
            if(verbosity_>=5) {
              log_(5, "Degenerate orbitals: iblock iorb E\n");
              for(size_t iorb=ifill; iorb<jfill; iorb++)
                log_(5, "%s %3i % .9f\n",
                       block_descriptions_[std::get<1>(all_energies[iorb])].c_str(),
                       (int) std::get<2>(all_energies[iorb]),
                       (double) (std::get<0>(all_energies[iorb])));
            }

            // Enumerate the extremal vertices of the integer-filling
            // polytope
            //     V = { n in R^N : 0 <= n_k <= c_k, sum_k n_k = num_left }
            // where the N=jfill-ifill orbitals of the degenerate group
            // are indexed flat across whichever blocks they live in.
            // Each vertex of V has at most one fractional component:
            // a subset S of the N indices is fully filled (n_k = c_k)
            // and at most one residual index j carries num_left -
            // sum_{k in S} c_k. This enumerates every distinct integer
            // filling regardless of whether the group spans one block
            // (intra-block accidental degeneracy in a no-symmetry run)
            // or several (cross-block crossings such as 4s/3d in
            // atoms). The dedup check below collapses gauge-equivalent
            // skeletons in symmetric cases.
            struct GroupOrb {
              size_t local_block;     // block_index - iblock_start
              size_t slot_offset;     // position among this block's group orbitals
              Tbase capacity;
            };
            std::vector<GroupOrb> group_orbs;
            std::map<size_t, size_t> next_offset_per_block;
            for(size_t iorb_idx=ifill; iorb_idx<jfill; iorb_idx++) {
              auto block_index = std::get<1>(all_energies[iorb_idx]);
              size_t local_block = block_index - iblock_start;
              size_t offset = next_offset_per_block[local_block]++;
              group_orbs.push_back({local_block, offset,
                                    maximum_occupation_(block_index)});
            }
            const size_t N_group = group_orbs.size();
            if(N_group > 8 * sizeof(size_t) - 1)
              throw std::logic_error("Degenerate group too large for subset enumeration; raise the degeneracy threshold or split the group.\n");

            auto try_emit = [&](const std::vector<Tbase> & fills) {
              auto iter_occupations = particle_occupations;
              for(size_t k=0; k<N_group; k++) {
                const auto & g = group_orbs[k];
                iter_occupations[g.local_block](
                  orbital_index(g.local_block) + g.slot_offset) = fills[k];
              }
              auto match = [this, &iter_occupations](const std::vector<Vector<Tbase>> & list_occ) {
                Tbase sqdiff=Tbase(0);
                for(size_t iblock=0; iblock<list_occ.size(); iblock++)
                  sqdiff += (list_occ[iblock]-iter_occupations[iblock]).norm();
                return sqdiff < occupation_change_threshold_;
              };
              auto idx = std::find_if(trial_occupations_per_particle[iparticle].begin(),
                                      trial_occupations_per_particle[iparticle].end(),
                                      match);
              if(idx == trial_occupations_per_particle[iparticle].end())
                trial_occupations_per_particle[iparticle].push_back(iter_occupations);
            };

            const Tbase tol = occupation_change_threshold_;
            const size_t n_subsets = size_t(1) << N_group;
            for(size_t mask=0; mask<n_subsets; mask++) {
              Tbase filled = 0;
              for(size_t k=0; k<N_group; k++)
                if(mask & (size_t(1) << k))
                  filled += group_orbs[k].capacity;
              Tbase residual = num_left - filled;
              if(residual < -tol)
                continue;  // S overfills

              std::vector<Tbase> fills(N_group, Tbase(0));
              for(size_t k=0; k<N_group; k++)
                if(mask & (size_t(1) << k))
                  fills[k] = group_orbs[k].capacity;

              if(std::abs(residual) < tol) {
                try_emit(fills);
              } else {
                // residual > 0: pick one orbital not in S to take it.
                // residual >= c_k is equivalent to flipping k into S,
                // which is enumerated by a different mask, so skip it
                // here to avoid trivial duplicates.
                for(size_t k=0; k<N_group; k++) {
                  if(mask & (size_t(1) << k))
                    continue;
                  Tbase cap_k = group_orbs[k].capacity;
                  if(residual < cap_k - tol) {
                    auto fills_with_frac = fills;
                    fills_with_frac[k] = residual;
                    try_emit(fills_with_frac);
                  }
                }
              }
            }

            num_left = Tbase(0);
          }

          ifill = jfill;
        }
      }

      // Hand the enumerated set back so later calls can hold it fixed.
      if(!reuse_skeletons)
        skeletons = trial_occupations_per_particle;

      // Everything past this point evaluates densities. A caller that
      // only wanted to know the shape of the polytope stops here.
      if(enumerate_only)
        return false;

      // Skeleton-set attempts. When the full enumeration would yield
      // more skeletons than there are particle types (i.e. at least
      // one particle has a degenerate group whose integer occupation
      // patterns produce multiple vertices), first try a "smudged"
      // pass where each particle's skeleton set collapses to its
      // arithmetic mean: one skeleton per particle, one lambda per
      // particle. That reduces npars from the enumerated total down
      // to n_particle_types, cutting the vertex-evaluation cost from
      // the pathological "high-symmetry guess" case (28 dimensions
      // in the trace that motivated this) to a handful. Trust-region
      // refits are only enabled on the fallback full-skeleton pass;
      // the collapsed pass is a coarse first-approximation escape.
      const size_t n_particle_types = number_of_blocks_per_particle_type_.size();
      size_t full_npars = 0;
      for(auto & trial: trial_occupations_per_particle)
        full_npars += trial.size();
      std::vector<std::vector<std::vector<std::vector<Vector<Tbase>>>>> attempts;
      if(!force_full && full_npars > n_particle_types) {
        auto collapsed = trial_occupations_per_particle;
        for(auto & particle : collapsed) {
          if(particle.size() < 2) continue;
          auto avg = particle[0];
          for(size_t s = 1; s < particle.size(); s++)
            for(size_t b = 0; b < avg.size(); b++)
              avg[b] += particle[s][b];
          for(size_t b = 0; b < avg.size(); b++)
            avg[b] /= Tbase(particle.size());
          particle = { std::move(avg) };
        }
        attempts.push_back(std::move(collapsed));
      }
      attempts.push_back(std::move(trial_occupations_per_particle));

      last_oda_via_collapsed_ = false;
      bool overall_succ = false;
      for(size_t iattempt = 0; iattempt < attempts.size() && !overall_succ; iattempt++) {
        trial_occupations_per_particle = std::move(attempts[iattempt]);

        if(exclude_reference) {
          // Reparametrise the polytope so the current density drops
          // out of it. One skeleton is promoted to the lambda = 0
          // vertex and removed from the axes, which leaves n skeletons
          // described by n-1 parameters -- the simplex they span,
          // exactly. Promoting a skeleton without removing it would
          // describe the same simplex with n parameters, and the extra
          // one does nothing: raising its lambda while the slack
          // 1 - sum(lambda) falls moves no density, so the model
          // Hessian would carry a zero mode and the axis vertex would
          // cost a Fock build to re-evaluate the reference.
          //
          // The polytope keeps its shape, {lambda >= 0,
          // sum(lambda) <= 1}, so the QP, the cubic rays and the
          // backoff scaling need no special case. Every vertex is then
          // an Aufbau filling of one common set of orbitals, so every
          // point in it is a convex combination of those and has them
          // as its natural orbitals. Mixing densities that carry
          // *different* orbitals is what makes natural occupations
          // depart from Aufbau, and the current density is the only
          // ingredient that would.
          for(size_t iparticle = 0;
              iparticle < trial_occupations_per_particle.size(); iparticle++) {
            auto & trials = trial_occupations_per_particle[iparticle];
            if(trials.empty()) continue;
            const size_t offset = particle_block_offset(iparticle);
            for(size_t iblock = 0; iblock < trials[0].size(); iblock++)
              reference_occupations[offset + iblock] = trials[0][iblock];
            trials.erase(trials.begin());
          }
          auto reference_build = fock_builder_(
              std::make_pair(reference_orbitals, reference_occupations));
          number_of_fock_evaluations_++;
          reference_energy = reference_build.first;
          // The promoted vertex is a genuine density; let it compete
          // for the solution like any other trial this step evaluates.
          add_entry(std::make_pair(reference_orbitals, reference_occupations),
                    reference_build);
          best_evaluated = std::make_pair(
              std::make_pair(reference_orbitals, reference_occupations),
              reference_build);
          log_(5, "Aufbau cleanup: lambda = 0 vertex set to a skeleton, "
                  "energy % .10f\n", (double) (reference_energy));
        }

        const bool refits_enabled = (iattempt == attempts.size() - 1);
        const bool this_attempt_is_collapsed =
            (iattempt == 0 && attempts.size() > 1);
        if(iattempt > 0)
          log_(5, "Collapsed polytope did not descend; retrying with full skeleton set.\n");
        size_t npars = 0;
        for(auto & trial: trial_occupations_per_particle)
          npars += trial.size();
        // Record the polytope dimension so the outer SCF state machine
        // can size its post-ODA orbital-rotation burst when the user has not overridden
        // orbital_rotation_steps_after_oda_.
        last_polytope_dimension_ = npars;
        log_(5, "%i parameters in optimal damping (attempt %s)\n", (int) npars,
             attempts.size() > 1
               ? (iattempt == 0 ? "collapsed" : "full")
               : "full");
        log_flush_();
        if(npars==0)
          continue;

        // Build mixed density from a parameter vector lambda. Per
        // particle type, the lambda subvector has one entry per skeleton;
        // the residual (1 - sum) goes to the reference density.
        auto interpolate_density = [this, &reference_orbitals, &reference_occupations, &new_orbitals, &trial_occupations_per_particle](const Vector<Tbase> & lambda) {
          Orbitals<Torb> interp_orbs(reference_orbitals.size());
          OrbitalOccupations<Tbase> interp_occs(reference_orbitals.size());

          size_t iparam=0;
          for(Index iparticle=0; iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
            size_t ntrial = trial_occupations_per_particle[iparticle].size();
            if(ntrial==0) {
              // No axes for this particle, so its density is the
              // lambda = 0 vertex and nothing interpolates. Copy the
              // reference across rather than falling through, which
              // would leave this particle's blocks default-constructed
              // and hand back a density with holes in it.
              for(size_t iblock_particle = 0;
                  iblock_particle < (size_t)number_of_blocks_per_particle_type_(iparticle);
                  iblock_particle++) {
                size_t iblock = iblock_particle + particle_block_offset(iparticle);
                interp_orbs[iblock] = reference_orbitals[iblock];
                interp_occs[iblock] = reference_occupations[iblock];
              }
              continue;
            }
            // Project this particle's lambda block onto its simplex
            // {lambda >= 0, sum(lambda) <= 1} before using it.
            //
            // The QP solver enforces that simplex only to the accuracy
            // of its constrained linear solve, so an ill-conditioned
            // reduced Hessian -- exactly what a starting density
            // projected between two different basis sets produces --
            // can leave sum(lambda) above 1 by of order cond * eps.
            // Any overshoot hands old_dm a negative weight and makes
            // the mixed density non-positive-semidefinite, which then
            // surfaces far downstream as a negative natural occupation.
            // Nothing in the algorithm wants lambda outside the
            // simplex: the trial loop only ever scales candidates
            // *down*, so clamping here loses no intended step.
            Vector<Tbase> lam_p = lambda.segment(iparam, ntrial);
            HelperRoutines::project_onto_unit_simplex<Tbase>(lam_p);
            const Tbase lambda_sum = lam_p.sum();

            for(size_t iblock_particle = 0; iblock_particle < (size_t)number_of_blocks_per_particle_type_(iparticle); iblock_particle++) {
              size_t iblock = iblock_particle + particle_block_offset(iparticle);
              if(empty_block(iblock))
                continue;

              Matrix<Torb> old_dm = build_density_block_(
                  reference_orbitals[iblock], reference_occupations[iblock],
                  maximum_occupation_(iblock));

              Vector<Tbase> new_occ = lam_p(0)*trial_occupations_per_particle[iparticle][0][iblock_particle];
              for(size_t itrial=1; itrial<ntrial; itrial++)
                new_occ += lam_p(itrial)*trial_occupations_per_particle[iparticle][itrial][iblock_particle];

              Matrix<Torb> new_dm = build_density_block_(
                  new_orbitals[iblock], new_occ, maximum_occupation_(iblock));
              Matrix<Torb> mix_dm = (Tbase(1) - lambda_sum)*old_dm + new_dm;

              // Unlike the DIIS extrapolation this *is* a convex
              // combination -- lam_p was just projected onto its
              // simplex -- so the mixed density must be positive
              // semidefinite and must carry the trace its ingredients
              // carried. Both are invariants of the step, so they are
              // enforced below rather than left to the accuracy of the
              // eigendecomposition.
              //
              // The abort guard is what distinguishes noise from a
              // genuinely corrupt density, and it fires below
              // -eps^(1/4) (~1.2e-4 in double, 1e-8 in quad). That is
              // deliberately loose: a density projected between basis
              // sets and then mixed has an eigendecomposition carrying
              // error of order cond * eps, which can be thousands of
              // epsilons. It is checked on the raw occupations, before
              // the clean-up below hides anything from it.
              const Tbase eps_occ   = std::numeric_limits<Tbase>::epsilon();
              const Tbase occ_scale = std::max(Tbase(1), maximum_occupation_(iblock));
              const Tbase fail_tol  = std::sqrt(std::sqrt(eps_occ)) * occ_scale;

              // Nothing is snapped to zero here: see
              // conserve_block_occupations_ for why discarding small
              // occupations is not free.
              natural_orbitals_(mix_dm, Tbase(0),
                                interp_orbs[iblock], interp_occs[iblock]);
              require_nonnegative_occupations_(interp_occs[iblock], iblock,
                                               fail_tol);
              // The target trace comes from the ingredients, not from
              // mix_dm: the orbitals are orthonormal, so each block's
              // density has the trace its occupation vector sums to,
              // and reading it off the inputs keeps the target free of
              // the very roundoff being corrected for.
              conserve_block_occupations_(
                  interp_occs[iblock],
                  (Tbase(1) - lambda_sum) * reference_occupations[iblock].sum()
                    + new_occ.sum());
            }
            iparam += ntrial;
          }
          if(iparam != (size_t)lambda.size())
            throw std::logic_error("Indexing inconsistency in optimal_damping_step\n");

          return std::make_pair(interp_orbs, interp_occs);
        };

        auto evaluate = [this, &interpolate_density](const Vector<Tbase> & lambda) {
          auto dm = interpolate_density(lambda);
          auto fock = fock_builder_(dm);
          return std::make_pair(dm, fock);
        };

        // Trace of ``(D_a - D_b) F`` given pre-materialised density
        // blocks. Uses tr(A F) = sum_{ij} A(i,j) * F(j,i) evaluated as
        // an element-wise product sum -- O(N^2) instead of the O(N^3)
        // matmul-then-trace form. Combined with the pre-materialised
        // density blocks this collapses the axis-Hessian construction
        // from O(npars^2 * N^3) to O(npars * N^3 + npars^2 * N^2).
        auto trace_diff = [this](const std::vector<Matrix<Torb>> & D_a,
                                 const std::vector<Matrix<Torb>> & D_b,
                                 const FockMatrix<Torb> & fock) {
          Tbase tr = Tbase(0);
          for(size_t iblock=0; iblock<fock.size(); iblock++) {
            if(empty_block(iblock))
              continue;
            tr += tr_of_product_(D_a[iblock] - D_b[iblock], fock[iblock]);
          }
          return tr;
        };

        Vector<Tbase> x0 = Vector<Tbase>::Zero(npars);
        const Tbase E_orig = reference_energy;
        const auto & F_orig = reference_fock;
        const DensityMatrix<Torb,Tbase> P_orig = std::make_pair(reference_orbitals, reference_occupations);

        // Evaluate each canonical vertex (one lambda_i = 1, others 0).
        // The axis-vertex densities all share the same orbitals
        // (new_orbitals) and differ only in their occupation vectors,
        // so they go through the batched Fock-builder helper which the
        // caller can override to amortise integral / grid setup.
        std::vector<DensityMatrix<Torb,Tbase>> axis_densities(npars);
        for(size_t idim=0; idim<npars; idim++) {
          x0.setZero();
          x0(idim) = Tbase(1);
          axis_densities[idim] = interpolate_density(x0);
        }
        auto axis_fock = evaluate_batch_(axis_densities);

        std::vector<std::pair<DensityMatrix<Torb,Tbase>,FockBuilderReturn<Torb,Tbase>>> evaluations(npars);
        for(size_t idim=0; idim<npars; idim++) {
          evaluations[idim] = std::make_pair(std::move(axis_densities[idim]),
                                             std::move(axis_fock[idim]));
          log_(5, "Roothaan step in dimension %i yields energy % .10f change %e\n",
                 (int) idim, (double) (evaluations[idim].second.first), (double) (evaluations[idim].second.first - E_orig));
        }

        // Build a second-order Taylor model of the energy on the product
        // of per-particle simplices around lambda = 0:
        //   E(lambda) ~= E_orig + g^T lambda + 0.5 lambda^T H lambda.
        // Gradient: g_i = tr(F_orig * (P_i - P_orig)).
        // Diagonal Hessian: H_ii = 2*(E_i - E_orig - g_i), the Hermite
        //   quadratic fit through (0, E_orig, g_i) and (1, E_i).
        // Off-diagonal Hessian: H_ij ~= tr((F_j - F_orig) * (P_i - P_orig)),
        //   exact when the energy is quadratic in P (Hartree-Fock) and a
        //   second-order finite difference otherwise; symmetrized over (i,j).
        // No additional Fock evaluations beyond the npars axis vertices.
        //
        // Materialise D_orig and each D_axis[i] once; the trace-diff
        // helper then does O(npars^2) elementwise-product traces on
        // those cached blocks rather than rebuilding a density matrix
        // per call.
        const std::vector<Matrix<Torb>> D_orig = build_density_blocks_(P_orig);
        std::vector<std::vector<Matrix<Torb>>> D_axis(npars);
        for(size_t i=0; i<npars; i++)
          D_axis[i] = build_density_blocks_(evaluations[i].first);

        Vector<Tbase> grad(npars);
        for(size_t i=0; i<npars; i++)
          grad(i) = trace_diff(D_axis[i], D_orig, F_orig);
        Matrix<Tbase> hess(npars, npars);
        for(size_t i=0; i<npars; i++) {
          Tbase E_i = evaluations[i].second.first;
          hess(i, i) = 2*(E_i - E_orig - grad(i));
          for(size_t j=i+1; j<npars; j++) {
            const auto & F_j = evaluations[j].second.second;
            const auto & F_i = evaluations[i].second.second;
            Tbase from_j = trace_diff(D_axis[i], D_orig, F_j) - grad(i);
            Tbase from_i = trace_diff(D_axis[j], D_orig, F_i) - grad(j);
            hess(i, j) = (Tbase(1)/Tbase(2))*(from_j + from_i);
            hess(j, i) = hess(i, j);
          }
        }

        // Particle layout for the polytope: each active particle's lambda
        // sub-vector lives on its own simplex {lambda >= 0, sum(lambda) <= 1}.
        std::vector<size_t> particle_off, particle_len;
        {
          size_t off = 0;
          for(Index p=0; p<number_of_blocks_per_particle_type_.size(); p++) {
            size_t nt = trial_occupations_per_particle[p].size();
            if(nt > 0) {
              particle_off.push_back(off);
              particle_len.push_back(nt);
            }
            off += nt;
          }
        }
        // Minimise the quadratic model on the product-of-simplices
        // polytope via the active-set QP solver. The previous code
        // enumerated every face of the polytope (product over particles
        // of 2^(n_p+1)-1 faces), which is intractable when degenerate
        // groups span several blocks and produce npars in the tens.
        const Tbase eps = std::numeric_limits<Tbase>::epsilon();
        auto model_value = [&](const Vector<Tbase> & lam) {
          return E_orig + grad.dot(lam) + (Tbase(1)/Tbase(2))*(lam.transpose()*hess*lam).value();
        };
        (void) model_value;
        Vector<Tbase> lam_opt;
        Tbase model_min;
        std::tie(lam_opt, model_min) = solve_polytope_qp_(
            hess, grad, E_orig, particle_off, particle_len);
        log_(5, "Quadratic model minimum at lambda = (");
        for(Index i=0; i<lam_opt.size(); i++)
          log_(5, "%s%g", i ? "," : "", (double) (lam_opt(i)));
        log_(5, "), model energy change %e\n", (double) (model_min - E_orig));

        // Candidate list: (lambda, tag, model-predicted energy at lambda).
        // The QP model gives one; along each 1D axis and each pair-
        // diagonal edge we also fit a cubic Hermite polynomial through
        // the two endpoint energies and the two endpoint slopes. Those
        // four data exactly determine a cubic and cost no Fock builds
        // beyond the npars axis vertices already evaluated. Interior
        // minima of each 1D cubic become candidates, scored by that
        // polynomial's own value at the root (the multi-dimensional
        // quadratic underestimates 1D non-linearity along the ray).
        //
        // A quartic is deliberately NOT used. Along a linear ray in
        // density space the Hartree-Fock energy is exactly quadratic, so
        // the cubic already carries a spare order for the
        // exchange-correlation non-linearity. More to the point, no
        // genuinely independent fifth datum is available for free:
        //   * H_ii = 2 (E_1 - E_0 - g_0) is itself the Hermite quadratic
        //     through the same three data, so imposing it makes the
        //     quartic's residual vanish identically -- it adds nothing.
        //   * E'(1) - E'(0) is a function of two constraints the fit
        //     already carries; its residual equals -a3/2 of this very
        //     cubic, i.e. the same information redistributed.
        // Both would also form that residual as a difference of two
        // total energies, so near convergence they amplify roundoff into
        // spurious stationary points. A real fifth datum would need a
        // midpoint energy (an extra Fock build per ray) or the
        // exchange-correlation kernel, which the Fock-builder interface
        // does not expose.
        struct Candidate {
          Vector<Tbase> lam;
          std::string tag;
          Tbase model_score;
        };
        std::vector<Candidate> candidates;
        if(lam_opt.template lpNorm<Eigen::Infinity>() > 100*eps)
          candidates.push_back({lam_opt, "model min", model_min});

        // Fit the cubic through (0, E0, dE0) and (1, E1, dE1) and emit a
        // candidate at each interior minimum. ``place`` writes the root
        // into the lambda vector for the ray being probed.
        auto add_cubic_candidates =
            [&](Tbase E0, Tbase dE0, Tbase E1, Tbase dE1,
                const std::string & tag, auto && place) {
          auto c = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(
                       E0, dE0, Tbase(1), E1, dE1);
          std::pair<Tbase, Tbase> zeros;
          try {
            zeros = std::apply(HelperRoutines::cubic_polynomial_zeros<Tbase>, c);
          } catch(std::logic_error &) {
            return;   // constant derivative, or no real stationary point
          }
          const std::array<Tbase, 4> coeffs = {std::get<0>(c), std::get<1>(c),
                                               std::get<2>(c), std::get<3>(c)};
          bool emitted = false;
          Tbase first_root = Tbase(0);
          for(Tbase z : {zeros.first, zeros.second}) {
            if(!(z > 100*eps && z < Tbase(1) - 100*eps))
              continue;
            // Keep minima only: f''(z) = 2 a2 + 6 a3 z > 0. The other
            // root of the derivative is a maximum and is never a useful
            // trial step. This is the same test the sigma line search
            // applies to its cubic fits.
            if(!(Tbase(2)*coeffs[2] + Tbase(6)*coeffs[3]*z > Tbase(0)))
              continue;
            // cubic_polynomial_zeros returns a doubled root when the
            // cubic degenerates to a quadratic; do not emit it twice.
            if(emitted && std::abs(z - first_root) <= 100*eps)
              continue;
            Vector<Tbase> xc = Vector<Tbase>::Zero(npars);
            place(z, xc);
            candidates.push_back({std::move(xc), tag,
                                  HelperRoutines::evaluate_polynomial<Tbase, 4>(coeffs, z)});
            emitted = true;
            first_root = z;
          }
        };

        for(size_t i=0; i<npars; i++) {
          Tbase E_i = evaluations[i].second.first;
          Tbase g_i = grad(i);
          Tbase slope_at_1 = trace_diff(D_axis[i], D_orig, evaluations[i].second.second);
          add_cubic_candidates(E_orig, g_i, E_i, slope_at_1,
                               std::string("cubic axis ") + std::to_string(i),
                               [i](Tbase z, Vector<Tbase> & xc) { xc(i) = z; });
        }
        // Pair-diagonal cubics along each edge from vertex e_i to vertex
        // e_j, parameterised by t with lambda = (1-t) e_i + t e_j.
        for(size_t i=0; i<npars; i++) {
          for(size_t j=i+1; j<npars; j++) {
            const auto & F_i = evaluations[i].second.second;
            const auto & F_j = evaluations[j].second.second;
            Tbase E_i = evaluations[i].second.first;
            Tbase E_j = evaluations[j].second.first;
            Tbase slope_i = trace_diff(D_axis[j], D_axis[i], F_i);
            Tbase slope_j = trace_diff(D_axis[j], D_axis[i], F_j);
            add_cubic_candidates(E_i, slope_i, E_j, slope_j,
                                 std::string("cubic edge ") + std::to_string(i) + "-" + std::to_string(j),
                                 [i,j](Tbase z, Vector<Tbase> & xc) {
                                   xc(i) = Tbase(1) - z;
                                   xc(j) = z;
                                 });
          }
        }

        // Rank candidates by their model-predicted energy so the trial
        // loop tries the most-promising step first. For HF the model
        // is exact along any linear ray, so the top candidate is the
        // true minimum and a single Fock build lands the ODA step; for
        // DFT the ordering is still an accurate heuristic near
        // convergence, cutting the trial loop from O(N) Fock builds to
        // one whenever the model doesn't lie.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate & a, const Candidate & b) {
                    return a.model_score < b.model_score;
                  });

        if(verbosity_ >= 5) {
          size_t n_model = 0, n_axis = 0, n_edge = 0;
          for(const auto & cand : candidates) {
            if(cand.tag == "model min") n_model++;
            else if(cand.tag.rfind("cubic axis", 0) == 0) n_axis++;
            else if(cand.tag.rfind("cubic edge", 0) == 0) n_edge++;
          }
          log_(5, "Trial loop: %zu candidates (%zu quadratic-model + "
                 "%zu cubic-axis + %zu cubic-edge); ordered by "
                 "predicted energy so the first Fock build usually "
                 "lands the ODA step.\n",
                 candidates.size(), n_model, n_axis, n_edge);
        }

        // Trial loop: at each backoff scale, evaluate candidates in
        // ranked order and stop at the first descent step. When the
        // model is accurate (always for HF along a linear ray, and
        // typically for DFT near convergence) the first candidate wins
        // and only a single Fock build runs per ODA step. Rejected
        // candidates still feed the orbital history via add_entry, so
        // subsequent DIIS iterations see the trial densities.
        bool succ = false;
        Vector<Tbase> x_accepted;
        std::pair<DensityMatrix<Torb,Tbase>,FockBuilderReturn<Torb,Tbase>> eval_accepted;
        for(int scalefac=0; scalefac<=5; scalefac++) {
          Tbase scale = std::pow(Tbase(2), -scalefac);
          for(const auto & cand : candidates) {
            Vector<Tbase> x_scaled = scale * cand.lam;
            auto eval = evaluate(x_scaled);
            number_of_fock_evaluations_++;
            // Only the excluded-reference path wants this, and there
            // best_evaluated was seeded from the promoted vertex above,
            // so the comparison always has something to compare to.
            if(exclude_reference
               && eval.second.first < best_evaluated.second.first)
              best_evaluated = eval;
            bool ok = add_entry(eval.first, eval.second);
            if(ok) {
              succ = true;
              x_accepted = x_scaled;
              eval_accepted = std::move(eval);
            }
            log_(10, "ODA %s at scale %g gives E = % .10f, change %e%s\n",
                   cand.tag.c_str(), (double) (scale), (double) (ok ? eval_accepted.second.first : eval.second.first),
                   (double) ((ok ? eval_accepted.second.first : eval.second.first) - E_orig),
                   ok ? " (accepted)" : "");
            if(ok) break;
          }
          if(succ) break;
        }

        // Trust-region refinement: after the initial descent, re-anchor
        // the polytope quadratic model at the accepted iterate using
        // the gradient observed there (free -- F_accepted has just been
        // computed) and re-solve the QP. Along a linear ray in density
        // space the Hartree-Fock energy is exactly quadratic, so this
        // converges in one refit; for DFT a few iterations catch the
        // residual non-quadraticity that the initial axis-vertex data
        // missed. Each refit costs one Fock build. The refined
        // densities also flow through add_entry so DIIS sees them.
        //
        // Refinement stops when any of:
        //   (a) the QP anchor barely moves (|dlambda|_inf < 100 eps);
        //   (b) the actual energy drop this refit is below one tenth
        //       of the SCF convergence threshold (further Fock builds
        //       cannot influence the outer SCF stopping decision);
        //   (c) the refit fails to descend;
        //   (d) max_oda_refits_ iterations have been taken.
        if(succ && refits_enabled && max_oda_refits_ > 0) {
          // Anchor for the "worth refining" test: use the improvement
          // achieved by the accepted initial trial-loop step. Refits
          // that squeeze at most one-tenth of the convergence threshold
          // out of it cannot change the outer SCF's stopping decision.
          const Tbase refit_progress_tol = minimum_useful_descent_();
          for(int refit=0; refit < max_oda_refits_; refit++) {
            const auto & F_at_x = eval_accepted.second.second;
            Tbase E_at_x = eval_accepted.second.first;
            Vector<Tbase> g_at_x(npars);
            for(size_t i=0; i<npars; i++)
              g_at_x(i) = trace_diff(D_axis[i], D_orig, F_at_x);
            // Re-express the model anchored at x_accepted in the QP's
            // canonical (E_const + g*lambda + 0.5 lambda^T H lambda)
            // form. Expanding the Taylor around x_accepted:
            //   E(lambda) = E_at_x + g_at_x . (lambda - x_accepted)
            //             + 0.5 (lambda - x_accepted)^T H (lambda - x_accepted).
            Vector<Tbase> g_eff = g_at_x - hess * x_accepted;
            Tbase E_eff = E_at_x - g_at_x.dot(x_accepted)
                        + Tbase(1)/Tbase(2) * (x_accepted.transpose() * hess * x_accepted).value();
            Vector<Tbase> lam_new;
            Tbase model_min_new;
            std::tie(lam_new, model_min_new) = solve_polytope_qp_(
                hess, g_eff, E_eff, particle_off, particle_len);
            Vector<Tbase> delta = lam_new - x_accepted;
            Tbase delta_inf = delta.template lpNorm<Eigen::Infinity>();
            if(delta_inf < Tbase(100) * eps) {
              log_(10, "ODA refit %i: |dlambda|_inf = %e below noise, model has "
                       "converged at the accepted iterate.\n", refit + 1, (double) (delta_inf));
              break;
            }
            // Predicted improvement pre-check: if the QP's own model
            // says the new anchor barely lowers the energy, we can bail
            // without building the Fock. The model is exact for HF
            // along a linear ray and typically accurate for DFT near
            // convergence, so trusting its prediction here saves a Fock
            // per idle refit call.
            Tbase model_delta = model_min_new - E_at_x;
            if(-model_delta < refit_progress_tol) {
              log_(10, "ODA refit %i: model predicts progress %e below %e; "
                       "skipping Fock build and exiting refinement.\n",
                       refit + 1, (double) (-model_delta), (double) (refit_progress_tol));
              break;
            }
            auto eval_new = evaluate(lam_new);
            number_of_fock_evaluations_++;
            bool ok = add_entry(eval_new.first, eval_new.second);
            Tbase E_new = eval_new.second.first;
            Tbase delta_E = E_new - E_at_x;
            log_(10, "ODA refit %i: |dlambda|_inf = %e, model E = % .10f, "
                     "actual E = % .10f, change %e%s\n",
                     refit + 1, (double) (delta_inf), (double) (model_min_new), (double) (E_new), (double) (delta_E),
                     ok ? " (accepted)" : "");
            if(!ok) {
              // Refined step didn't improve on the previously accepted
              // one. Keep the previous accept and stop refining.
              break;
            }
            x_accepted = lam_new;
            eval_accepted = std::move(eval_new);
            if(-delta_E < refit_progress_tol) {
              // Improvement is smaller than a fraction of the SCF
              // convergence threshold; further refits cannot influence
              // the outer stopping decision. Take the last accepted
              // iterate and let DIIS spend its Fock builds instead.
              log_(10, "ODA refit %i: further refit progress %e below "
                       "%e; exiting refinement loop.\n",
                       refit + 1, (double) (-delta_E), (double) (refit_progress_tol));
              break;
            }
          }
        }

        if(succ && this_attempt_is_collapsed)
          last_oda_via_collapsed_ = true;
        overall_succ = succ;
      }  // end for(iattempt)

      if(overall_succ) {
        // ODA has globally rearranged the orbital basis (and possibly
        // the occupation pattern), so the orbital-rotation history no
        // longer describes the current iterate.
        // The end-of-iteration cleanup() in run() runs the density-
        // matrix-difference pruning; doing it here too would print the
        // "Density matrix difference ..." line twice per ODA iteration.
        clear_orbital_rotation_history_();
      }
      // Update the active-rotation count seen at the new iterate so the
      // outer state machine can size its orbital-rotation burst from it.
      last_active_rotation_count_ = compute_active_rotation_count();
      return overall_succ;
    }

    /// Clean up history from incorrect occupations
    void cleanup() {
      // Pruning compares every entry against the lowest-energy one, so
      // with fewer than two there is nothing to compare and nothing to
      // remove. Leaving early also keeps the unsigned ``size()-1``
      // below from wrapping on an empty history, and the reads of
      // ``density_differences(0)`` and ``idx(0)`` from running off the
      // end of a vector that would have no entries. A one-entry
      // history reaches here whenever a step declines to evaluate any
      // density -- which needs the method set to leave the stall exit
      // unreachable, as ``ODA + CG`` and ``ODA + LBFGS`` do, DIIS
      // being absent.
      if(orbital_history_.size() < 2)
        return;

      Vector<Tbase> density_differences = Vector<Tbase>::Zero(orbital_history_.size()-1);
      for(size_t ihist=1;ihist<orbital_history_.size();ihist++) {
        density_differences(ihist-1)=density_matrix_difference(ihist, 0);
      }
      if(verbosity_ >= 10) {
        log_stream_(10) << "Density differences: " << density_differences.transpose() << std::endl;
      } else if(verbosity_>=5) {
        log_(5, "Density matrix difference %e between lowest-energy and newest entry\n",(double) (density_differences(0)));
      }

      // Sort the differences
      IndexVector idx(sort_index_ascending(density_differences));
      // Pick the indices that don't satisfy the criterion
      Tbase ref_diff = density_differences(idx(0));
      IndexVector sub_idx = find_indices_where(idx, [&](Index k){
        return density_restart_factor_*density_differences(k) > ref_diff;
      });
      if(sub_idx.size()) {
        IndexVector filtered_idx(sub_idx.size());
        for(Index k=0;k<sub_idx.size();k++)
          filtered_idx(k) = idx(sub_idx(k));
        // Sort descending
        std::sort(filtered_idx.data(), filtered_idx.data()+filtered_idx.size(), std::greater<Index>());
        log_(10, "Removing %i entries corresponding to large change in density matrix\n",(int) filtered_idx.size());
        for(Index k=0;k<filtered_idx.size();k++) {
          Index ihistm1 = filtered_idx(k);
          // Remember the off-by-one in the indices
          orbital_history_.erase(orbital_history_.begin()+ihistm1+1);
        }
        prune_diis_caches_();
      }
    }

    /// Take one preconditioned scaled-steepest-descent step on the
    /// orbital-rotation manifold. Pseudo-diagonalizes the reference
    /// Fock matrix within each equal-occupation sub-block to obtain
    /// canonical-orbital energy estimates; uses those for a Newton-
    /// like diagonal-Hessian preconditioner so that rotations whose
    /// natural energy scales span many orders of magnitude are still
    /// scaled appropriately. Line-searches along the unitary curve
    /// C(t) = C_pseudo * exp(t K) by parabolic-fit refinement.
    /// Returns true if a strictly lower-energy entry was added to
    /// the history, false on stall (no descent, no rotation degrees
    /// of freedom, or line search exhausted).
    // --- Orbital-rotation step infrastructure ----------------------
    //
    // scaled_steepest_descent_step() (PR+ CG) and lbfgs_step()
    // (limited-memory BFGS, follow-up) both follow the same outline:
    //
    //   1. Pseudo-diagonalise F within equal-occupation sub-blocks to
    //      get a canonical orbital basis and orbital energies eps.
    //   2. Collect orbital-rotation degrees of freedom (b, i, j) with
    //      different occupations.
    //   3. Compute the gradient g_alpha and diagonal Hessian h_alpha.
    //   4. Build a descent direction d, possibly with curvature
    //      correction (PR+ CG or L-BFGS).
    //   5. Run a sigma-line-search along C(t) = C_pseudo * exp(t K).
    //
    // Steps 1-3 and the K, t_max, evaluate-at, slope-at-trial pieces
    // of step 5 are shared. The two algorithms differ only in how
    // step 4 builds the trial-1 direction (PR+ CG mix vs L-BFGS two-
    // loop recursion) and in what state they carry between calls.
    struct RotationStepContext {
      Orbitals<Torb> C_pseudo;
      OrbitalOccupations<Tbase> n_ref;
      std::vector<Vector<Tbase>> eps;
      std::vector<OrbitalRotation> dofs;
      Vector<Tbase> g;
      Vector<Tbase> h;
      Tbase E_ref = 0;
      size_t n_dof = 0;
      size_t n_par = 0;
      bool is_complex = false;
    };

    /// Build the rotation-step context from the current iterate.
    /// Returns false if there are no orbital-rotation DOFs.
    bool build_rotation_step_context_(RotationStepContext & ctx) const {
      if(orbital_history_.empty()) return false;

      ctx.n_ref = get_orbital_occupations();
      auto C_ref = get_orbitals();
      auto F_ref = get_fock_matrix();
      ctx.E_ref = get_energy();
      size_t nblocks = C_ref.size();
      ctx.is_complex = Eigen::NumTraits<Torb>::IsComplex;

      // Step 1: pseudo-diagonalise F within equal-occupation sub-blocks.
      pseudo_canonicalise_(C_ref, F_ref, ctx.n_ref, ctx.C_pseudo, ctx.eps);

      // Step 2: collect orbital-rotation degrees of freedom (i > j
      // pairs within the same block, with non-trivially different
      // occupations). Equal-occupation pairs are gauge directions
      // and excluded.
      std::vector<Matrix<Torb>> F_pseudo(nblocks);
      for(size_t b = 0; b < nblocks; b++) {
        if(empty_block(b) || ctx.C_pseudo[b].cols() == 0) {
          F_pseudo[b].resize(0, 0);
          continue;
        }
        F_pseudo[b] = ctx.C_pseudo[b].adjoint() * F_ref[b] * ctx.C_pseudo[b];
        Index n_b = ctx.C_pseudo[b].cols();
        for(Index i = 0; i < n_b; i++)
          for(Index j = 0; j < i; j++)
            if(std::abs(ctx.n_ref[b](i) - ctx.n_ref[b](j)) >= occupation_change_threshold_)
              ctx.dofs.emplace_back(b, i, j);
      }
      if(ctx.dofs.empty()) {
        log_(5, "Rotation step: no orbital rotation degrees of freedom.\n");
        return false;
      }

      // Step 3: gradient g_alpha = 2 Re(F_ij)(n_j - n_i) and diagonal
      // Hessian estimate h_alpha = 2 (eps_i - eps_j)(n_j - n_i). For
      // complex orbitals the real and imaginary parts of K_ij are
      // independent DOFs sharing h.
      ctx.n_dof = ctx.dofs.size();
      ctx.n_par = ctx.is_complex ? 2 * ctx.n_dof : ctx.n_dof;
      ctx.g = Vector<Tbase>::Zero(ctx.n_par);
      ctx.h = Vector<Tbase>::Zero(ctx.n_par);
      for(size_t a = 0; a < ctx.n_dof; a++) {
        const auto & dof = ctx.dofs[a];
        size_t b = std::get<0>(dof);
        Index i = std::get<1>(dof);
        Index j = std::get<2>(dof);
        Tbase dn = ctx.n_ref[b](j) - ctx.n_ref[b](i);
        Tbase de = ctx.eps[b](i) - ctx.eps[b](j);
        Torb Fij = F_pseudo[b](i, j);
        ctx.g(a) = 2 * std::real(Fij) * dn;
        ctx.h(a) = 2 * de * dn;
        if(ctx.is_complex) {
          ctx.g(ctx.n_dof + a) = 2 * std::imag(Fij) * dn;
          ctx.h(ctx.n_dof + a) = ctx.h(a);
        }
      }
      return true;
    }

    /// Wrong-sign-safe preconditioned-SD direction at level shift sigma:
    ///     d_alpha = -g_alpha / (sigma + max(0, h_alpha)).
    /// Uses the diagonal Hessian estimate h_alpha when it is positive
    /// (Newton-like step in the well-conditioned direction) and drops
    /// it otherwise (scaled SD in the ill-conditioned or wrong-sign
    /// direction). sigma >= initial_level_shift_ > 0 keeps the
    /// denominator strictly positive throughout.
    Vector<Tbase> preconditioned_sd_direction_(
        const RotationStepContext & ctx, Tbase sigma) const {
      Vector<Tbase> d(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        d(k) = -ctx.g(k) / (sigma + std::max(Tbase(0), ctx.h(k)));
      return d;
    }

    /// Assemble the anti-Hermitian rotation generator K (one block per
    /// orbital block) from a flat amplitude vector d.
    Orbitals<Torb> build_K_(const Vector<Tbase> & d,
                            const RotationStepContext & ctx) const {
      Orbitals<Torb> K(ctx.C_pseudo.size());
      for(size_t b = 0; b < ctx.C_pseudo.size(); b++)
        if(ctx.C_pseudo[b].cols() > 0)
          K[b] = Matrix<Torb>::Zero(ctx.C_pseudo[b].cols(), ctx.C_pseudo[b].cols());
      for(size_t a = 0; a < ctx.n_dof; a++) {
        const auto & dof = ctx.dofs[a];
        size_t b = std::get<0>(dof);
        Index i = std::get<1>(dof);
        Index j = std::get<2>(dof);
        if constexpr (!Eigen::NumTraits<Torb>::IsComplex) {
          K[b](i, j) = d(a);
          K[b](j, i) = -d(a);
        } else {
          Torb val(d(a), d(ctx.n_dof + a));
          K[b](i, j) = val;
          K[b](j, i) = -std::conj(val);
        }
      }
      return K;
    }

    /// Maximum step length along exp(t K): one quasi-period of the
    /// largest-magnitude eigenvalue of -i K (which is real, since K
    /// is anti-Hermitian).
    Tbase t_max_for_K_(const Orbitals<Torb> & K) const {
      Tbase t_max = std::numeric_limits<Tbase>::max();
      for(size_t b = 0; b < K.size(); b++) {
        if(K[b].size() == 0) continue;
        Matrix<std::complex<Tbase>> KI = K[b].template cast<std::complex<Tbase>>() * std::complex<Tbase>(Tbase{0}, Tbase{-1});
        // Hermitize to suppress eig_sym roundoff warnings; -iK is
        // analytically Hermitian for anti-Hermitian K.
        KI = std::complex<Tbase>(Tbase{(Tbase(1)/Tbase(2))}) * (KI + KI.adjoint().eval());
        // Only the extreme eigenvalue is wanted, so the eigenvectors are
        // not formed: the step length is a trust radius, and the
        // rotation itself is built by expm_antihermitian.
        Eigen::SelfAdjointEigenSolver<Matrix<std::complex<Tbase>>> es(KI, Eigen::EigenvaluesOnly);
        Vector<Tbase> ev = es.eigenvalues();
        if(ev.size() > 0) {
          Tbase max_abs = ev.array().abs().maxCoeff();
          if(max_abs > 0) t_max = std::min(t_max, Tbase(M_PI / 2) / max_abs);
        }
      }
      return t_max;
    }

    /// Evaluate the Fock builder at the rotated iterate
    /// C_pseudo * exp(t K) with the same occupations as the reference
    /// iterate. Increments the global Fock-evaluation counter.
    std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>
    evaluate_rotation_at_(const Orbitals<Torb> & K, Tbase t,
                          const RotationStepContext & ctx) {
      size_t nblocks = ctx.C_pseudo.size();
      Orbitals<Torb> C_new(nblocks);
      for(size_t b = 0; b < nblocks; b++) {
        if(ctx.C_pseudo[b].cols() == 0) {
          C_new[b] = ctx.C_pseudo[b];
          continue;
        }
        Matrix<Torb> tK = t * K[b];
        C_new[b] = ctx.C_pseudo[b] * expm_antihermitian(tK);
      }
      DensityMatrix<Torb, Tbase> dm = std::make_pair(C_new, ctx.n_ref);
      auto fock = fock_builder_(dm);
      number_of_fock_evaluations_++;
      return std::make_pair(dm, fock);
    }

    /// Directional gradient at a trial iterate, computed from F^MO at
    /// that iterate: g_trial_a = 2 Re[(C_trial^dag F_trial C_trial)_{i,j}](n_j - n_i).
    /// F^MO is computed once per block and reused across DOFs.
    Vector<Tbase> directional_gradient_at_trial_(
        const RotationStepContext & ctx,
        const Orbitals<Torb> & C_trial,
        const FockMatrix<Torb> & F_trial) const {
      std::vector<Matrix<Torb>> F_MO(C_trial.size());
      for(size_t b = 0; b < C_trial.size(); b++) {
        if(C_trial[b].cols() == 0) continue;
        F_MO[b] = C_trial[b].adjoint() * F_trial[b] * C_trial[b];
      }
      Vector<Tbase> g_trial = Vector<Tbase>::Zero(ctx.n_par);
      for(size_t a = 0; a < ctx.n_dof; a++) {
        const auto & dof = ctx.dofs[a];
        size_t b = std::get<0>(dof);
        Index i = std::get<1>(dof);
        Index j = std::get<2>(dof);
        Torb Fij = F_MO[b](i, j);
        Tbase dn = ctx.n_ref[b](j) - ctx.n_ref[b](i);
        g_trial(a) = 2 * std::real(Fij) * dn;
        if(ctx.is_complex)
          g_trial(ctx.n_dof + a) = 2 * std::imag(Fij) * dn;
      }
      return g_trial;
    }

    /// Line search along the unitary curve C(t) = C_pseudo * exp(t K)
    /// at fixed occupations. trial_0_direction(sigma) returns the
    /// initial descent direction at sigma = initial_level_shift_,
    /// including whatever curvature correction the caller applies
    /// (PR+ CG, L-BFGS, ...).
    ///
    /// Two nested loops:
    ///   * Inner t-walk: at fixed (direction, K, t_max) start at
    ///     t = min(1, t_max) and refine by cubic-Hermite minimisation
    ///     using the directional gradient at every rejected trial.
    ///     Each step uses (E_ref, slope at t=0) and (E_trial, slope at
    ///     t_trial) to locate the next interior minimum of the cubic;
    ///     iterates until either a descent step is found, t falls
    ///     below 1e-6 * t_max, or max_t_trials evaluations have been
    ///     spent. This captures the points where the directional
    ///     gradient changes sign, both "overshot" (slope > 0 at
    ///     t_trial) and "non-monotonic" (slope < 0 still but the
    ///     energy has risen and fallen).
    ///   * Outer sigma fallback: if the t-walk exhausts trials, the
    ///     direction itself is suspect; raise sigma (geometric or
    ///     cubic-Hermite-in-sigma when first-trial slope data is on
    ///     hand) and rebuild the preconditioned SD direction.
    /// Returns true on success, writing the accepted direction to
    /// d_accepted and the step length taken along it to t_accepted.
    /// The two are reported separately because callers want different
    /// things: the CG recursion conjugates against the previous
    /// direction, whereas L-BFGS needs the displacement t*d that the
    /// step actually made in parameter space.
    template<typename Trial0DirectionFunc>
    bool sigma_line_search_(const RotationStepContext & ctx,
                            Trial0DirectionFunc trial_0_direction,
                            Vector<Tbase> & d_accepted,
                            Tbase & t_accepted,
                            const char * tag) {
      const Tbase sigma_0 = initial_level_shift_;
      const int max_sigma_trials = 3;
      const int max_t_trials = 8;
      const Tbase t_floor_ratio = Tbase(1e-6);
      const Tbase gnorm2 = ctx.g.dot(ctx.g);
      const Tbase slope_u_at_0 = -gnorm2 / sigma_0;

      Tbase sigma = sigma_0;
      bool first_sigma = true;
      bool have_sigma_cubic = false;
      Tbase E_first_sigma_trial = ctx.E_ref;
      Tbase slope_u_at_1 = 0;

      bool success = false;
      for(int sigma_trial = 0; sigma_trial < max_sigma_trials && !success; sigma_trial++) {
        Vector<Tbase> d = first_sigma
          ? trial_0_direction(sigma)
          : preconditioned_sd_direction_(ctx, sigma);

        Tbase slope_0 = d.dot(ctx.g);  // dE/dt at t = 0
        if(!std::isfinite(slope_0) || slope_0 >= 0) {
          log_(5, "%s: direction at sigma = %e is not descent (g.d = %e).\n",
                 tag, (double) (sigma), (double) (slope_0));
          sigma *= 2;
          first_sigma = false;
          continue;
        }

        Orbitals<Torb> K = build_K_(d, ctx);
        Tbase t_max = t_max_for_K_(K);
        if(!std::isfinite(t_max) || t_max <= 0) {
          log_(5, "%s: t_max not well-defined at sigma = %e.\n", tag, (double) (sigma));
          sigma *= 2;
          first_sigma = false;
          continue;
        }

        // Inner t-walk: cubic-Hermite-refined Armijo line search.
        Tbase t = std::min(Tbase(1), t_max);
        const Tbase t_floor = t_max * t_floor_ratio;
        bool first_t_trial_in_sigma = true;
        for(int t_trial = 0; t_trial < max_t_trials && !success; t_trial++) {
          auto trial_result = evaluate_rotation_at_(K, t, ctx);
          Tbase E_t = trial_result.second.first;
          log_(5, "%s: trial sigma %e t %e, energy % .10f, change %e\n",
                 tag, (double) (sigma), (double) (t), (double) (E_t), (double) (E_t - ctx.E_ref));

          if(E_t < ctx.E_ref) {
            add_entry(trial_result.first, trial_result.second);
            d_accepted = d;
            t_accepted = t;
            success = true;
            break;
          }

          // Failed t-trial: gather slope at the trial point so we can
          // predict either where the slope flips sign (overshoot) or
          // where the interior minimum of a non-monotonic profile sits.
          Vector<Tbase> g_t = directional_gradient_at_trial_(
              ctx, trial_result.first.first, trial_result.second.second);
          Tbase slope_t = g_t.dot(d);

          // Record info for the outer sigma cubic Hermite (only the
          // very first trial in this sigma-trial provides it).
          if(first_sigma && first_t_trial_in_sigma) {
            Tbase dEdsigma = 0;
            for(size_t k = 0; k < ctx.n_par; k++) {
              Tbase denom = sigma_0 + std::max(Tbase(0), ctx.h(k));
              dEdsigma += g_t(k) * ctx.g(k) / (denom * denom);
            }
            slope_u_at_1 = -sigma_0 * dEdsigma;
            E_first_sigma_trial = E_t;
            have_sigma_cubic = true;
          }
          first_t_trial_in_sigma = false;

          if(t <= t_floor) break;

          // Predict next t from the cubic Hermite fit on [0, t] with
          // E(0) = E_ref, E'(0) = slope_0, E(t) = E_t, E'(t) = slope_t.
          Tbase t_next = t * Tbase(1)/Tbase(2);
          try {
            auto cubic = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(
                ctx.E_ref, slope_0, t, E_t, slope_t);
            Tbase a2 = std::get<2>(cubic);
            Tbase a3 = std::get<3>(cubic);
            auto roots = HelperRoutines::cubic_polynomial_zeros<Tbase>(
                std::get<0>(cubic), std::get<1>(cubic), a2, a3);
            Tbase t_star = std::numeric_limits<Tbase>::quiet_NaN();
            for(Tbase r : {roots.first, roots.second}) {
              if(!(r > 0 && r < t)) continue;
              if(2*a2 + 6*a3*r > 0) { t_star = r; break; }
            }
            if(std::isfinite(t_star) && t_star > 0 && t_star < t) {
              t_next = t_star;
              log_(5, "%s: cubic Hermite predicts t = %e (in [0, %e]).\n",
                     tag, (double) (t_next), (double) (t));
            }
          } catch(const std::logic_error &) {
            // Cubic derivative has no real roots; fall through to halving.
          }
          if(t_next < t_floor) t_next = t_floor;
          if(t_next >= t)      t_next = t * Tbase(1)/Tbase(2);  // ensure progress
          t = t_next;
        }
        if(success) break;

        first_sigma = false;
        if(sigma_trial + 1 == max_sigma_trials) break;

        // Outer sigma fallback: predict the next sigma from a cubic
        // Hermite fit in u = sigma_0/sigma using slope_u data, or fall
        // back to geometric doubling.
        Tbase sigma_next = sigma * 2;
        if(have_sigma_cubic) {
          auto cubic = HelperRoutines::fit_cubic_polynomial_with_derivatives<Tbase>(
              ctx.E_ref, slope_u_at_0, Tbase(1), E_first_sigma_trial, slope_u_at_1);
          Tbase a2 = std::get<2>(cubic);
          Tbase a3 = std::get<3>(cubic);
          try {
            auto roots = HelperRoutines::cubic_polynomial_zeros<Tbase>(
                std::get<0>(cubic), std::get<1>(cubic), a2, a3);
            Tbase u_star = std::numeric_limits<Tbase>::quiet_NaN();
            for(Tbase u : {roots.first, roots.second}) {
              if(!(u > 0 && u < 1)) continue;
              if(2*a2 + 6*a3*u > 0) { u_star = u; break; }
            }
            if(std::isfinite(u_star) && u_star > 0 && u_star < 1) {
              Tbase predicted = sigma_0 / u_star;
              if(std::isfinite(predicted) && predicted > sigma
                 && predicted < sigma * 100) {
                sigma_next = predicted;
                log_(5, "%s: cubic Hermite predicts sigma = %e (u* = %e).\n",
                       tag, (double) (sigma_next), (double) (u_star));
              }
            }
          } catch(const std::logic_error &) {
            // Cubic derivative has no real roots; fall through to geometric.
          }
        }
        sigma = sigma_next;
      }
      return success;
    }

    /// Apply the PR+ CG mix in place to a preconditioned-SD direction
    /// when compatible state from the previous call is on hand. Falls
    /// through silently (leaves d unchanged) otherwise; rejects the CG
    /// direction if it has lost descent character.
    void apply_pr_plus_cg_mix_(Vector<Tbase> & d,
                               const RotationStepContext & ctx) const {
      if(previous_orbital_gradient_.size() != ctx.g.size()
         || previous_orbital_direction_.size() != ctx.g.size()
         || previous_orbital_dofs_ != ctx.dofs)
        return;
      Tbase denom = previous_orbital_gradient_.dot(previous_orbital_gradient_);
      if(denom <= std::numeric_limits<Tbase>::min()) return;
      Tbase beta_PR = ctx.g.dot(ctx.g - previous_orbital_gradient_) / denom;
      Tbase beta = std::max(beta_PR, Tbase(0));
      Vector<Tbase> d_cg = d + beta * previous_orbital_direction_;
      if(d_cg.dot(ctx.g) < 0) {
        log_(5, "Scaled SD: CG update with beta = %e (PR = %e).\n", (double) (beta), (double) (beta_PR));
        d = d_cg;
      } else if(verbosity_ >= 5) {
        log_(5, "Scaled SD: CG direction not descent, resetting to preconditioned SD.\n");
      }
    }

    /// Preconditioned PR+ scaled-steepest-descent step on the orbital
    /// rotations at fixed occupations.
    bool scaled_steepest_descent_step() {
      RotationStepContext ctx;
      if(!build_rotation_step_context_(ctx)) return false;

      Vector<Tbase> d_accepted;
      Tbase t_accepted = 0;
      bool success = sigma_line_search_(
          ctx,
          [&](Tbase sigma) {
            Vector<Tbase> d = preconditioned_sd_direction_(ctx, sigma);
            apply_pr_plus_cg_mix_(d, ctx);
            return d;
          },
          d_accepted,
          t_accepted,
          "Scaled SD");

      if(success) {
        previous_orbital_gradient_ = ctx.g;
        previous_orbital_direction_ = d_accepted;
        previous_orbital_dofs_ = ctx.dofs;
      } else {
        previous_orbital_gradient_.resize(0);
        previous_orbital_direction_.resize(0);
        previous_orbital_dofs_.clear();
      }
      return success;
    }

    /// Forget everything the orbital-rotation steps carry between
    /// calls -- the PR+ CG direction and the L-BFGS curvature pairs.
    ///
    /// Both describe a trajectory of successive rotation steps, so
    /// they are only meaningful while the rotations are the only thing
    /// moving the iterate. Whenever another method (an accepted ODA
    /// step, an accepted extrapolation) relocates it, the recorded
    /// gradient belongs to a point the solver has left, and the next
    /// y = g_new - g_old would be measured across that jump as well as
    /// across the rotation.
    void clear_orbital_rotation_history_() {
      previous_orbital_gradient_.resize(0);
      previous_orbital_direction_.resize(0);
      previous_orbital_dofs_.clear();
      lbfgs_ = LBFGSState();
    }

    /// L-BFGS two-loop recursion applied to the current gradient ctx.g,
    /// using the wrong-sign-safe diagonal H_0 = diag(1/(sigma + max(0, h)))
    /// as the initial inverse-Hessian approximation. Returns the search
    /// direction d = -H_k g, where H_k is the limited-memory inverse
    /// Hessian built from the stored (s_i, y_i, rho_i) triples.
    Vector<Tbase> lbfgs_direction_(
        const RotationStepContext & ctx, Tbase sigma) const {
      const auto & s = lbfgs_.s;
      const auto & y = lbfgs_.y;
      const auto & rho = lbfgs_.rho;
      Vector<Tbase> q = ctx.g;
      size_t m = s.size();
      std::vector<Tbase> alpha(m);
      for(size_t i = m; i-- > 0;) {
        alpha[i] = rho[i] * s[i].dot(q);
        q -= alpha[i] * y[i];
      }
      Vector<Tbase> r(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        r(k) = q(k) / (sigma + std::max(Tbase(0), ctx.h(k)));
      for(size_t i = 0; i < m; i++) {
        Tbase beta = rho[i] * y[i].dot(r);
        r += (alpha[i] - beta) * s[i];
      }
      return -r;
    }

    /// Fraction of the preconditioned-SD descent rate that the L-BFGS
    /// two-loop direction has to retain to be worth taking; see
    /// apply_lbfgs_correction_.
    ///
    /// Fitted on O/PBE/cc-pVDZ at M = 1 and M = 3 over the mixes that
    /// exercise the two-loop direction hardest, the ones without ODA
    /// to re-anchor the iterate. Requiring nothing beyond descent
    /// (the equivalent of 0) leaves "DIIS + LBFGS" needing 200
    /// iterations at M = 3 and bare "LBFGS" not converging at all;
    /// 0.2, 0.5 and 0.8 take that case to 22, 15 and 13 iterations
    /// respectively, but 0.8 is tight enough to stall bare "LBFGS" at
    /// M = 1, which 0.5 converges in 31. Mixes containing ODA are
    /// insensitive across the whole range. 0.5 is thus the strictest
    /// setting that leaves every tested configuration converging.
    static constexpr Tbase lbfgs_minimum_relative_descent_ = Tbase(0.5);

    /// Apply the L-BFGS correction to d in place. d on entry is the
    /// preconditioned-SD direction (L-BFGS with empty history); replace
    /// it with the full two-loop direction when the stored history is
    /// compatible with the current DOFs and the two-loop direction is
    /// the better of the two. Otherwise leave d as preconditioned SD.
    ///
    /// "Better" is measured as the descent rate per unit step length,
    /// d.g/|d|, which is what the line search along exp(t K) actually
    /// gets to spend: a direction may descend and still be a poor
    /// trade if it descends far more slowly for its length than the
    /// direction it displaces. Merely requiring d.g < 0 admits exactly
    /// that, and the curvature history then compounds it, since the
    /// step it produces becomes the next s.
    void apply_lbfgs_correction_(Vector<Tbase> & d,
                                 const RotationStepContext & ctx) const {
      if(lbfgs_.s.empty()) return;
      if(lbfgs_.history_dofs != ctx.dofs) return;
      Vector<Tbase> d_lbfgs = lbfgs_direction_(ctx, initial_level_shift_);
      Tbase rate_lbfgs = -d_lbfgs.dot(ctx.g) / d_lbfgs.norm();
      Tbase rate_sd = -d.dot(ctx.g) / d.norm();
      if(rate_lbfgs >= lbfgs_minimum_relative_descent_ * rate_sd) {
        log_(5, "L-BFGS: applying two-loop direction (history size %zu).\n",
               lbfgs_.s.size());
        d = d_lbfgs;
      } else if(verbosity_ >= 5) {
        log_(5, "L-BFGS: two-loop direction descends at %e per unit length "
                "against preconditioned SD's %e, keeping SD.\n",
             (double) (rate_lbfgs), (double) (rate_sd));
      }
    }

    /// L-BFGS step on the orbital rotations. Mirrors
    /// scaled_steepest_descent_step but builds the trial-1 direction
    /// from a limited-memory BFGS approximation to the inverse Hessian
    /// rather than from PR+ CG.
    bool lbfgs_step() {
      RotationStepContext ctx;
      if(!build_rotation_step_context_(ctx)) return false;
      LBFGSState & st = lbfgs_;

      // Promote the pending (s, g_prev) into a full (s, y) history
      // pair using the current gradient, but only if the DOF set is
      // unchanged since the pair was recorded.
      if(st.pending_s.size() == (Index)ctx.n_par
         && st.pending_g.size() == (Index)ctx.n_par
         && st.history_dofs == ctx.dofs) {
        Vector<Tbase> y = ctx.g - st.pending_g;
        Tbase ys = y.dot(st.pending_s);
        if(ys > std::numeric_limits<Tbase>::min()) {
          st.s.push_back(st.pending_s);
          st.y.push_back(y);
          st.rho.push_back(Tbase(1) / ys);
          while(st.s.size() > (size_t) maximum_history_length_) {
            st.s.pop_front();
            st.y.pop_front();
            st.rho.pop_front();
          }
        } else if(verbosity_ >= 5) {
          log_(5, "L-BFGS: curvature condition violated (y.s = %e), pair dropped.\n", (double) (ys));
        }
      } else if(!st.history_dofs.empty() && st.history_dofs != ctx.dofs) {
        st = LBFGSState();
      }
      // Pending pair has been consumed; clear it before this step.
      st.pending_s.resize(0);
      st.pending_g.resize(0);

      Vector<Tbase> d_accepted;
      Tbase t_accepted = 0;
      bool success = sigma_line_search_(
          ctx,
          [&](Tbase sigma) {
            Vector<Tbase> d = preconditioned_sd_direction_(ctx, sigma);
            apply_lbfgs_correction_(d, ctx);
            return d;
          },
          d_accepted,
          t_accepted,
          "L-BFGS");

      if(success) {
        // s is the displacement the step actually made, t*d, not the
        // direction d: the gradient difference y that pairs with it was
        // measured across exp(t*K), and the line search routinely
        // accepts t several orders of magnitude away from 1.
        st.pending_s = t_accepted * d_accepted;
        st.pending_g = ctx.g;
        st.history_dofs = ctx.dofs;
      } else {
        st = LBFGSState();
      }
      return success;
    }

    /// Curvature pairs for the augmented Roothaan-Hall Hessian model,
    /// projected from the orbital history onto the current rotation
    /// DOFs. Columns of S are displacements kappa_i, columns of Y the
    /// gradient differences the quasi-Newton condition pairs with
    /// them. Returns the number of pairs built.
    ///
    /// Each history entry supplies one pair relative to the reference
    /// iterate: the density difference gives the displacement and the
    /// Fock difference the gradient change it produced. Both come from
    /// matrices the history already holds, so a pair costs no Fock
    /// build. The condition H (D_i - D_0) = 2 (F_i - F_0) is exact for
    /// Hartree-Fock, where F is linear in D, and approximate for
    /// Kohn-Sham; nothing here is KS-specific, and the line search
    /// absorbs the difference by testing the step it produces.
    ///
    /// The pairs are joint over particle types and symmetry blocks --
    /// one vector spanning the whole DOF set, in the ordering
    /// build_rotation_step_context_ established -- rather than one
    /// subspace per particle. That is not a simplification: the Fock
    /// matrix of one particle type responds linearly to the density of
    /// every other, so the curvature coupling them exists only in the
    /// joint space, and one subspace, one level shift and one line
    /// search cover it.
    size_t arh_curvature_pairs_(const RotationStepContext & ctx,
                                Matrix<Tbase> & S, Matrix<Tbase> & Y,
                                size_t & skipped_incommensurate) const {
      skipped_incommensurate = 0;
      const size_t nhist = orbital_history_.size();
      const size_t nblocks = ctx.C_pseudo.size();
      if(nhist < 2) return 0;

      std::vector<Matrix<Torb>> D0(nblocks), F0(nblocks), F0_mo(nblocks);
      for(size_t b = 0; b < nblocks; b++) {
        if(ctx.C_pseudo[b].cols() == 0) continue;
        D0[b] = get_density_matrix_block(0, b);
        F0[b] = get_fock_matrix_block(0, b);
        F0_mo[b] = ctx.C_pseudo[b].adjoint() * F0[b] * ctx.C_pseudo[b];
      }

      // Coordinate layout: the rotation amplitudes the step works in,
      // followed by one occupation coordinate per orbital. The extra
      // coordinates exist only so that the curvature pairs are
      // consistent; the step itself is taken in the rotation
      // coordinates alone, arh_model_ truncating back to them.
      std::vector<size_t> occupation_offset(nblocks, 0);
      size_t n_occupation = 0;
      for(size_t b = 0; b < nblocks; b++) {
        occupation_offset[b] = n_occupation;
        if(ctx.C_pseudo[b].cols() > 0)
          n_occupation += (size_t) ctx.C_pseudo[b].cols();
      }
      const size_t n_extended = ctx.n_par + n_occupation;

      const size_t max_pairs =
        std::min(nhist - 1, (size_t) maximum_history_length_);
      S.resize(n_extended, max_pairs);
      Y.resize(n_extended, max_pairs);

      size_t npairs = 0;
      for(size_t ihist = 1; ihist < nhist && npairs < max_pairs; ihist++) {
        // An entry whose orbital count differs cannot be projected on
        // the current DOF set at all; one taken at different
        // occupations can, the occupation change being carried by the
        // extra coordinates below.
        bool commensurate = true;
        for(size_t b = 0; b < nblocks && commensurate; b++) {
          if(ctx.C_pseudo[b].cols() == 0) continue;
          if(get_orbital_occupation_block(ihist, b).size()
               != ctx.n_ref[b].size())
            commensurate = false;
        }
        if(!commensurate) {
          skipped_incommensurate++;
          continue;
        }

        // Density and Fock differences in the reference MO basis. The
        // pseudo-canonicalisation only mixes orbitals of equal
        // occupation, so D_0 is diag(n_ref) in this basis and the
        // difference is the displacement from it.
        std::vector<Matrix<Torb>> dD(nblocks), dF(nblocks);
        for(size_t b = 0; b < nblocks; b++) {
          if(ctx.C_pseudo[b].cols() == 0) continue;
          const auto & Cp = ctx.C_pseudo[b];
          dD[b] = Cp.adjoint()
            * (get_density_matrix_block(ihist, b) - D0[b]) * Cp;
          dF[b] = Cp.adjoint()
            * (get_fock_matrix_block(ihist, b) - F0[b]) * Cp;
        }

        Vector<Tbase> s = Vector<Tbase>::Zero(n_extended);
        Vector<Tbase> y = Vector<Tbase>::Zero(n_extended);
        for(size_t a = 0; a < ctx.n_dof; a++) {
          const auto & dof = ctx.dofs[a];
          size_t b = std::get<0>(dof);
          Index i = std::get<1>(dof);
          Index j = std::get<2>(dof);
          // Occupation differences of the two points. They coincide
          // unless the entry was taken at different occupations, and
          // which one belongs where is not a free choice.
          //
          // Writing U = C_0^dag C_i = exp(kappa), the density
          // difference in the reference MO basis is
          //     (U n_i U^dag)_ij ~ kappa_ij (n_i,j - n_i,i)
          // off the diagonal, carrying the occupations of the *entry*.
          // Dividing by the reference's instead scales the
          // displacement by the ratio of the two, which does not
          // vanish as the step shrinks: measured at 90% error, flat in
          // the step length, against an error linear in it for the
          // entry's.
          //
          // The gradient follows the same rule. In the reference chart
          // the derivative at the displaced point is
          // tr(F_i . dD/dx_a) with dD/dx_a = C_0 [T_a, n_i] C_0^dag,
          // so it too is weighted by the entry's occupations, and the
          // change is the difference of two differently weighted
          // terms rather than one weighting of the Fock difference.
          const Tbase dn_ref = ctx.n_ref[b](j) - ctx.n_ref[b](i);
          const auto & n_entry = get_orbital_occupation_block(ihist, b);
          const Tbase dn_entry = n_entry(j) - n_entry(i);
          if(!(std::abs(dn_entry) >= occupation_change_threshold_)) continue;

          const Torb dDij = dD[b](i, j);
          const Torb F0ij = F0_mo[b](i, j);
          const Torb Fiij = F0ij + dF[b](i, j);
          s(a) = std::real(dDij) / dn_entry;
          y(a) = 2 * (std::real(Fiij) * dn_entry - std::real(F0ij) * dn_ref);
          if(ctx.is_complex) {
            s(ctx.n_dof + a) = std::imag(dDij) / dn_entry;
            y(ctx.n_dof + a) =
              2 * (std::imag(Fiij) * dn_entry - std::imag(F0ij) * dn_ref);
          }
        }
        // Occupation coordinates. An occupation change is diagonal in
        // the reference MO basis and a rotation is off-diagonal, so
        // the two read off disjoint parts of the very same matrices:
        // the displacement is the diagonal of dD and the gradient
        // change the diagonal of dF, the derivative of the energy
        // with respect to an occupation being that orbital's energy
        // (Janak), which is what the ODA occupation gradient uses.
        //
        // Including them is what makes the pair *consistent* when the
        // occupations moved between the two entries: the Fock
        // difference responds to the whole move, so pairing it with a
        // displacement that described only the rotation would charge
        // occupation-driven curvature to rotation directions.
        //
        // The check on all this is that the extended inner product is
        // s.y = tr(dD dF) in the reference MO basis, with off-diagonal
        // pairs counted twice and the diagonal once -- which is where
        // the factor of 2 on the rotation coordinates comes from.
        for(size_t b = 0; b < nblocks; b++) {
          if(ctx.C_pseudo[b].cols() == 0) continue;
          for(Index k = 0; k < ctx.C_pseudo[b].cols(); k++) {
            const size_t idx = ctx.n_par + occupation_offset[b] + (size_t) k;
            s(idx) = std::real(dD[b](k, k));
            y(idx) = std::real(dF[b](k, k));
          }
        }

        if(!s.allFinite() || !y.allFinite()) continue;
        if(!(s.norm() > 0)) continue;
        S.col(npairs) = s;
        Y.col(npairs) = y;
        npairs++;
      }
      S.conservativeResize(n_extended, npairs);
      Y.conservativeResize(n_extended, npairs);
      if(npairs > 0 && n_occupation > 0)
        rescale_occupation_coordinates_(ctx.n_par, S, Y);
      return npairs;
    }

    /// Put the occupation coordinates on the same footing as the
    /// rotation ones before the directions are orthonormalised.
    ///
    /// The two are measured in one Euclidean metric, so their relative
    /// scale decides which combinations of history entries form the
    /// basis the model is fitted in. Left alone it is an arbitrary
    /// choice, and not a harmless one: sweeping a fixed weight over
    /// four decades moves oxygen's Fock count by up to a factor of
    /// two, and the best fixed value differs between M = 1 and M = 3,
    /// so there is no constant to pick.
    ///
    /// What there is, is a scale the pairs themselves report. For a
    /// block of coordinates the ratio
    ///     c = sum_i s_i . y_i / sum_i |s_i|^2
    /// is its average curvature, in energy per squared displacement,
    /// and is exactly the quantity a metric ought to equalise. Scaling
    /// the occupation coordinates by sqrt(c_occupation / c_rotation)
    /// leaves both blocks at the same average curvature, so the
    /// singular values being compared measure the same thing.
    ///
    /// This is a change of coordinates rather than a reweighting: the
    /// displacement is multiplied and the paired gradient change
    /// divided, so the pairing s . y = tr(dD dF) -- the one quantity
    /// that does not depend on the metric -- is untouched, and so is
    /// the rotation-rotation block the step is taken in.
    ///
    /// Falls back to leaving the coordinates alone when either block
    /// reports non-positive curvature, which is what a history that
    /// has not yet seen a minimum looks like.
    void rescale_occupation_coordinates_(size_t n_rotation,
                                         Matrix<Tbase> & S,
                                         Matrix<Tbase> & Y) const {
      Tbase scale = arh_occupation_scale_;
      if(!(scale > 0)) {
        const Index n_occ = S.rows() - (Index) n_rotation;
        const auto S_rot = S.topRows((Index) n_rotation);
        const auto Y_rot = Y.topRows((Index) n_rotation);
        const auto S_occ = S.bottomRows(n_occ);
        const auto Y_occ = Y.bottomRows(n_occ);
        const Tbase norm_rot = S_rot.squaredNorm();
        const Tbase norm_occ = S_occ.squaredNorm();
        if(!(norm_rot > 0) || !(norm_occ > 0)) return;
        const Tbase curvature_rot =
          S_rot.cwiseProduct(Y_rot).sum() / norm_rot;
        const Tbase curvature_occ =
          S_occ.cwiseProduct(Y_occ).sum() / norm_occ;
        if(!(curvature_rot > 0) || !(curvature_occ > 0)) return;
        scale = std::sqrt(curvature_occ / curvature_rot);
        if(!std::isfinite((double) scale) || !(scale > 0)) return;
        log_(5, "ARH: occupation coordinates scaled by %e"
                " (curvature %e against %e).\n", (double) scale,
             (double) curvature_occ, (double) curvature_rot);
      }
      const Index n_occ = S.rows() - (Index) n_rotation;
      S.bottomRows(n_occ) *= scale;
      Y.bottomRows(n_occ) /= scale;
    }

    /// Low-rank factors of the ARH Hessian model. Returns false when
    /// the history carries no usable curvature, in which case the
    /// caller keeps the plain preconditioned-SD direction.
    ///
    /// The energy Hessian in the rotation parameters has two pieces,
    ///     d2E/dx_a dx_b = tr(dD/dx_a . dF/dD . dD/dx_b)
    ///                   + tr(F . d2D/dx_a dx_b),
    /// and they are supplied by different things. The second is the
    /// Roothaan-Hall diagonal Omega, exactly: along one rotation the
    /// energy of a one-electron model is
    /// const + (n_j - n_i)(eps_i - eps_j) sin^2 x, whose curvature at
    /// the origin is the 2 (eps_i - eps_j)(n_j - n_i) that ctx.h
    /// already holds. The first is what the quasi-Newton condition
    /// measures, F being the derivative of the energy with respect to
    /// the density -- so the history supplies the piece Omega omits
    /// rather than a replacement for it, and the two are added:
    ///     H Q = Omega Q + Y.
    ///
    /// Hence the model, exact on the span of the retained history
    /// directions and equal to Omega on the orthogonal complement:
    ///     H = Omega + W Q^T + Q W^T - Q B Q^T,
    /// with orthonormal Q, W the secant matrix and B = sym(Q^T W).
    /// Applying it to Q returns Omega Q + W exactly when Q^T W is
    /// symmetric, which holds for a true Hessian because Q^T W is then
    /// a congruence of one. It is only approximately symmetric here,
    /// and the residual is quasi-Newton noise rather than curvature,
    /// so it is symmetrised deliberately: the Woodbury solve below
    /// assumes a symmetric correction, and an asymmetric model handed
    /// to it would be neither the model intended nor a Hessian.
    ///
    /// Rationale for the split: the RH Hessian is accurate except
    /// between near-degenerate orbitals, and the density-difference
    /// history spans primarily those directions -- so the subspace
    /// where the model is made exact is the subspace where the
    /// diagonal is worst.
    bool arh_model_(const RotationStepContext & ctx, Matrix<Tbase> & Q,
                    Matrix<Tbase> & W, Matrix<Tbase> & B) const {
      Matrix<Tbase> S, Y;
      size_t skipped = 0;
      const size_t npairs = arh_curvature_pairs_(ctx, S, Y, skipped);
      if(skipped > 0)
        log_(5, "ARH: %zu history entries skipped, their orbital counts"
                " differing from the current ones.\n", skipped);
      if(npairs == 0) return false;

      // Orthonormalise through the Gram matrix rather than by an SVD
      // of the tall factor: S^T S is npairs x npairs with npairs at
      // most the history depth, so this is one small symmetric
      // eigendecomposition instead of a decomposition of an
      // n_par x npairs matrix. Squaring the condition number is
      // acceptable precisely because the directions that squaring
      // hurts are the ones being dropped.
      const Matrix<Tbase> gram = S.transpose() * S;
      Eigen::SelfAdjointEigenSolver<Matrix<Tbase>> es(gram);
      const Vector<Tbase> ev = es.eigenvalues();
      if(ev.size() == 0) return false;
      const Tbase ev_max = ev.maxCoeff();
      if(!(ev_max > 0)) return false;
      // Eigenvalues of S^T S are squared singular values, so a
      // relative cutoff on the singular values squares here too.
      const Tbase cut =
        arh_subspace_threshold_ * arh_subspace_threshold_ * ev_max;

      std::vector<Index> keep;
      for(Index k = 0; k < ev.size(); k++)
        if(ev(k) > cut) keep.push_back(k);
      if(keep.empty()) return false;

      const Index nkeep = (Index) keep.size();
      Matrix<Tbase> scale(npairs, nkeep);
      for(Index c = 0; c < nkeep; c++)
        scale.col(c) =
          es.eigenvectors().col(keep[c]) / std::sqrt(ev(keep[c]));
      // S = Q A with A = diag(sigma) V^T, so Q = S V diag(1/sigma) and
      // the secant data carries over to the orthonormal basis with the
      // same factor. When directions have been dropped A is not square
      // and A^+ makes W the least-squares fit to the curvature the
      // retained directions actually carry.
      // Orthonormalise and impose the secant in the *extended* space,
      // so that a history entry whose occupations moved contributes
      // the curvature it actually carries rather than a rotation-only
      // caricature of it.
      const Matrix<Tbase> Q_extended = S * scale;
      const Matrix<Tbase> W_extended = Y * scale;
      const Matrix<Tbase> QtW = Q_extended.transpose() * W_extended;
      B = Tbase(0.5) * (QtW + QtW.transpose());

      // The step is taken in the rotation coordinates, so what it
      // needs is the rotation-rotation block of the model. For
      // H = Omega + W Q^T + Q W^T - Q B Q^T that block is obtained by
      // restricting Q and W to their rotation rows and leaving B --
      // built from the full inner products, and therefore carrying
      // what the occupation coordinates contributed -- alone.
      Q = Q_extended.topRows(ctx.n_par);
      W = W_extended.topRows(ctx.n_par);

      log_(5, "ARH: %zu curvature pairs, %i subspace directions retained.\n",
           npairs, (int) nkeep);
      return true;
    }

    /// ARH search direction at level shift sigma: the solution of
    ///     (H_ARH + sigma) d = -g
    /// by the Woodbury identity. Writing the correction as U M U^T
    /// with U = [Q W] and M = [[-B, I], [I, 0]] costs one diagonal
    /// division, two n_par x 2k products and one 2k x 2k solve -- no
    /// Krylov loop, and nothing remotely the size of a Fock build.
    /// M^{-1} = [[0, I], [I, B]] is written down rather than computed,
    /// so B is never inverted and may be singular.
    ///
    /// With an empty subspace this reduces exactly to
    /// preconditioned_sd_direction_, which is what makes the model a
    /// refinement of the existing step rather than a replacement.
    Vector<Tbase> arh_direction_(const RotationStepContext & ctx, Tbase sigma,
                                 const Matrix<Tbase> & Q,
                                 const Matrix<Tbase> & W,
                                 const Matrix<Tbase> & B) const {
      Vector<Tbase> Dinv(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        Dinv(k) = Tbase(1) / (sigma + std::max(Tbase(0), ctx.h(k)));
      const Vector<Tbase> Dig = Dinv.cwiseProduct(ctx.g);

      const Index nk = Q.cols();
      if(nk == 0) return -Dig;

      Matrix<Tbase> U(ctx.n_par, 2 * nk);
      U.leftCols(nk) = Q;
      U.rightCols(nk) = W;
      Matrix<Tbase> Minv = Matrix<Tbase>::Zero(2 * nk, 2 * nk);
      Minv.topRightCorner(nk, nk).setIdentity();
      Minv.bottomLeftCorner(nk, nk).setIdentity();
      Minv.bottomRightCorner(nk, nk) = B;

      const Matrix<Tbase> DiU = Dinv.asDiagonal() * U;
      const Matrix<Tbase> core = Minv + U.transpose() * DiU;
      const Vector<Tbase> w = core.partialPivLu().solve(U.transpose() * Dig);
      if(!w.allFinite()) return -Dig;
      return -Dig + DiU * w;
    }

    /// Replace the preconditioned-SD direction d with the ARH
    /// direction when the latter is the better trade. Same test and
    /// same threshold as apply_lbfgs_correction_: descent rate per
    /// unit step length, d.g/|d|, which is what the line search along
    /// exp(t K) actually gets to spend. A quasi-Newton model built
    /// from first-order density differences can be wrong, and the
    /// guard costs two dot products.
    void apply_arh_correction_(Vector<Tbase> & d,
                               const RotationStepContext & ctx, Tbase sigma,
                               const Matrix<Tbase> & Q,
                               const Matrix<Tbase> & W,
                               const Matrix<Tbase> & B) const {
      const Vector<Tbase> d_arh = arh_direction_(ctx, sigma, Q, W, B);
      const Tbase norm_arh = d_arh.norm();
      if(!(norm_arh > 0) || !d_arh.allFinite()) return;
      const Tbase rate_arh = -d_arh.dot(ctx.g) / norm_arh;
      const Tbase rate_sd = -d.dot(ctx.g) / d.norm();
      if(rate_arh >= lbfgs_minimum_relative_descent_ * rate_sd) {
        log_(5, "ARH: applying augmented Roothaan-Hall direction"
                " (subspace dimension %i).\n", (int) Q.cols());
        d = d_arh;
      } else if(verbosity_ >= 5) {
        log_(5, "ARH: direction descends at %e per unit length against"
                " preconditioned SD's %e, keeping SD.\n",
             (double) (rate_arh), (double) (rate_sd));
      }
    }

    /// Augmented Roothaan-Hall step on the orbital rotations. Sibling
    /// of lbfgs_step and scaled_steepest_descent_step: same context,
    /// same line search, and the same fall back to preconditioned SD
    /// when the curvature model has nothing to add.
    ///
    /// Unlike L-BFGS it carries no state between calls. The curvature
    /// is rebuilt from orbital_history_ every step, so an entry that
    /// another method contributed counts as much as one this step
    /// made, a rejected trial still leaves its (D, F) pair behind as
    /// free curvature, and there is nothing to invalidate when the
    /// iterate is relocated by an ODA or extrapolation step.
    bool arh_step() {
      RotationStepContext ctx;
      if(!build_rotation_step_context_(ctx)) return false;

      Matrix<Tbase> Q, W, B;
      const bool have_model = arh_model_(ctx, Q, W, B);
      if(!have_model)
        log_(5, "ARH: no usable curvature in the history, taking a"
                " preconditioned steepest-descent step.\n");

      Vector<Tbase> d_accepted;
      Tbase t_accepted = 0;
      return sigma_line_search_(
          ctx,
          [&](Tbase sigma) {
            Vector<Tbase> d = preconditioned_sd_direction_(ctx, sigma);
            if(have_model)
              apply_arh_correction_(d, ctx, sigma, Q, W, B);
            return d;
          },
          d_accepted,
          t_accepted,
          "ARH");
    }

    /// Second-order estimate of the energy the orbitals would gain by
    /// relaxing at the iterate the solver currently holds, without
    /// relaxing them.
    ///
    /// A quadratic model along the rotations is minimised at
    /// d = -(H + sigma)^-1 g, where it lowers the energy by
    ///     dE = -1/2 g^T (H + sigma)^-1 g,
    /// which is what this returns (a non-positive number). With the
    /// Roothaan-Hall diagonal for H that is the familiar second-order
    /// expression: semicanonicalise, then sum the squared off-diagonal
    /// Fock elements over their orbital-energy denominators,
    ///     dE = - sum_ij Re(F_ij)^2 (n_j - n_i) / (eps_i - eps_j),
    /// the occupation weights coming from the gradient and the
    /// denominators from the diagonal Hessian. When the ARH curvature
    /// model has something to say, its H is used in place of the
    /// diagonal, which is strictly better where the history has
    /// explored the directions concerned.
    ///
    /// The level shift is kept in the denominator deliberately. It
    /// damps the estimate toward zero on the near-degenerate pairs
    /// where second-order perturbation theory is least reliable,
    /// which is the conservative direction: understating the gain
    /// costs a relaxation that turns out worthwhile, overstating it
    /// sends the search to a point that does not exist.
    ///
    /// Cost is one gradient contraction and, with the model, one small
    /// Woodbury solve -- no Fock build, no relaxation.
    Tbase perturbative_relaxation_energy_() const {
      RotationStepContext ctx;
      if(!const_cast<SCFSolver *>(this)->build_rotation_step_context_(ctx))
        return Tbase(0);
      if(ctx.n_par == 0) return Tbase(0);

      Matrix<Tbase> Q, W, B;
      Vector<Tbase> step;
      const bool have_model = arh_model_(ctx, Q, W, B);
      if(!have_model
         || !solve_rotation_block_(ctx, initial_level_shift_, Q, W, B,
                                   ctx.g, step)) {
        // Diagonal fallback: the Roothaan-Hall denominators alone.
        step.resize(ctx.n_par);
        for(size_t k = 0; k < ctx.n_par; k++)
          step(k) = ctx.g(k)
                    / (initial_level_shift_ + std::max(Tbase(0), ctx.h(k)));
      }

      // Decline to estimate where the model does not reach.
      //
      // This is second order in the rotation, so it is only worth
      // anything while the step it predicts is small enough for a
      // quadratic to describe the surface. The callers do not all
      // respect that: the polytope search asks for an estimate at
      // skeleton *vertices*, which can sit a hartree above the relaxed
      // point, and the orbitals there are as far from stationary as
      // they get in the whole calculation. What comes back is not a
      // poor estimate but a meaningless one, and it is then differenced
      // to build a curvature.
      //
      // Gating on the predicted step rather than on the gradient is
      // the direct condition -- a large gradient against a
      // correspondingly large curvature still gives a short, honest
      // step -- and returning zero leaves the caller with the
      // uncorrected energy, which is a quantity it can trust. The
      // same substitution, made for the occupation curvature, is what
      // stopped that one coming out with the wrong sign.
      const Tbase longest = step.cwiseAbs().maxCoeff();
      if(!(longest <= perturbative_relaxation_max_step_)) {
        log_(10, "Perturbative relaxation: predicted step %e exceeds %e;"
                 " declining to estimate.\n", (double) longest,
             (double) perturbative_relaxation_max_step_.get());
        return Tbase(0);
      }

      const Tbase lowering = Tbase(-0.5) * ctx.g.dot(step);
      if(!std::isfinite((double) lowering) || lowering > 0) return Tbase(0);
      return lowering;
    }

    /// Solve (H_kappa,kappa + sigma) x = rhs for the ARH model's
    /// rotation block, by the same Woodbury identity arh_direction_
    /// uses. Factored out because the Schur complement needs the
    /// solve applied to something other than the gradient.
    bool solve_rotation_block_(const RotationStepContext & ctx, Tbase sigma,
                               const Matrix<Tbase> & Q,
                               const Matrix<Tbase> & W,
                               const Matrix<Tbase> & B,
                               const Vector<Tbase> & rhs,
                               Vector<Tbase> & solution) const {
      Vector<Tbase> Dinv(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        Dinv(k) = Tbase(1) / (sigma + std::max(Tbase(0), ctx.h(k)));
      const Vector<Tbase> Dirhs = Dinv.cwiseProduct(rhs);

      const Index nk = Q.cols();
      if(nk == 0) { solution = Dirhs; return true; }

      Matrix<Tbase> U(ctx.n_par, 2 * nk);
      U.leftCols(nk) = Q;
      U.rightCols(nk) = W;
      Matrix<Tbase> Minv = Matrix<Tbase>::Zero(2 * nk, 2 * nk);
      Minv.topRightCorner(nk, nk).setIdentity();
      Minv.bottomLeftCorner(nk, nk).setIdentity();
      Minv.bottomRightCorner(nk, nk) = B;

      const Matrix<Tbase> DiU = Dinv.asDiagonal() * U;
      const Matrix<Tbase> core = Minv + U.transpose() * DiU;
      const Vector<Tbase> w = core.partialPivLu().solve(U.transpose() * Dirhs);
      if(!w.allFinite()) return false;
      solution = Dirhs - DiU * w;
      return solution.allFinite();
    }

    /// Take the orbital-rotation step the method mix selected.
    /// Exactly one of CG, L-BFGS and ARH is enabled, the parser having
    /// rejected a request for more than one.
    bool orbital_rotation_step_(const AllowedMethods & allowed) {
      if(allowed.arh) return arh_step();
      return allowed.lbfgs ? lbfgs_step() : scaled_steepest_descent_step();
    }

    /// Display name of the selected orbital-rotation step.
    static const char * orbital_rotation_name_(const AllowedMethods & allowed) {
      if(allowed.arh) return "ARH";
      return allowed.lbfgs ? "L-BFGS" : "Scaled steepest descent";
    }

    /// Token reported to the iteration callback for the selected
    /// orbital-rotation step.
    static const char * orbital_rotation_token_(const AllowedMethods & allowed) {
      if(allowed.arh) return "ARH";
      return allowed.lbfgs ? "LBFGS" : "CG";
    }

    /// Evaluate the Fock builder on a batch of densities. Dispatches
    /// to batched_fock_builder_ when set, otherwise loops over
    /// fock_builder_. Either way, number_of_fock_evaluations_ is
    /// incremented once per density.
    std::vector<FockBuilderReturn<Torb, Tbase>>
    evaluate_batch_(const std::vector<DensityMatrix<Torb, Tbase>> & densities) {
      if(batched_fock_builder_) {
        auto results = batched_fock_builder_(densities);
        if(results.size() != densities.size()) {
          std::ostringstream oss;
          oss << "Batched Fock builder returned " << results.size()
              << " entries for " << densities.size() << " densities.\n";
          throw std::logic_error(oss.str());
        }
        number_of_fock_evaluations_ += (int) densities.size();
        return results;
      }
      std::vector<FockBuilderReturn<Torb, Tbase>> results;
      results.reserve(densities.size());
      for(const auto & dm: densities) {
        results.push_back(fock_builder_(dm));
        number_of_fock_evaluations_++;
      }
      return results;
    }

    /// List of occupied orbitals
    std::vector<IndexVector> occupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> occ_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        occ_idx[l]=find_indices_where(occupations[l], [this](Tbase v){ return v >= occupied_threshold_; });
      }
      return occ_idx;
    }

    /// List of occupied orbitals
    std::vector<IndexVector> unoccupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> virt_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        virt_idx[l]=find_indices_where(occupations[l], [this](Tbase v){ return v < occupied_threshold_; });
      }
      return virt_idx;
    }

    /// True iff every occupation in the current iterate is within
    /// occupation_change_threshold_ of an integer (i.e. the iterate
    /// sits at a vertex of the skeleton polytope). Used by the SCF
    /// state machine to decide whether to hand control from ODA back
    /// to DIIS: fractional occupations need orbital relaxation through
    /// CG before DIIS can be trusted; integer occupations are stable
    /// targets that DIIS can extrapolate freely.
    /// Count degenerate orbital-energy clusters that lie inside the
    /// Aufbau-occupied window AND carry a non-trivial occupation
    /// difference, summed across blocks. Each such cluster contributes
    /// 1, regardless of how many orbitals or active rotation pairs it
    /// contains; this matches the preconditioned-CG convergence bound,
    /// which is the number of distinct eigenvalues of P^{-1} H, not
    /// the dimension of the bad subspace. With the wrong-sign-safe
    /// preconditioner all rotation pairs inside one near-degenerate
    /// cluster collapse to a single preconditioned eigenvalue (= sigma),
    /// so they need at most one CG iteration to relax as a group.
    ///
    /// The Aufbau window is set, per block, by Aufbau-filling the
    /// current F^MO diagonals up to ``number_of_particles_`` and taking
    /// the highest such orbital's energy plus a
    /// ``optimal_damping_degeneracy_threshold_`` margin. Clusters
    /// sitting entirely above that edge cannot participate in the next
    /// occupation update and are dropped.
    ///
    /// Clusters come from ``degenerate_cluster_end_``, the same walk
    /// ``optimal_damping_step`` uses to build its skeleton sets, so
    /// the burst is sized for exactly the clusters ODA created.
    /// A cluster contributes 1 iff its lowest energy sits at or below
    /// the window edge, it contains at least two orbitals, and the
    /// occupation spread inside it is at least
    /// ``occupation_change_threshold_``.
    /// Pseudo-canonicalise the orbitals C_ref against the Fock F_ref,
    /// diagonalising F within each equal-occupation sub-block (defined
    /// by n_ref and ``occupation_change_threshold_``). Rotating within
    /// an equal-occupation subset leaves the density invariant, so
    /// the pseudo orbitals still represent the same density and Fock;
    /// the diagonal entries of C_pseudo^dag F C_pseudo on the
    /// differently-occupied directions are canonical-orbital energy
    /// estimates and are returned in eps_out. Used by the rotation
    /// step, the active-rotation count, and the post-ODA burst-watch
    /// tripwires.
    void pseudo_canonicalise_(const Orbitals<Torb> & C_ref,
                              const FockMatrix<Torb> & F_ref,
                              const OrbitalOccupations<Tbase> & n_ref,
                              Orbitals<Torb> & C_pseudo,
                              OrbitalEnergies<Tbase> & eps_out) const {
      size_t nblocks = C_ref.size();
      C_pseudo.assign(nblocks, Matrix<Torb>());
      eps_out.assign(nblocks, Vector<Tbase>());
      for(size_t b = 0; b < nblocks; b++) {
        if(empty_block(b) || C_ref[b].cols() == 0) {
          C_pseudo[b] = C_ref[b];
          eps_out[b].resize(0);
          continue;
        }
        Index n_b = C_ref[b].cols();
        Matrix<Torb> F_MO = C_ref[b].adjoint() * F_ref[b] * C_ref[b];
        Matrix<Torb> U = Matrix<Torb>::Identity(n_b, n_b);
        eps_out[b] = Vector<Tbase>::Zero(n_b);
        std::vector<bool> used(n_b, false);
        for(Index i = 0; i < n_b; i++) {
          if(used[i]) continue;
          std::vector<Index> grp = {i};
          used[i] = true;
          for(Index j = i + 1; j < n_b; j++)
            if(!used[j] &&
               std::abs(n_ref[b](i) - n_ref[b](j)) < occupation_change_threshold_) {
              grp.push_back(j);
              used[j] = true;
            }
          IndexVector idx(grp.size());
          for(size_t k = 0; k < grp.size(); k++) idx(k) = grp[k];
          Matrix<Torb> F_sub(idx.size(), idx.size());
          for(Index r = 0; r < idx.size(); r++)
            for(Index c = 0; c < idx.size(); c++)
              F_sub(r, c) = F_MO(idx(r), idx(c));
          // Enforce exact Hermiticity to silence eig_sym roundoff
          // warnings; the unsymmetric residual is O(eps) for an
          // analytically Hermitian operator.
          F_sub = Tbase(1)/Tbase(2) * (F_sub + F_sub.adjoint().eval());
          Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(F_sub);
          Vector<Tbase> eps_sub = es.eigenvalues();
          Matrix<Torb> U_sub = es.eigenvectors();
          for(Index k = 0; k < idx.size(); k++) {
            eps_out[b](idx(k)) = eps_sub(k);
            for(Index l = 0; l < idx.size(); l++)
              U(idx(l), idx(k)) = U_sub(l, k);
          }
        }
        C_pseudo[b] = C_ref[b] * U;
      }
    }

    size_t compute_active_rotation_count() const {
      const auto C  = get_orbitals();
      const auto n  = get_orbital_occupations();
      const auto F  = get_fock_matrix();
      Orbitals<Torb> C_pseudo_unused;
      OrbitalEnergies<Tbase> eps;
      pseudo_canonicalise_(C, F, n, C_pseudo_unused, eps);

      // Aufbau-fill these energies and take the upper window edge per block.
      auto aufbau = update_occupations(eps);
      const Tbase inf = std::numeric_limits<Tbase>::infinity();
      Vector<Tbase> window_edge(C.size());
      window_edge.setConstant(-inf);
      for(size_t b = 0; b < C.size(); b++) {
        if(eps[b].size() == 0) continue;
        Tbase max_occ_eps = -inf;
        for(Index i = 0; i < eps[b].size(); i++)
          if(aufbau[b](i) > occupation_change_threshold_ && eps[b](i) > max_occ_eps)
            max_occ_eps = eps[b](i);
        if(std::isfinite(max_occ_eps))
          window_edge(b) = max_occ_eps + optimal_damping_degeneracy_threshold_;
      }

      size_t total = 0;
      for(size_t b = 0; b < C.size(); b++) {
        if(empty_block(b) || C[b].cols() == 0) continue;
        Index n_b = C[b].cols();
        Tbase edge = window_edge(b);

        // Sort orbital indices by energy (ascending).
        std::vector<size_t> order(n_b);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t bb) { return eps[b](a) < eps[b](bb); });

        // Walk sorted orbitals and identify ODA-style clusters.
        size_t start = 0;
        while(start < (size_t)n_b) {
          const Tbase eps_start = eps[b](order[start]);
          const size_t end = degenerate_cluster_end_(
              start, (size_t)n_b,
              [&](size_t k) { return eps[b](order[k]); });

          // Skip clusters whose lowest energy is above the window
          // edge, and singletons, which carry no rotation pair.
          if(eps_start <= edge && end - start >= 2) {
            Tbase min_n = n[b](order[start]);
            Tbase max_n = min_n;
            for(size_t k = start + 1; k < end; k++) {
              Tbase nk = n[b](order[k]);
              if(nk < min_n) min_n = nk;
              if(nk > max_n) max_n = nk;
            }
            if(max_n - min_n >= occupation_change_threshold_)
              total++;
          }
          start = end;
        }
      }
      return total;
    }

    bool has_integer_occupations() const {
      const auto occupations = get_orbital_occupations();
      for(const auto & occ_block : occupations) {
        for(Index i = 0; i < occ_block.size(); i++) {
          Tbase n = occ_block(i);
          Tbase rounded = std::round(n);
          if(std::abs(n - rounded) >= occupation_change_threshold_)
            return false;
        }
      }
      return true;
    }

  public:
    /// Constructor
    /// Default constructor, private and used only by prototype_():
    /// every setting carries its own initialiser, so a default object
    /// describes the catalog correctly.
    SCFSolver() = default;

  public:
    SCFSolver(const IndexVector & number_of_blocks_per_particle_type, const Vector<Tbase> & maximum_occupation, const Vector<Tbase> & number_of_particles, const FockBuilder<Torb, Tbase> & fock_builder, const std::vector<std::string> & block_descriptions) : number_of_blocks_per_particle_type_(number_of_blocks_per_particle_type), maximum_occupation_(maximum_occupation), number_of_particles_(number_of_particles), fock_builder_(fock_builder), block_descriptions_(block_descriptions) {
      // Run sanity checks
      number_of_blocks_ = number_of_blocks_per_particle_type_.sum();
      if((size_t)maximum_occupation_.size() != number_of_blocks_) {
        std::ostringstream oss;
        oss << "Vector of maximum occupation is not of expected length! Got " << maximum_occupation_.size() << " elements, expected " << number_of_blocks_ << "!\n";
        throw std::logic_error(oss.str());
      }
      if(number_of_particles_.size() != number_of_blocks_per_particle_type_.size()) {
        std::ostringstream oss;
        oss << "Vector of number of particles is not of expected length! Got " << number_of_particles_.size() << " elements, expected " << number_of_blocks_per_particle_type_.transpose() << "!\n";
        throw std::logic_error(oss.str());
      }
      if(block_descriptions_.size() != number_of_blocks_) {
        std::ostringstream oss;
        oss << "Vector of block descriptions is not of expected length! Got " << block_descriptions_.size() << " elements, expected " << number_of_blocks_ << "!\n";
        throw std::logic_error(oss.str());
      }
    }

    /// Initialize the solver with a guess Fock matrix
    void initialize_with_fock(const FockMatrix<Torb> & fock_guess) {
      if(fock_guess.size() != number_of_blocks_)
        throw std::logic_error("Fed in Fock matrix does not have the required number of blocks!\n");

      // Compute orbitals
      auto diagonalized_fock = compute_orbitals(fock_guess);
      const auto & orbitals = diagonalized_fock.first;
      const auto & orbital_energies = diagonalized_fock.second;

      // Compute the occupations
      orbital_occupations_ = update_occupations(orbital_energies);
      // This routine handles the rest
      initialize_with_orbitals(orbitals, orbital_occupations_);
    }

    /// Initialize with precomputed orbitals and occupations
    void initialize_with_orbitals(const Orbitals<Torb> & orbitals, const OrbitalOccupations<Tbase> & orbital_occupations) {
      if(orbitals.size() != orbital_occupations.size())
        throw std::logic_error("Fed in orbitals and orbital occupations are not consistent!\n");
      if(orbitals.size() != number_of_blocks_)
        throw std::logic_error("Fed in orbitals and orbital occupations do not have the required number of blocks!\n");

      // Refuse a basis that cannot hold the particles it is being asked
      // to hold. Every occupation rule in the solver fills orbitals in
      // energy order until the particles run out, so with too few
      // orbitals the fill runs off the end of the list. Detecting that
      // here rather than there is the difference between naming the
      // problem and failing deep inside an occupation walk with no
      // indication of why -- and silently filling what orbitals there
      // are would be worse still, since the answer would then be a
      // system with the wrong number of particles in it.
      for(Index iparticle = 0;
          iparticle < number_of_blocks_per_particle_type_.size(); iparticle++) {
        const size_t offset = particle_block_offset(iparticle);
        Tbase capacity = 0;
        for(size_t iblock = offset;
            iblock < offset + (size_t) number_of_blocks_per_particle_type_(iparticle);
            iblock++)
          capacity += Tbase(orbitals[iblock].cols()) * maximum_occupation_(iblock);
        if(capacity < number_of_particles_(iparticle)) {
          std::ostringstream oss;
          oss << "Particle type " << iparticle << " has " << (double) capacity
              << " orbital capacity but is asked to hold "
              << (double) number_of_particles_(iparticle) << " particles!\n";
          throw std::logic_error(oss.str());
        }
      }

      orbital_history_.clear();
      clear_diis_caches_();

      // Reset the per-run state. last_oda_via_collapsed_ matters:
      // left set from an earlier run it would make the convergence-
      // time full-polytope check fire on a run that never took a
      // collapsed ODA step at all.
      // A caller that is only relocating the iterate -- moving to trial
      // occupations and back, say -- is not starting a new calculation,
      // and the builds it spends are real work that a benchmark has to
      // see. Zeroing the counter under it makes every later iteration
      // line report only the builds since the last relocation, which
      // understated the cost of the occupation refinement by more than
      // an order of magnitude before this was noticed.
      if(!relocating_iterate_)
        number_of_fock_evaluations_ = 0;
      last_oda_via_collapsed_ = false;
      last_polytope_dimension_ = 0;
      add_entry(std::make_pair(orbitals, orbital_occupations));

      // Check that dimensions are consistent
      bool consistent=true;
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        if(get_orbital_block(0,iblock).cols() != get_fock_matrix_block(0,iblock).cols()) {
          // Unconditional -- this always indicated a consistency
          // problem the caller needed to see.
          log_(0, "get_orbital_block(0,iblock).cols()=%i != get_fock_matrix_block(0,iblock).cols())=%i\n",(int) get_orbital_block(0,iblock).cols(),(int) get_fock_matrix_block(0,iblock).cols());
          consistent=false;
        }
        if(get_orbital_occupation_block(0,iblock).size() != get_fock_matrix_block(0,iblock).cols()) {
          log_(10, "get_orbital_occupation_block(0,iblock).size()=%i != get_fock_matrix_block(0,iblock).cols()=%i\n",(int) get_orbital_occupation_block(0,iblock).size(),(int) get_fock_matrix_block(0,iblock).cols());
          consistent=false;
        }
      }
      // If they are not consistent (e.g. when a read-in guess has been used)
      if(not consistent) {
        log_(5, "Fed-in orbitals are not consistent with Fock matrix, recomputing orbitals\n");

        // Diagonalize the Fock matrix we just computed
        auto new_orbitals = compute_orbitals(get_fock_matrix());
        // Determine new occupations
        auto new_occupations = update_occupations(new_orbitals.second);

        // Clear out the old history
        orbital_history_.clear();
      clear_diis_caches_();
        // and add the new entry
        add_entry(std::make_pair(new_orbitals.first, new_occupations));
      }
    }

    /// Fix the number of occupied orbitals per block
    void fixed_number_of_particles_per_block(const Vector<Tbase> & number_of_particles_per_block) {
      fixed_number_of_particles_per_block_ = number_of_particles_per_block;
    }

    // === Settings façade ==================================================
    //
    // Type-tagged set / get / options catalog. Every knob the solver
    // exposes is enumerated in options(); every writable knob is
    // reachable via set(key, value); every knob and read-only
    // diagnostic is reachable via get_real / get_int / get_string
    // according to its declared type. Unknown or wrong-type keys
    // throw std::invalid_argument.

    /// Descriptor for a single option in the catalog.
    struct OptionInfo {
      const char * key;
      const char * type;      ///< "real", "int", "string"
      bool         writable;  ///< false for read-only diagnostics
      const char * doc;       ///< one-line description
    };

  private:
    /// Look a setting up by key and check it is of the requested type.
    /// Shared by every typed getter and setter; the type parameter is
    /// the only thing that differs between them.
    template<typename T>
    Setting<T> * find_setting_(const std::string & key, const char * what) {
      for(SettingBase * s : all_settings_()) {
        if(key != s->key()) continue;
        Setting<T> * typed = dynamic_cast<Setting<T> *>(s);
        if(!typed)
          throw std::invalid_argument(std::string("SCFSolver::") + what
              + ": '" + key + "' is a " + s->type() + " setting");
        return typed;
      }
      throw std::invalid_argument(std::string("SCFSolver::") + what
          + ": unknown setting '" + key + "'");
    }

    template<typename T>
    const Setting<T> * find_setting_(const std::string & key,
                                     const char * what) const {
      return const_cast<SCFSolver *>(this)->template find_setting_<T>(key, what);
    }

    /// Shared body of the typed setters: locate, refuse read-only
    /// diagnostics, run the validator if the setting has one, store.
    template<typename T>
    void assign_setting_(const std::string & key, const T & v,
                         const char * what) {
      Setting<T> * s = find_setting_<T>(key, what);
      if(!s->writable())
        throw std::invalid_argument(std::string("SCFSolver::") + what
            + ": '" + key + "' is a read-only diagnostic");
      *s = s->hook() ? (this->*(s->hook()))(v) : v;
    }

    /// Validator for error_norm. norm() already takes the norm name as
    /// an argument, so a trivial probe rejects an unknown name without
    /// touching the stored value.
    std::string canonicalise_error_norm_(const std::string & v) const {
      Vector<Tbase> test = Vector<Tbase>::Ones(1);
      (void) norm(test, v);
      return v;
    }

    /// Source for the converged diagnostic: it has no stored
    /// value, it re-runs the convergence rule on every read.
    int converged_as_int_() const {
      return converged() ? 1 : 0;
    }

    /// Validator for methods: parse to check the tokens, store the
    /// canonical uppercase spelling.
    std::string canonicalise_methods_(const std::string & v) const {
      (void) parse_method_string(v);
      return to_upper_copy(v);
    }

  public:
    /// Enumerate every option the solver understands, in declaration
    /// order. Read straight off the settings themselves, so it cannot
    /// drift out of step with what set_* and get_* accept.
    ///
    /// Static, so callers can inspect the catalog without building a
    /// solver -- the Python layer does exactly that. The settings own
    /// their values, so describing them needs *an* object; a private
    /// default-constructed prototype supplies one. Every setting
    /// carries its own default initialiser, so the prototype has the
    /// right metadata even though its other members are empty.
    static const std::vector<OptionInfo> & options() {
      static const std::vector<OptionInfo> catalog = prototype_().build_catalog_();
      return catalog;
    }

  private:
    /// Default-constructed solver used only by options(). Never runs
    /// an SCF: its Fock builder and block layout are empty.
    static const SCFSolver & prototype_() {
      static const SCFSolver p{};   // value-initialised: settings take their own defaults
      return p;
    }

    std::vector<OptionInfo> build_catalog_() const {
      std::vector<OptionInfo> out;
      for(const SettingBase * s : all_settings_())
        out.push_back({s->key(), s->type(), s->writable(), s->doc()});
      return out;
    }

  public:

    /// Set an option, dispatching on the argument type: integral
    /// arguments go to ``set_int``, floating-point (and ``Tbase``)
    /// arguments to ``set_real``, strings to ``set_string``.
    ///
    /// These are SFINAE-constrained templates rather than plain
    /// overloads on ``(Tbase)`` and ``(int)``: with plain overloads a
    /// literal like ``1e-9`` converts to both ``int`` and a non-double
    /// ``Tbase`` at the same rank, so ``set("convergence_threshold",
    /// 1e-9)`` was ambiguous — i.e. it did not compile at all — for the
    /// ``float`` and ``_Float128`` instantiations, and ``set(key, 100u)``
    /// was ambiguous for every instantiation. Dispatching on
    /// ``is_integral`` removes the tie.
    template<typename T,
             std::enable_if_t<std::is_integral_v<T>, int> = 0>
    void set(const std::string & key, T value) {
      set_int(key, static_cast<int>(value));
    }

    template<typename T,
             std::enable_if_t<!std::is_integral_v<T> &&
                              (std::is_floating_point_v<T> ||
                               std::is_same_v<T, Tbase>), int> = 0>
    void set(const std::string & key, T value) {
      set_real(key, static_cast<Tbase>(value));
    }

    void set(const std::string & key, const std::string & value) {
      set_string(key, value);
    }

    /// String-literal overload; without it a ``const char *`` argument
    /// would not match the ``std::string`` overload any better than the
    /// numeric templates reject it, and the diagnostic would be poor.
    void set(const std::string & key, const char * value) {
      set_string(key, std::string(value));
    }


    /// Set a real-valued option.
    void set_real(const std::string & key, Tbase v) {
      assign_setting_<Tbase>(key, v, "set_real");
    }

    /// Set an integer-valued option. Bool-like settings ride here as 0/1.
    void set_int(const std::string & key, int v) {
      assign_setting_<int>(key, v, "set_int");
    }

    /// Set a string-valued option.
    void set_string(const std::string & key, const std::string & v) {
      assign_setting_<std::string>(key, v, "set_string");
    }

    /// Get a real-valued option or diagnostic.
    Tbase get_real(const std::string & key) const {
      const Setting<Tbase> * s = find_setting_<Tbase>(key, "get_real");
      return s->source() ? (this->*(s->source()))() : s->get();
    }

    /// Get an integer-valued option or diagnostic.
    int get_int(const std::string & key) const {
      const Setting<int> * s = find_setting_<int>(key, "get_int");
      return s->source() ? (this->*(s->source()))() : s->get();
    }

    /// Get a string-valued option.
    std::string get_string(const std::string & key) const {
      const Setting<std::string> * s = find_setting_<std::string>(key, "get_string");
      return s->source() ? (this->*(s->source()))() : s->get();
    }

    /// Print every catalog entry with its current value to ``os``.
    /// Read-only diagnostics that require a populated orbital history
    /// (converged; anything derived from the current Fock) print as
    /// "n/a" before the first ``initialize_with_*``.
    void print_settings(std::ostream & os = std::cout) const {
      const auto & catalog = options();
      size_t maxlen = 0;
      for (const auto & o : catalog)
        maxlen = std::max(maxlen, std::string(o.key).size());
      os << "OpenOrbitalOptimizer settings:\n";
      for (const auto & o : catalog) {
        os << "  " << std::left << std::setw((int)maxlen) << o.key << " = ";
        try {
          std::string t = o.type;
          if (t == "real") {
            os << std::scientific << std::setprecision(6) << get_real(o.key);
          } else if (t == "int") {
            os << get_int(o.key);
          } else if (t == "string") {
            os << "\"" << get_string(o.key) << "\"";
          } else {
            os << "?";
          }
        } catch (const std::exception &) {
          // Read-only diagnostic not yet available (e.g. converged
          // before initialize_with_*). Report as unavailable rather
          // than propagating -- print_settings shouldn't throw just
          // because history is empty.
          os << "n/a";
        }
        if (!o.writable) os << "  (read-only)";
        os << "\n";
      }
      os.flush();
    }

    /// Canonical citation for the library. Downstream drivers should
    /// forward this to their users; the string is deliberately kept as
    /// a single line so it wraps cleanly in log output.
    static std::string citation() {
      return "Susi Lehtola and Lori A. Burns, "
             "\"OpenOrbitalOptimizer -- a reusable open source library "
             "for self-consistent field calculations\", "
             "J. Phys. Chem. A 129, 5651 (2025). "
             "doi:10.1021/acs.jpca.5c02110";
    }

    /// Print a two-line "please cite" block to ``os``.
    static void print_citation(std::ostream & os = std::cout) {
      os << "If you use OpenOrbitalOptimizer, please cite:\n"
         << "  " << citation() << "\n";
      os.flush();
    }

    // === End settings façade ==============================================

    /// Register a batched Fock builder. When set, optimal_damping_step
    /// uses it for the axis-vertex sweep, sharing integral / grid
    /// setup across the N_par builds. The single-density fock_builder
    /// remains in use for mixed-density trials (model minimum, cubic
    /// edges, backoff scales). Passing a default-constructed
    /// std::function clears the override and restores the loop-over-
    /// fock_builder default.
    void set_batched_fock_builder(BatchedFockBuilder<Torb, Tbase> builder) {
      batched_fock_builder_ = std::move(builder);
    }

    /// Whether a batched Fock builder is registered.
    bool has_batched_fock_builder() const {
      return batched_fock_builder_ != nullptr;
    }

    /// Get the energy for the n:th entry
    Tbase get_energy(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Invalid entry!\n");
      return std::get<1>(orbital_history_[ihist]).first;
    }


    /// Density matrix difference norm
    Tbase density_matrix_difference(size_t ihist, size_t jhist) const {
      // Symmetric in (ihist, jhist); cache by ordered (idx_i, idx_j)
      // so a persistent history entry pair reuses its previous value
      // across SCF iterations.
      const auto key = sorted_pair_(get_index(ihist), get_index(jhist));
      auto it = density_diff_cache_.find(key);
      if(it != density_diff_cache_.end()) return it->second;
      Tbase diff_norm = Tbase(0);
      for(size_t iblock=0;iblock<number_of_blocks_;iblock++) {
        if(empty_block(iblock))
          continue;
        diff_norm += norm(vectorise(Matrix<Torb>(get_density_matrix_block(ihist, iblock)-get_density_matrix_block(jhist, iblock))));
      }
      density_diff_cache_[key] = diff_norm;
      return diff_norm;
    }

    /// Evaluate the norm
    Tbase norm(const Matrix<Tbase> & mat, std::string norm="") const {
      if(norm == "")
        norm=error_norm_;
      if(norm == "rms") {
        // rms isn't implemented in Armadillo for some reason
        if(mat.size() == 0)
          return 0;
        return mat.norm()/std::sqrt(Tbase(1)*mat.size());
      } else if(norm == "inf") {
        return mat.template lpNorm<Eigen::Infinity>();
      } else if(norm == "fro") {
        return mat.norm();
      } else if(norm == "1") {
        return mat.template lpNorm<1>();
      } else if(norm == "2") {
        return mat.norm();
      } else {
        throw std::logic_error("Unknown norm: " + norm);
      }
    }

    /// Make an orbital history entry, stamping it with a
    /// monotonically increasing index.
    ///
    /// The index is a per-solver member rather than a function-local
    /// static. It was originally a static, which was harmless while
    /// the index served only to order the history stack; but the DIIS
    /// caches key on it, so it is now correctness-critical that it be
    /// unique within a solver. A static is shared by every instance of
    /// a given instantiation and ``index++`` is a non-atomic
    /// read-modify-write, so two solvers driven from different threads
    /// could lose an update and hand one solver a repeated index --
    /// which would make a cache return another entry's commutator and
    /// silently corrupt the DIIS extrapolation.
    OrbitalHistoryEntry<Torb, Tbase> make_history_entry(const DensityMatrix<Torb, Tbase> & density_matrix, const FockBuilderReturn<Torb, Tbase> & fock) const {
      return std::make_tuple(density_matrix, fock, next_history_index_++);
    }

    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tbase> & density) {
      // Compute the Fock matrix
      auto fock = fock_builder_(density);
      number_of_fock_evaluations_++;

      if(verbosity_>=5) {
        auto reference_energy = orbital_history_.size()>0 ? get_energy() : Tbase(0);
        log_(5, "Evaluated energy % .10f (change from lowest %e)\n", (double) (fock.first), (double) (fock.first-reference_energy));
      }
      return add_entry(density, fock);
    }

    /// Add entry to history, return value is True if energy was lowered
    bool add_entry(const DensityMatrix<Torb, Tbase> & density, const FockBuilderReturn<Torb, Tbase> & fock) {
      // Make a pair
      orbital_history_.push_back(make_history_entry(density, fock));

      if(std::isnan(fock.first)) {
        throw std::logic_error("Got NaN total energy!\n");
      }
      if(std::isinf(fock.first)) {
        throw std::logic_error("Got +-infinite total energy!\n");
      }
      for(size_t iblock=0;iblock<fock.second.size();iblock++) {
        if(fock.second[iblock].rows()==0)
          continue;
        if(has_nan(fock.second[iblock])) {
          throw std::logic_error("Got NaN in Fock matrix!\n");
        }
        if(has_inf(fock.second[iblock])) {
          throw std::logic_error("Got +-infinity in Fock matrix!\n");
        }
      }

      if(orbital_history_.size()==1)
        // First try is a success by definition
        return true;
      else {
        // Otherwise we have to check if we lowered the energy
        Tbase new_energy = fock.first;
        Tbase old_energy = get_energy();
        bool return_value = new_energy < old_energy;

        // Now, we first sort the stack in increasing energy to get
        // the lowest energy solution at the beginning
        std::sort(orbital_history_.begin(), orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tbase> & a, const OrbitalHistoryEntry<Torb, Tbase> & b) {return std::get<1>(a).first < std::get<1>(b).first;});

        // and then the rest of the stack in decreasing iteration
        // number so that we always remove the oldest vector (lowest
        // index)
        std::sort(orbital_history_.begin()+1, orbital_history_.end(), [](const OrbitalHistoryEntry<Torb, Tbase> & a, const OrbitalHistoryEntry<Torb, Tbase> & b) {return std::get<2>(a) > std::get<2>(b);});

        if(verbosity_>=20) {
          print_history();
        }

        // Drop last entry if we are over the history length limit
        if((int) orbital_history_.size() > maximum_history_length_) {
          orbital_history_.pop_back();
          prune_diis_caches_();
        }

        return return_value;
      }
    }

    /// Print the DIIS history
    void print_history() const {
      // Unconditional (caller invokes this explicitly for diagnostics).
      log_(0, "Orbital history\n");
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++)
        log_(0, "%2i % .9f % e % i\n",(int) ihist, (double) (get_energy(ihist)), (double) (get_energy(ihist)-get_energy()), (int) get_index(ihist));
    }

    /// Reset the DIIS history
    void reset_history() {
      while(orbital_history_.size()>1)
        orbital_history_.pop_back();
      clear_diis_caches_();
    }

    /// Computes orbitals and orbital energies by diagonalizing the Fock matrix
    DiagonalizedFockMatrix<Torb,Tbase> compute_orbitals(const FockMatrix<Torb> & fock) const {
      DiagonalizedFockMatrix<Torb, Tbase> diagonalized_fock;
      // Allocate memory for orbitals and orbital energies
      diagonalized_fock.first.resize(fock.size());
      diagonalized_fock.second.resize(fock.size());

      // Diagonalize all blocks
      for(size_t iblock = 0; iblock < fock.size(); iblock++) {
        if(fock[iblock].size()==0)
          continue;
        // Symmetrize Fock matrix
        Matrix<Torb> fsymm((Tbase(1)/Tbase(2))*(fock[iblock]+fock[iblock].adjoint()));
        Eigen::SelfAdjointEigenSolver<Matrix<Torb>> es(fsymm);
        diagonalized_fock.second[iblock] = es.eigenvalues();
        diagonalized_fock.first[iblock] = es.eigenvectors();

        log_stream_(10) << block_descriptions_[iblock] + " orbital energies: " << diagonalized_fock.second[iblock].transpose() << std::endl;
        log_flush_();
      }

      return diagonalized_fock;
    }

    /// Determines the offset for the blocks of the iparticle:th particle
    Index particle_block_offset(size_t iparticle) const {
      return (iparticle>0) ? number_of_blocks_per_particle_type_.head(iparticle).sum() : 0;
    }

    /// Find the end of the near-degenerate orbital cluster starting
    /// at index ``start`` in an energy-ascending list of ``n``
    /// orbitals. The cluster is anchored on its first member: it
    /// extends while ``energy(k) - energy(start) <=
    /// optimal_damping_degeneracy_threshold_``. The return value is
    /// one past the last member, so the cluster is the half-open
    /// range ``[start, end)`` and is never empty.
    ///
    /// Anchoring on the first member rather than on the previous one
    /// is what keeps the cluster width bounded by the threshold; a
    /// pairwise-gap walk would chain arbitrarily far up a dense
    /// ladder of orbitals.
    ///
    /// This is the single definition of "degenerate group" in the
    /// solver. The ODA skeleton enumeration uses it to decide which
    /// orbitals share a fractional filling, and the active-rotation
    /// count uses it to size the post-ODA CG burst -- the latter has
    /// to size the burst *for the clusters the former created*, so
    /// the two must agree exactly, boundary included.
    template<typename EnergyAt>
    size_t degenerate_cluster_end_(size_t start, size_t n,
                                   EnergyAt && energy_at) const {
      const Tbase eps_start = energy_at(start);
      size_t end = start + 1;
      while(end < n && energy_at(end) - eps_start
                         <= optimal_damping_degeneracy_threshold_)
        end++;
      return end;
    }

    /// Collect orbital energies for a given particle type, sorted in
    /// increasing energy. Each tuple holds (energy, iblock, iorb).
    std::vector<std::tuple<Tbase, size_t, size_t>> order_orbitals_by_energy(const OrbitalEnergies<Tbase> & orbital_energies, size_t iparticle) const {
      size_t block_offset = particle_block_offset(iparticle);
      std::vector<std::tuple<Tbase, size_t, size_t>> all_energies;
      for(size_t iblock = block_offset; iblock < block_offset + (size_t)number_of_blocks_per_particle_type_(iparticle); iblock++)
        for(Index iorb = 0; iorb < orbital_energies[iblock].size(); iorb++)
          all_energies.push_back(std::make_tuple(orbital_energies[iblock](iorb), iblock, iorb));
      std::stable_sort(all_energies.begin(), all_energies.end(), [](const std::tuple<Tbase, size_t, size_t> & a, const std::tuple<Tbase, size_t, size_t> & b) {return std::get<0>(a) < std::get<0>(b);});
      return all_energies;
    }

    /// Determine number of particles in each block
    Vector<Tbase> determine_number_of_particles_by_aufbau(const OrbitalEnergies<Tbase> & orbital_energies) const {
      Vector<Tbase> number_of_particles = Vector<Tbase>::Zero(number_of_blocks_);

      // Loop over particle types
      for(Index particle_type = 0; particle_type < number_of_blocks_per_particle_type_.size(); particle_type++) {
        auto all_energies = order_orbitals_by_energy(orbital_energies, particle_type);

        // Fill the orbitals in increasing energy. This is how many
        // particles we have to place
        Tbase num_left = number_of_particles_(particle_type);
        for(auto fill_orbital : all_energies) {
          // Increase number of occupied orbitals
          auto iblock = std::get<1>(fill_orbital);
          // Compute how many particles fit this orbital
          auto fill = std::min(maximum_occupation_(iblock), num_left);
          number_of_particles(iblock) += fill;
          num_left -= fill;
          // This should be sufficently tolerant to roundoff error
          if(num_left <= 10*std::numeric_limits<Tbase>::epsilon())
            break;
        }
      }

      return number_of_particles;
    }

    /// Determines occupations based on the current orbital energies
    OrbitalOccupations<Tbase> update_occupations(const OrbitalEnergies<Tbase> & orbital_energies) const {
      if(frozen_occupations_)
        return get_orbital_occupations();

      // Number of particles per block
      Vector<Tbase> number_of_particles = ((size_t)fixed_number_of_particles_per_block_.size() == number_of_blocks_) ? fixed_number_of_particles_per_block_ : determine_number_of_particles_by_aufbau(orbital_energies);

      // Determine the number of occupied orbitals
      OrbitalOccupations<Tbase> occupations(orbital_energies.size());
      for(size_t iblock=0; iblock<orbital_energies.size(); iblock++) {
        if(orbital_energies[iblock].size()==0)
          continue;
        occupations[iblock] = Vector<Tbase>::Zero(orbital_energies[iblock].size());

        Tbase num_left = number_of_particles(iblock);
        for(Index iorb=0; iorb < occupations[iblock].size(); iorb++) {
          auto fill = std::min(maximum_occupation_(iblock), num_left);
          occupations[iblock](iorb) = fill;
          num_left -= fill;
          // This should be sufficently tolerant to roundoff error
          if(num_left <= 10*std::numeric_limits<Tbase>::epsilon())
            break;
        }
      }

      return occupations;
    }

    /// Replace the converged iterate's occupations with the Aufbau
    /// filling of the converged Fock matrix.
    ///
    /// What the SCF reports at convergence is the natural occupation
    /// vector of a *mixed* density. A mixture of densities carrying
    /// different orbitals is not idempotent shell by shell, so a
    /// nominally full shell comes out at max_occ - epsilon and
    /// orbitals well above the Fermi level carry epsilon -- even
    /// though the minimiser of a fractional-occupation energy
    /// functional is Aufbau: full below the Fermi level, zero above
    /// it, fractional only inside the degenerate cluster at it.
    ///
    /// This is one ODA step with the current density left out of the
    /// polytope. That is all it takes, because the mixing is the whole
    /// problem: every skeleton is an Aufbau filling of one common set
    /// of orbitals, so any combination of skeletons alone has those
    /// same orbitals as its natural orbitals and the combined
    /// occupation vector as its occupations, exactly. The Aufbau
    /// structure is inherited rather than imposed, and the
    /// Fermi-level fractions come from minimising the energy over the
    /// skeleton simplex rather than from a filling rule -- which
    /// matters, since that is the one place where the occupations are
    /// genuinely free.
    ///
    /// Being an ODA step, it is adopted only if it lowers the energy;
    /// a cleanup that raised it would mean the converged iterate was
    /// not the Aufbau minimiser it is reported to be, which is worth
    /// leaving visible rather than papering over.
    /// Relax the orbitals at the occupations the iterate already
    /// carries, the way the state machine does after an ODA step.
    /// Runs the same burst, and stops early once a rotation step
    /// stops descending.
    void relax_orbitals_at_fixed_occupations_(const AllowedMethods & allowed) {
      if(!allowed.orbital_rotation()) return;
      // The rotation step works at fixed occupations by construction;
      // freezing them as well keeps anything it calls from quietly
      // re-Aufbau-filling and undoing the choice being tested.
      struct FrozenGuard {
        Setting<int> * flag;
        int saved;
        ~FrozenGuard() { *flag = saved; }
      } guard{&frozen_occupations_, frozen_occupations_};
      frozen_occupations_ = 1;

      // Relax to stationarity rather than for a fixed burst. The
      // rotation step reports failure once it can no longer descend,
      // which is the natural stopping point; the cap is only there so
      // a pathological case cannot spin. A burst sized for the usual
      // post-ODA relaxation is far too short here -- the occupations
      // have just moved by a finite amount, not a converged step's
      // worth, and it is the fully relaxed energy that decides whether
      // the new occupations are worth adopting at all.
      const size_t maximum_steps = 100;
      for(size_t step = 0; step < maximum_steps; step++)
        if(!orbital_rotation_step_(allowed))
          break;
    }

    /// Axis layout of the skeleton polytope once the first skeleton of
    /// each particle is promoted to the lambda = 0 vertex: one axis per
    /// skeleton after that, and ``axis[k]`` names the (particle,
    /// skeleton) the k'th axis carries. Matches what
    /// ``optimal_damping_step_`` searches under ``exclude_reference``,
    /// so the same QP can minimise over it.
    void skeleton_axis_layout_(const SkeletonOccupations & skeletons,
                               std::vector<size_t> & particle_off,
                               std::vector<size_t> & particle_len,
                               std::vector<std::pair<size_t, size_t>> & axis) const {
      particle_off.clear();
      particle_len.clear();
      axis.clear();
      for(size_t iparticle = 0; iparticle < skeletons.size(); iparticle++) {
        const size_t ntrial = skeletons[iparticle].size();
        if(ntrial < 2) continue;
        particle_off.push_back(axis.size());
        particle_len.push_back(ntrial - 1);
        for(size_t itrial = 1; itrial < ntrial; itrial++)
          axis.emplace_back(iparticle, itrial);
      }
    }

    /// Occupations at a point of that polytope: the promoted skeleton
    /// carries the slack 1 - sum(lambda), the axes carry the rest, and
    /// a particle offering only one skeleton contributes it whole.
    OrbitalOccupations<Tbase> occupations_from_lambda_(
        const SkeletonOccupations & skeletons,
        const std::vector<std::pair<size_t, size_t>> & axis,
        const Vector<Tbase> & lambda) const {
      std::vector<Tbase> spent(skeletons.size(), Tbase(0));
      for(size_t k = 0; k < axis.size(); k++)
        spent[axis[k].first] += lambda(k);

      OrbitalOccupations<Tbase> occupations(number_of_blocks_);
      for(size_t iparticle = 0; iparticle < skeletons.size(); iparticle++) {
        if(skeletons[iparticle].empty()) continue;
        const size_t offset = particle_block_offset(iparticle);
        for(size_t iblock = 0; iblock < skeletons[iparticle][0].size(); iblock++)
          occupations[offset + iblock] =
            ((Tbase(1) - spent[iparticle]) * skeletons[iparticle][0][iblock]).eval();
      }
      for(size_t k = 0; k < axis.size(); k++) {
        const size_t iparticle = axis[k].first, itrial = axis[k].second;
        const size_t offset = particle_block_offset(iparticle);
        for(size_t iblock = 0; iblock < skeletons[iparticle][itrial].size(); iblock++)
          occupations[offset + iblock] +=
            lambda(k) * skeletons[iparticle][itrial][iblock];
      }
      return occupations;
    }

    /// Gradient of the relaxed energy with respect to those axes, at
    /// the current iterate. The one-dimensional case of this is
    /// Free of any orbital-response term: at an orbital-stationary
    /// point that term carries a factor dE/dkappa = 0, and
    /// Hellmann-Feynman is all that is left.
    Vector<Tbase> relaxed_occupation_gradient_(
        const SkeletonOccupations & skeletons,
        const std::vector<std::pair<size_t, size_t>> & axis) const {
      Orbitals<Torb> C_pseudo;
      OrbitalEnergies<Tbase> eps;
      pseudo_canonicalise_(get_orbitals(), get_fock_matrix(),
                           get_orbital_occupations(), C_pseudo, eps);
      Vector<Tbase> gradient = Vector<Tbase>::Zero(axis.size());
      for(size_t k = 0; k < axis.size(); k++) {
        const size_t iparticle = axis[k].first, itrial = axis[k].second;
        const size_t offset = particle_block_offset(iparticle);
        Tbase value = 0;
        for(size_t iblock = 0; iblock < skeletons[iparticle][itrial].size(); iblock++)
          for(Index iorb = 0; iorb < skeletons[iparticle][itrial][iblock].size(); iorb++)
            value += (skeletons[iparticle][itrial][iblock](iorb)
                      - skeletons[iparticle][0][iblock](iorb))
                     * eps[offset + iblock](iorb);
        gradient(k) = value;
      }
      return gradient;
    }

    /// The polytope point whose occupations best reproduce those the
    /// solver is standing on, in the least-squares sense.
    ///
    /// occ(lambda) is affine in lambda -- the promoted skeleton plus a
    /// combination of the differences to the others -- so the fit is a
    /// quadratic on the same simplex the energy model is minimised
    /// over, and the same QP solves it.
    Vector<Tbase> project_occupations_onto_polytope_(
        const SkeletonOccupations & skeletons,
        const std::vector<std::pair<size_t, size_t>> & axis,
        const std::vector<size_t> & particle_off,
        const std::vector<size_t> & particle_len) const {
      const size_t npars = axis.size();
      const auto current = get_orbital_occupations();
      const Vector<Tbase> zero = Vector<Tbase>::Zero(npars);
      const auto base = occupations_from_lambda_(skeletons, axis, zero);

      // Columns of A are the occupation changes each axis produces.
      std::vector<OrbitalOccupations<Tbase>> column(npars);
      for(size_t k = 0; k < npars; k++) {
        Vector<Tbase> unit = Vector<Tbase>::Zero(npars);
        unit(k) = Tbase(1);
        column[k] = occupations_from_lambda_(skeletons, axis, unit);
      }

      auto dot = [&](const OrbitalOccupations<Tbase> & u,
                     const OrbitalOccupations<Tbase> & v,
                     const OrbitalOccupations<Tbase> & u0,
                     const OrbitalOccupations<Tbase> & v0) {
        Tbase acc = 0;
        for(size_t b = 0; b < u.size(); b++)
          for(Index i = 0; i < u[b].size(); i++)
            acc += (u[b](i) - u0[b](i)) * (v[b](i) - v0[b](i));
        return acc;
      };

      Matrix<Tbase> H(npars, npars);
      Vector<Tbase> g(npars);
      for(size_t k = 0; k < npars; k++) {
        for(size_t l = 0; l < npars; l++)
          H(k, l) = Tbase(2) * dot(column[k], column[l], base, base);
        g(k) = Tbase(-2) * dot(column[k], current, base, base);
      }

      Vector<Tbase> lambda;
      Tbase value;
      std::tie(lambda, value) = solve_polytope_qp_(H, g, Tbase(0),
                                                  particle_off, particle_len);
      return lambda;
    }

    /// Minimise the relaxed energy over a skeleton polytope of any
    /// dimension, and leave the iterate at the best point found.
    ///
    /// The relaxed Hessian
    ///
    ///     H_relaxed = H_ll - H_lk H_kk^-1 H_kl
    ///
    /// is what governs where the minimum sits, and it is never formed
    /// here -- the Schur complement would need the orbital response
    /// equations the solver does not have. It does not need to be. The
    /// object being minimised is the reduced surface
    /// E_relaxed(lambda) = min_kappa E(lambda, kappa), whose gradient
    /// is free at any relaxed point, so relaxing at the lambda = 0
    /// vertex and at each axis vertex gives H_relaxed column by column
    /// as a difference of gradients:
    ///
    ///     H_relaxed e_j = grad(e_j) - grad(0).
    ///
    /// That is the whole generalisation. The energies come along for
    /// the ride and are kept as candidates. Cost is one relaxation per
    /// vertex plus one at the predicted minimum, which is why the
    /// caller caps the dimension it will attempt this at.
    Tbase relaxed_occupation_search_(const AllowedMethods & allowed,
                                     const SkeletonOccupations & skeletons,
                                     const Orbitals<Torb> & orbitals,
                                     int & fock_evaluations) {
      std::vector<size_t> particle_off, particle_len;
      std::vector<std::pair<size_t, size_t>> axis;
      skeleton_axis_layout_(skeletons, particle_off, particle_len, axis);
      const size_t npars = axis.size();

      auto sample = [&](const Vector<Tbase> & lambda) {
        fock_evaluations += number_of_fock_evaluations_;
        initialize_with_orbitals(
            orbitals, occupations_from_lambda_(skeletons, axis, lambda));
        relax_orbitals_at_fixed_occupations_(allowed);
        return get_energy();
      };

      // The same point without the relaxation: one Fock build to place
      // the orbitals at these occupations, plus the second-order
      // estimate of what relaxing them would gain. See
      // perturbative_relaxation_energy_. A relaxation here costs tens
      // of Fock builds on a transition metal in a large basis, and the
      // search only needs an ordering over candidates to decide which
      // one is worth paying for.
      auto estimate = [&](const Vector<Tbase> & lambda) {
        initialize_with_orbitals(
            orbitals, occupations_from_lambda_(skeletons, axis, lambda));
        return get_energy() + perturbative_relaxation_energy_();
      };

      // Start where the solver already is, not at an arbitrary corner.
      // The converged occupations are typically Aufbau to within the
      // residual tail, so the polytope point closest to them is a
      // fraction of a millihartree from the answer, whereas a skeleton
      // vertex can sit an entire hartree above it. Relaxing from there
      // is a correction; relaxing from a vertex is a climb, and a climb
      // that stops short is reported as the Aufbau state losing on
      // energy when it has merely not arrived.
      const Vector<Tbase> anchor_start = project_occupations_onto_polytope_(
          skeletons, axis, particle_off, particle_len);
      const Tbase E_start = sample(anchor_start);
      auto best_state = get_solution();
      Tbase best_energy = E_start;
      log_stream_(5) << "Relaxed polytope: starting from lambda = "
                     << anchor_start.transpose() << ", E = " << E_start
                     << std::endl;

      const Vector<Tbase> origin = Vector<Tbase>::Zero(npars);
      const Tbase E_origin = sample(origin);
      const Vector<Tbase> gradient = relaxed_occupation_gradient_(skeletons, axis);
      const auto origin_state = get_solution();
      if(E_origin < best_energy) {
        best_energy = E_origin;
        best_state = origin_state;
      }

      // Curvature of the relaxed energy over the polytope, by second
      // differences of the perturbatively relaxed energy.
      //
      // The obvious construction -- relax at each vertex and difference
      // the gradients -- costs one full orbital optimisation per axis,
      // tens of Fock builds each on a transition metal in a large
      // basis. The estimate costs one build per point.
      //
      // It has to go through *energies* rather than gradients.
      // relaxed_occupation_gradient_ reads the orbital energies of
      // whatever iterate the solver is standing on, so at an unrelaxed
      // point it returns the bare occupation Hessian, not the relaxed
      // one, and the two differ by exactly the orbital response the
      // search exists to account for. Second differences of the
      // corrected energy carry that response because the correction
      // does.
      Matrix<Tbase> hessian(npars, npars);
      if(perturbative_occupation_hessian_ == 0) {
        for(size_t j = 0; j < npars; j++) {
          Vector<Tbase> vertex = Vector<Tbase>::Zero(npars);
          vertex(j) = Tbase(1);
          const Tbase E_vertex = sample(vertex);
          if(E_vertex < best_energy) {
            best_energy = E_vertex;
            best_state = get_solution();
          }
          hessian.col(j) =
            relaxed_occupation_gradient_(skeletons, axis) - gradient;
        }
        hessian = ((hessian + hessian.transpose()) / Tbase(2)).eval();
        initialize_with_orbitals(origin_state.first, origin_state.second);
      } else {
      const Tbase E_pt_origin = estimate(origin);
      std::vector<Tbase> E_pt_axis(npars);
      for(size_t j = 0; j < npars; j++) {
        Vector<Tbase> vertex = Vector<Tbase>::Zero(npars);
        vertex(j) = Tbase(1);
        E_pt_axis[j] = estimate(vertex);
      }
      for(size_t j = 0; j < npars; j++) {
        for(size_t k = j; k < npars; k++) {
          Tbase mixed;
          if(j == k) {
            // A pure axis: the second difference needs a third point,
            // and the polytope's own vertex doubles as one only if it
            // stays inside. Use the midpoint instead, which always
            // does, and rescale.
            Vector<Tbase> half = Vector<Tbase>::Zero(npars);
            half(j) = Tbase(1) / Tbase(2);
            const Tbase E_half = estimate(half);
            mixed = Tbase(4) * (E_pt_axis[j] - Tbase(2) * E_half + E_pt_origin);
          } else {
            Vector<Tbase> both = Vector<Tbase>::Zero(npars);
            both(j) = Tbase(1);
            both(k) = Tbase(1);
            const Tbase E_both = estimate(both);
            mixed = E_both - E_pt_axis[j] - E_pt_axis[k] + E_pt_origin;
          }
          hessian(j, k) = mixed;
          hessian(k, j) = mixed;
        }
      }
      // Restore the solver to the relaxed origin. The estimates left
      // it wherever the last one put it, and the walk below expects
      // to start from a relaxed iterate -- but that iterate was
      // already computed above, so put it back rather than paying for
      // it twice.
      initialize_with_orbitals(origin_state.first, origin_state.second);
      }

      // Walk to the stationary point of the relaxed energy rather than
      // stopping at the model's first guess. The model is fitted from
      // finite differences across the whole polytope, so it locates the
      // minimum only to a few percent, which leaves the energy a part
      // in 1e6 high -- on a spin-restricted transition metal that is
      // the entire gain the cleanup is trying to bank. Each step
      // re-anchors the quadratic on the point just sampled, using the
      // gradient there, which costs nothing beyond the relaxation
      // already paid for.
      Vector<Tbase> anchor = origin;
      Vector<Tbase> anchor_gradient = gradient;
      Tbase anchor_energy = E_origin;
      const size_t maximum_refinements =
        (size_t) std::max(0, relaxed_occupation_refinements_.get());
      for(size_t refinement = 0; refinement < maximum_refinements; refinement++) {
        // Re-expand the model about the anchor: the QP works in
        // absolute lambda, so fold the shift into the linear and
        // constant terms.
        const Vector<Tbase> g_effective = anchor_gradient - hessian * anchor;
        const Tbase E_effective =
            anchor_energy - anchor_gradient.dot(anchor)
            + Tbase(1)/Tbase(2) * (anchor.transpose() * hessian * anchor).value();

        Vector<Tbase> lambda_star;
        Tbase model_minimum;
        std::tie(lambda_star, model_minimum) = solve_polytope_qp_(
            hessian, g_effective, E_effective, particle_off, particle_len);
        const Vector<Tbase> step = lambda_star - anchor;
        if(step.template lpNorm<Eigen::Infinity>()
             <= Tbase(100) * std::numeric_limits<Tbase>::epsilon())
          break;

        // Screen the step before paying for it. The estimate costs one
        // Fock build against the tens a relaxation costs, so a step
        // the model already expects to lose is not worth confirming.
        // The margin is the same sub-noise tolerance the refit loop
        // uses, so a step inside the arithmetic floor is not rejected
        // on the strength of an estimate.
        const Tbase E_estimated = estimate(lambda_star);
        if(E_estimated >= anchor_energy + minimum_useful_descent_()) {
          log_stream_(5) << "Relaxed polytope: lambda = "
                         << lambda_star.transpose()
                         << " estimated at E = " << E_estimated
                         << ", above the anchor; not relaxing there."
                         << std::endl;
          break;
        }

        const Tbase E_star = sample(lambda_star);
        log_stream_(5) << "Relaxed polytope: lambda = "
                       << lambda_star.transpose() << ", E = " << E_star
                       << " (estimated " << E_estimated << ")" << std::endl;
        log_occupation_iterate_("Relaxed polytope", (int) refinement,
                                E_star - anchor_energy);
        if(E_star < best_energy) {
          best_energy = E_star;
          best_state = get_solution();
        }
        // A step that does not pay is the end of the walk; the model
        // has stopped describing the surface well enough to trust.
        if(E_star >= anchor_energy - minimum_useful_descent_()) break;

        const Vector<Tbase> new_gradient =
            relaxed_occupation_gradient_(skeletons, axis);

        // Correct the curvature along the step just taken. The
        // finite-difference Hessian spans the whole polytope, so it
        // describes the surface between its vertices rather than
        // around the minimum, and stepping on it alone overshoots and
        // rings. Imposing the secant condition H s = y on the
        // direction just walked -- a symmetric rank-one update --
        // makes the next step use the curvature actually seen there.
        const Vector<Tbase> y = new_gradient - anchor_gradient;
        const Vector<Tbase> residual = y - hessian * step;
        const Tbase denominator = residual.dot(step);
        if(std::abs(denominator)
             > Tbase(1e-8) * step.norm() * residual.norm())
          hessian += (residual * residual.transpose()) / denominator;

        anchor = lambda_star;
        anchor_gradient = new_gradient;
        anchor_energy = E_star;
      }

      fock_evaluations += number_of_fock_evaluations_;
      initialize_with_orbitals(best_state.first, best_state.second);
      return best_energy;
    }

    /// Replace the converged iterate's occupations with an Aufbau
    /// filling, relaxing the orbitals at those occupations before
    /// judging whether the swap was worth making.
    ///
    /// What the SCF converges to is the natural occupation vector of a
    /// *mixed* density. A mixture of densities carrying different
    /// orbitals is not idempotent shell by shell, so a nominally full
    /// shell comes out at max_occ - epsilon and orbitals far above the
    /// Fermi level carry epsilon -- even though the minimiser of a
    /// fractional-occupation functional is Aufbau.
    ///
    /// Both halves are needed and neither suffices alone. Choosing
    /// occupations is an ODA step over the skeletons with the current
    /// density excluded, which keeps the result Aufbau: every skeleton
    /// is an Aufbau filling of one common set of orbitals, so any
    /// combination of them has those orbitals as its natural orbitals
    /// and the combined vector as its occupations, exactly. Judging
    /// that vector on orbitals relaxed for the *previous* occupations
    /// costs 1e-4 to 2e-2 Eh on a transition metal, which is more than
    /// the swap is worth, so the orbitals are relaxed at the new
    /// occupations before any energy is compared.
    ///
    /// Where the polytope is one-dimensional the two are coupled
    /// directly up to ``maximum_modelled_dimension``, see
    /// ``relaxed_occupation_search_``; beyond that they alternate,
    /// which makes this a small SCF restricted to Aufbau-occupied
    /// densities.
    ///
    /// All of it runs on a scratch iterate seeded from the converged
    /// Fock matrix, so the history holds nothing but Aufbau states and
    /// its lowest-energy entry is the best of them rather than the
    /// mixed density -- which is what makes the final comparison
    /// possible at all, the history being kept sorted by energy.
    ///
    /// The result is adopted only if it does not cost energy. Losing
    /// after relaxation would say the mixed density sits below every
    /// Aufbau-occupied state, which a variational fractional-occupation
    /// functional should not permit, so a rejection is reported rather
    /// than passed over in silence.
    /// ``must_stay_converged`` says the caller arrived here converged,
    /// and so has a gradient criterion to lose: settling the
    /// occupations perturbs the orbitals, and the cleanup trades on
    /// energy and occupations while saying nothing about the
    /// commutator. It does not veto the swap -- an occupation state
    /// orders better is not worth refusing over a gradient a few per
    /// cent short -- but it does buy the orbitals a few relaxation
    /// passes at the refined occupations, to win back what can be won.
    /// The exits that arrive unconverged have nothing to protect and
    /// skip that work.
    bool aufbau_cleanup_step(const AllowedMethods & allowed,
                             bool must_stay_converged) {
      if(frozen_occupations_) {
        log_(5, "Aufbau cleanup skipped: occupations are frozen.\n");
        return false;
      }
      log_(5, "Aufbau cleanup: choosing occupations, then relaxing at them.\n");

      // The whole cleanup relocates the iterate -- it samples trial
      // occupations, relaxes at them and moves back -- so the
      // Fock-evaluation counter has to keep running across it. Guarding
      // only the refinement left the polytope walk resetting it, and
      // the phase then reported 71 builds where it had spent about 210.
      struct RelocationGuard {
        bool * flag;
        bool saved;
        ~RelocationGuard() { *flag = saved; }
      } relocation{&relocating_iterate_, relocating_iterate_};
      relocating_iterate_ = true;

      // Everything below runs on a scratch iterate; keep what it would
      // otherwise overwrite.
      const auto saved_history = orbital_history_;
      const Tbase reference_energy = get_energy();
      const int saved_fock_evaluations = number_of_fock_evaluations_;
      const auto saved_cg_gradient = previous_orbital_gradient_;
      const auto saved_cg_direction = previous_orbital_direction_;
      const auto saved_cg_dofs = previous_orbital_dofs_;
      const auto saved_lbfgs = lbfgs_;

      // initialize_with_orbitals resets the Fock counter, so the count
      // has to be banked before each pass wipes it.
      int cleanup_fock_evaluations = 0;

      // Best Aufbau state found so far, kept explicitly: a pass can
      // land below the one before it -- once the occupations stop
      // being fractional the skeleton simplex collapses to a point and
      // the step falls back on a plain Aufbau fill, which breaks the
      // symmetry of the degenerate shell and costs far more than the
      // pass was going to win -- and the answer must not follow it
      // down.
      DensityMatrix<Torb, Tbase> best_state;
      Tbase best_energy = std::numeric_limits<Tbase>::infinity();

      // The skeleton set is established once, here, from the converged
      // Fock matrix, and then held fixed for the rest of the cleanup.
      // It describes the solution being refined -- which orbitals are
      // degenerate at it and which integer fillings of them to span --
      // and re-deriving it from each relaxed iterate destroys it: the
      // relaxation moves the cluster apart by more than the degeneracy
      // window, the walk stops recognising it, and the simplex
      // collapses to a point after the first pass.
      SkeletonOccupations skeletons;

      // Establish the skeleton set, and with it the orbitals the
      // skeletons are indexed against. Enumeration alone, so this costs
      // a diagonalisation and no Fock builds.
      auto diagonalized = compute_orbitals(get_fock_matrix());
      {
        std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>
          discard;
        SkeletonOccupations enumerated;
        optimal_damping_step_(/*force_full=*/true, /*exclude_reference=*/true,
                              discard, enumerated, /*enumerate_only=*/true);
        skeletons = std::move(enumerated);
      }

      // Nothing to clean if the polytope has no dimensions. That means
      // no degenerate cluster is fractionally filled, so the occupations
      // are already the only Aufbau filling available and no amount of
      // relaxing will change them. Worth checking before anything else:
      // the relaxation that would otherwise follow is a full orbital
      // optimisation run outside the SCF's own convergence control, and
      // on a metal with an integer-filled Fermi level it can cost a
      // large fraction of the run for a result that cannot differ.
      {
        std::vector<size_t> off, len;
        std::vector<std::pair<size_t, size_t>> axis;
        skeleton_axis_layout_(skeletons, off, len, axis);
        if(axis.empty()) {
          log_(5, "Aufbau cleanup: no fractionally filled degenerate "
                  "cluster, so the occupations are already Aufbau; "
                  "nothing to do.\n");
          return false;
        }
      }

      // Where the polytope is small enough to afford it, the coupling
      // between the occupations and the orbitals is modelled directly
      // rather than alternated around. Both branches below relax at
      // every point they sample; they differ only in the fit.
      //
      // The dimension is capped because each sample costs a full
      // relaxation -- about a hundred Fock builds on the iron atom,
      // against 576 for its entire SCF -- and the count is one per
      // vertex plus one at the model minimum. A degenerate f shell can
      // enumerate skeletons into the tens, where this would cost
      // several times the SCF it is cleaning up after; those fall back
      // to the alternation, which is cheap and merely imperfect.
      const size_t maximum_modelled_dimension = 6;
      std::vector<size_t> off_unused, len_unused;
      std::vector<std::pair<size_t, size_t>> axis;
      skeleton_axis_layout_(skeletons, off_unused, len_unused, axis);

      if(!axis.empty() && axis.size() <= maximum_modelled_dimension) {
        log_(5, "Aufbau cleanup: relaxed search over %zu occupation "
                "degrees of freedom.\n", axis.size());
        relaxed_occupation_search_(allowed, skeletons, diagonalized.first,
                                   cleanup_fock_evaluations);

      } else {
        if(!axis.empty())
          log_(5, "Aufbau cleanup: %zu occupation degrees of freedom is "
                  "more than the %zu worth relaxing at, alternating "
                  "instead.\n", axis.size(), maximum_modelled_dimension);
      const size_t maximum_passes = 8;
      for(size_t pass = 0; pass < maximum_passes; pass++) {
        // Occupations: ODA over the fixed skeleton set, the reference
        // excluded so the result stays a combination of skeletons and
        // therefore Aufbau.
        std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>
          best_aufbau;
        optimal_damping_step_(/*force_full=*/true, /*exclude_reference=*/true,
                              best_aufbau, skeletons);
        if(best_aufbau.first.first.empty()) break;

        // The first pass moves whatever it costs -- swapping the
        // occupations is the point, and the relaxation that pays for
        // it has not run yet. After that, moving somewhere worse than
        // the best Aufbau state in hand is pure loss, the relaxation
        // only climbing back to where it started.
        if(pass > 0 && best_aufbau.second.first >= best_energy) break;

        // Stand on it, whether or not it beat what we were standing on
        // -- it is about to be relaxed, and it is the relaxed energy
        // that decides. This also makes the history hold nothing but
        // Aufbau states, so its lowest-energy entry is the best of
        // those rather than the mixed density we are trying to
        // improve on.
        cleanup_fock_evaluations += number_of_fock_evaluations_;
        initialize_with_orbitals(best_aufbau.first.first,
                                 best_aufbau.first.second);
        relax_orbitals_at_fixed_occupations_(allowed);

        // Progress is measured along the Aufbau trajectory, not
        // against the mixed density we started from: the first pass is
        // expected to land above it, that being the whole reason the
        // relaxation is needed.
        const Tbase now = get_energy();
        if(now >= best_energy - minimum_useful_descent_()) break;
        best_energy = now;
        best_state = get_solution();
      }

      }

      // Stand on the best pass, not the last one.
      // Both routes above settle which orbitals are occupied and
      // relax there, walking on energy decreases and stopping when the
      // next one falls below the threshold that keeps an already
      // converged SCF from churning at the arithmetic floor. That
      // threshold is on the *energy*, which near a stationary point is
      // quadratic in the occupation displacement, so it cannot resolve
      // where inside a fractionally occupied cluster the occupations
      // belong. The conditions fermi_level_error measures are linear
      // in that displacement and settle it directly -- the same job as
      // this routine's, finished with the instrument that can see it.
      if(kkt_occupation_refinement_(allowed) && must_stay_converged
         && !converged()) {
        // The refinement settles the occupations, and moving them is
        // an orbital perturbation: the orbitals that were stationary
        // for the old division of the shell are not stationary for
        // the new one. It relaxes at every trial it takes, but hands
        // back a single history entry, so the state the caller sees
        // has never had its gradient minimised from where it now
        // stands. A few passes here recover most of it, which is
        // worth having when the caller arrived converged.
        log_(5, "Aufbau cleanup: the refined occupations leave the gradient"
                " at %e; relaxing the orbitals at them.\n",
             (double) diis_error_norm());
        for(size_t attempt = 0; attempt < 4 && !converged(); attempt++)
          relax_orbitals_at_fixed_occupations_(allowed);
        log_(5, "Aufbau cleanup: gradient now %e, %s.\n",
             (double) diis_error_norm(),
             converged() ? "converged" : "still short");
      }

      // The refinement is held to the same tolerance the walk is. It
      // is allowed to climb, but only by an amount below the scale
      // the solver treats as a real energy difference, so this
      // comparison cannot undo a move it was let make. Widening it to
      // the convergence threshold instead, to give the refinement
      // room, does undo things -- it leaves a state 7e-7 above the
      // one the walk had already found, that deficit now sitting
      // under the bar that would have restored it.
      if(!best_state.first.empty()
         && get_energy() > best_energy + minimum_useful_descent_()) {
        cleanup_fock_evaluations += number_of_fock_evaluations_;
        initialize_with_orbitals(best_state.first, best_state.second);
      }

      const Tbase aufbau_energy = get_energy();
      // The scratch passes reset the counter; the Fock builds they made
      // were real, so fold them back onto the total rather than losing
      // them.
      cleanup_fock_evaluations += number_of_fock_evaluations_;
      number_of_fock_evaluations_ =
          saved_fock_evaluations + cleanup_fock_evaluations;

      // Adopt the cleanup whenever it does not cost real energy, and
      // judge that on the energy alone.
      //
      // The occupations are what this step is for. An energy
      // difference below the convergence threshold is not evidence of
      // a preference for the mixed density -- it is smaller than the
      // scale on which the whole calculation is being decided -- so a
      // rise inside it is not a reason to refuse.
      //
      // The gradient is deliberately not consulted. Conditioning the
      // swap on keeping converged() costs more than it protects: on a
      // spin-restricted iron atom the cleanup reaches a Fermi-level
      // residual of 4.3e-7 at a gradient of 1.08e-6, eight per cent
      // over its threshold, and refusing it on that ground reports
      // the mixed density it started from instead -- whose residual
      // is 0.13. That is not a safer answer, it is a wrong one in the
      // other variable, reported as converged.
      //
      // The price is that a run can finish with the occupations right
      // and the gradient a little short. run() says so rather than
      // announcing a convergence the cleanup has spent.
      const Tbase tolerance = effective_convergence_threshold_();
      if(aufbau_energy <= reference_energy + tolerance) {
        log_(5, "Aufbau cleanup accepted, energy change %e\n",
             (double) (aufbau_energy - reference_energy));
        return true;
      }

      log_(1, "Aufbau cleanup rejected: the relaxed Aufbau state lies %e Eh "
              "above the converged density, whose mixed occupations are "
              "reported instead.\n",
           (double) (aufbau_energy - reference_energy));
      orbital_history_ = saved_history;
      clear_diis_caches_();
      previous_orbital_gradient_ = saved_cg_gradient;
      previous_orbital_direction_ = saved_cg_direction;
      previous_orbital_dofs_ = saved_cg_dofs;
      lbfgs_ = saved_lbfgs;
      return false;
    }

    /// How far the occupations are from carrying the requested number
    /// of particles: the largest ``|sum(n) - N|`` over the particle
    /// types.
    ///
    /// This is an invariant rather than a convergence measure. Every
    /// step forms densities out of ingredients that already carry the
    /// right particle number, so the only way to lose any is for
    /// something to discard it, and iterating will not bring it back.
    /// It is reported so that a run cannot quietly finish carrying a
    /// micro-electron of error, but deliberately does not gate
    /// ``converged()``: a solver that cannot converge is worse than
    /// one that tells you its answer is off.
    Tbase particle_number_error() const {
      if(orbital_history_.empty()) return 0;
      const auto occupations = get_orbital_occupations();
      Tbase worst = 0;
      for(Index iparticle = 0;
          iparticle < number_of_blocks_per_particle_type_.size(); iparticle++) {
        const size_t offset = particle_block_offset(iparticle);
        Tbase sum = 0;
        for(size_t iblock = offset;
            iblock < offset + (size_t) number_of_blocks_per_particle_type_(iparticle);
            iblock++)
          sum += occupations[iblock].sum();
        worst = std::max(worst,
                         std::abs(sum - number_of_particles_(iparticle)));
      }
      return worst;
    }

    /// How far the occupations are from Aufbau: the largest occupation
    /// sitting above the Fermi level, or missing from below it,
    /// measured against the Aufbau filling of the current orbital
    /// energies.
    ///
    /// Orbitals inside the degenerate cluster the Fermi level lands in
    /// are exempt, since that is where fractional occupation is
    /// legitimate; the window is the same
    /// ``optimal_damping_degeneracy_threshold_`` the ODA skeleton walk
    /// uses. The comparison is against the whole particle type's
    /// energy ordering rather than each block's, because the Fermi
    /// level is filled across blocks and its degeneracies routinely
    /// span them.
    ///
    /// Unlike the particle-number error this *is* a convergence
    /// measure: the iterate is a mixed density whose occupations are
    /// only Aufbau once the mixing has collapsed onto the minimiser,
    /// so this falls as the SCF converges. Returns 0 when the
    /// occupations are not the solver's to choose.
    /// Set while a routine is moving the iterate to trial points and
    /// back rather than starting a fresh calculation. Consulted by
    /// initialize_with_orbitals, which otherwise resets the
    /// Fock-evaluation counter.
    bool relocating_iterate_ = false;

    /// Everything the history-based methods carry between steps.
    ///
    /// initialize_with_orbitals clears the orbital history and the
    /// DIIS caches and starts again from the state it is handed, which
    /// is what a routine wanting to move the iterate has to use. At
    /// the end of a run that costs nothing. Called from inside the SCF
    /// it leaves the state machine running on extrapolation and
    /// curvature histories that no longer describe where the iterate
    /// has been -- measured, that segfaulted on the next rotation
    /// step. Saving and restoring around such a routine keeps the
    /// relocation local to it.
    struct HistorySnapshot {
      OrbitalHistory<Torb, Tbase> history;
      size_t next_index = 0;
      std::map<size_t, std::vector<Matrix<Torb>>> commutators;
      std::map<std::pair<size_t, size_t>, Tbase> trace_DF;
      std::map<std::pair<size_t, size_t>, Tbase> diis_matrix;
      std::map<std::pair<size_t, size_t>, Tbase> density_diff;
      LBFGSState lbfgs;
      Vector<Tbase> previous_gradient;
      Vector<Tbase> previous_direction;
      std::vector<OrbitalRotation> previous_dofs;
      bool last_oda_collapsed = false;
      Tbase old_energy = 0;
    };

    HistorySnapshot save_history_() const {
      HistorySnapshot snapshot;
      snapshot.history = orbital_history_;
      snapshot.next_index = next_history_index_;
      snapshot.commutators = diis_commutator_cache_;
      snapshot.trace_DF = trace_DF_cache_;
      snapshot.diis_matrix = diis_matrix_cache_;
      snapshot.density_diff = density_diff_cache_;
      snapshot.lbfgs = lbfgs_;
      snapshot.previous_gradient = previous_orbital_gradient_;
      snapshot.previous_direction = previous_orbital_direction_;
      snapshot.previous_dofs = previous_orbital_dofs_;
      snapshot.last_oda_collapsed = last_oda_via_collapsed_;
      snapshot.old_energy = old_energy_;
      return snapshot;
    }

    void restore_history_(const HistorySnapshot & snapshot) {
      orbital_history_ = snapshot.history;
      next_history_index_ = snapshot.next_index;
      diis_commutator_cache_ = snapshot.commutators;
      trace_DF_cache_ = snapshot.trace_DF;
      diis_matrix_cache_ = snapshot.diis_matrix;
      density_diff_cache_ = snapshot.density_diff;
      lbfgs_ = snapshot.lbfgs;
      previous_orbital_gradient_ = snapshot.previous_gradient;
      previous_orbital_direction_ = snapshot.previous_direction;
      previous_orbital_dofs_ = snapshot.previous_dofs;
      last_oda_via_collapsed_ = snapshot.last_oda_collapsed;
      old_energy_ = snapshot.old_energy;
    }

    /// Drive the occupations onto the conditions fermi_level_error
    /// measures, by Newton on the residual rather than by minimising
    /// the energy.
    ///
    /// The distinction is the whole point. Every occupation move the
    /// solver otherwise makes is accepted on an energy decrease, and
    /// near a stationary point the energy is quadratic in the
    /// displacement while the residual is linear -- so an energy test
    /// cannot resolve displacements this one settles directly. On a
    /// spin-restricted iron atom the optimal-damping step stopped with
    /// a predicted gain of 9.9e-8, just under the 1e-7 below which it
    /// refuses to churn, leaving 2e-6 Eh and 0.005 electrons on the
    /// table with every other diagnostic reporting convergence.
    ///
    /// The unknowns are the fractionally occupied occupations and the
    /// chemical potential. Transfers between them conserve particle
    /// number by construction, so the step is expressed in the
    /// differences d_m = e_{k_m} - e_{k_0}: the gradient along one is
    /// eps_{k_m} - eps_{k_0}, free, and the curvature comes from
    /// second differences of the perturbatively relaxed energy, one
    /// Fock build per point rather than one orbital optimisation.
    ///
    /// Orbitals pinned full or empty enter through an active set. A
    /// pinned orbital on the wrong side of the chemical potential has
    /// a descent direction into the fractional region, so it joins the
    /// set and the step is re-solved; one that stays on its own side
    /// is at a legitimate bound and is left alone.
    ///
    /// The step is chosen by the residual and only *checked* against
    /// the energy: a step that raises it is rejected, so the rule that
    /// nothing is adopted unless it lowers the energy survives.
    bool kkt_occupation_refinement_(const AllowedMethods & allowed) {
      if(orbital_history_.empty() || frozen_occupations_) {
        log_(5, "KKT refinement: skipped (history empty or occupations"
                " frozen).\n");
        return false;
      }
      if(kkt_occupation_refinement_steps_ <= 0) return false;

      // The sweeps below relocate the iterate with
      // initialize_with_orbitals, which restarts the orbital history.
      // Save it, and hand back only the improved iterate as one new
      // entry, so a caller in the middle of an SCF keeps the history
      // it was relying on.
      const HistorySnapshot snapshot = save_history_();
      // The sweeps move the iterate about; that is relocation, not a
      // new calculation, so the Fock counter must keep running.
      struct RelocationGuard {
        bool * flag;
        bool saved;
        ~RelocationGuard() { *flag = saved; }
      } relocation{&relocating_iterate_, relocating_iterate_};
      relocating_iterate_ = true;

      bool improved = false;

      // Curvature carried between sweeps, one record per particle
      // type. The probes below measure the curvature at *fixed*
      // orbitals, but every step is judged after the orbitals have
      // relaxed, and relaxation screens the occupation curvature --
      // the fixed-orbital value is an upper bound on the one that
      // governs the iteration. On a spin-restricted iron atom the two
      // happen to agree and the refinement converges quadratically;
      // on the M = 5 atom the fixed-orbital curvature is about four
      // times too stiff, every Newton step comes out four times too
      // short, and the residual then contracts by a constant 0.758 a
      // sweep -- linear convergence, which runs out of sweeps at 1e-5
      // instead of arriving. Raising the arithmetic does not touch it:
      // the residual is the same in double, in long double and in
      // quadruple precision.
      //
      // The relaxed curvature does not have to be modelled, only
      // read. Each accepted sweep leaves a step dn and, at the head of
      // the next sweep, the orbital energies it produced *after*
      // relaxation, so the secant condition H dn = deps holds for
      // exactly the curvature that governs the iteration. The
      // symmetric rank-one update imposes it while keeping H
      // symmetric, and costs nothing: both halves of the pair are
      // already computed.
      struct SecantRecord {
        std::vector<std::pair<size_t, size_t>> active;
        Matrix<Tbase> H;
        Vector<Tbase> eps_previous;
        Vector<Tbase> step_previous;
        bool has_curvature = false;
        bool has_pair = false;
      };
      std::vector<SecantRecord> secant(number_of_blocks_per_particle_type_.size());

      // Where the refinement started, so that the climb it is allowed
      // is bounded over the whole of it rather than per step.
      const Tbase entry_energy = get_energy();
      DensityMatrix<Torb, Tbase> accepted_density;
      FockBuilderReturn<Torb, Tbase> accepted_fock;
      Tbase trust = kkt_occupation_trust_radius_;
      for(int sweep = 0; sweep < kkt_occupation_refinement_steps_; sweep++) {
        const Tbase residual_before = fermi_level_error();
        log_(5, "KKT refinement: sweep %i, residual %e.\n", sweep,
             (double) residual_before);
        if(residual_before <= kkt_occupation_threshold_) break;

        Orbitals<Torb> C_pseudo;
        OrbitalEnergies<Tbase> eps;
        auto occupations = get_orbital_occupations();
        pseudo_canonicalise_(get_orbitals(), get_fock_matrix(), occupations,
                             C_pseudo, eps);
        const Tbase E_before = get_energy();
        const auto state_before = get_solution();

        // Active set: the fractional orbitals, plus any pinned orbital
        // whose energy puts it on the wrong side of the chemical
        // potential.
        bool any_step = false;
        for(Index iparticle = 0;
            iparticle < number_of_blocks_per_particle_type_.size();
            iparticle++) {
          std::vector<std::pair<size_t, size_t>> active;
          std::vector<Tbase> active_eps;
          std::vector<Tbase> fractional_eps;
          for(const auto & level : order_orbitals_by_energy(eps, iparticle)) {
            const size_t iblock = std::get<1>(level);
            const size_t iorb = std::get<2>(level);
            const Tbase n = occupations[iblock](iorb);
            if(n > occupation_change_threshold_
               && maximum_occupation_(iblock) - n > occupation_change_threshold_)
              fractional_eps.push_back(std::get<0>(level));
          }
          if(fractional_eps.empty()) {
            log_(5, "KKT refinement: particle %i has no fractional"
                    " occupations.\n", (int) iparticle);
            continue;
          }
          Tbase mu = 0;
          for(Tbase e : fractional_eps) mu += e;
          mu /= (Tbase) fractional_eps.size();

          for(const auto & level : order_orbitals_by_energy(eps, iparticle)) {
            const Tbase energy = std::get<0>(level);
            const size_t iblock = std::get<1>(level);
            const size_t iorb = std::get<2>(level);
            const Tbase n = occupations[iblock](iorb);
            const bool is_empty = n <= occupation_change_threshold_;
            const bool is_full =
              maximum_occupation_(iblock) - n <= occupation_change_threshold_;
            const bool wrong_side =
              (is_empty && energy < mu) || (is_full && energy > mu);
            if((!is_empty && !is_full) || wrong_side) {
              active.emplace_back(iblock, iorb);
              active_eps.push_back(energy);
            }
          }
          if(active.size() < 2) {
            log_(5, "KKT refinement: particle %i has %zu active orbitals,"
                    " nothing to transfer between.\n", (int) iparticle,
                 active.size());
            continue;
          }

          // The step is expressed in the occupations themselves,
          // with the particle number carried by an explicit
          // multiplier, rather than as transfers against one chosen
          // active orbital. What we want is the point where every
          // active orbital shares a chemical potential,
          //
          //   eps_k + sum_l H_kl dn_l = mu,   sum_l dn_l = 0,
          //
          // which is the bordered system solved below. Two things
          // follow from writing it this way. mu comes out of the
          // solve instead of being averaged over the fractional
          // orbitals, and taking an orbital out of the active set is
          // the deletion of a row and a column -- which is what makes
          // the bound handling below expressible at all. In the
          // transfer parametrisation the reference orbital appears in
          // every coordinate, so pinning *it* redefines all of them.
          const Index m_active = (Index) active.size();
          Vector<Tbase> eps_active(m_active);
          for(Index k = 0; k < m_active; k++) eps_active(k) = active_eps[k];

          // Curvature from the first derivatives of the orbital
          // energies, H_kl = d eps_k / d n_l.
          //
          // By Janak's theorem this is the same matrix as the second
          // derivative of the energy, but a first difference of eps
          // carries an error of order delta/h where a second
          // difference of the energy carries delta/h^2 -- three
          // orders better at the probe size used here, on a quantity
          // whose corruption at the 1e-9 level is what previously
          // inverted the sign of this curvature. It also costs one
          // probe per active orbital rather than growing with the
          // square of their number.
          //
          // The probe displaces a single occupation and does not
          // conserve the particle number: the derivative wanted is
          // the unconstrained one, and the constraint is imposed by
          // the border. Orbital energies are read as diagonal
          // elements in the *reference* pseudo-canonical basis rather
          // than by re-diagonalising, which keeps the orbital indices
          // meaning the same thing across probes and costs nothing
          // beyond the Fock build.
          const Tbase h = kkt_occupation_probe_;
          auto epsilon_at = [&](const OrbitalOccupations<Tbase> & occ,
                                Vector<Tbase> & out) {
            initialize_with_orbitals(get_orbitals(), occ);
            const auto & F = get_fock_matrix();
            out.resize(m_active);
            for(Index k = 0; k < m_active; k++) {
              const size_t b = active[k].first;
              const size_t i = active[k].second;
              out(k) = std::real((C_pseudo[b].adjoint() * F[b]
                                  * C_pseudo[b])(i, i));
            }
          };

          SecantRecord & memory = secant[iparticle];
          const bool same_active = memory.has_curvature
                                   && memory.active == active;
          Matrix<Tbase> H;
          if(same_active) {
            // Carry the curvature and correct it with what the last
            // step actually did. Re-probing here would throw that
            // information away and pay |A|+1 Fock builds to recover
            // the fixed-orbital answer we already know is the wrong
            // one.
            H = memory.H;
            if(memory.has_pair) {
              const Vector<Tbase> y = eps_active - memory.eps_previous;
              const Vector<Tbase> & sstep = memory.step_previous;
              const Vector<Tbase> residual = y - H * sstep;
              const Tbase denominator = residual.dot(sstep);
              // The usual safeguard: the rank-one update is unbounded
              // where the pair says nothing about the curvature along
              // the step, and applying it there is worse than keeping
              // the stale value.
              if(std::abs(denominator)
                   > Tbase(1e-8) * sstep.norm() * residual.norm()) {
                H += (residual * residual.transpose()) / denominator;
                log_(5, "KKT refinement: particle %i curvature updated on the"
                        " secant pair, diagonal now %e (was %e).\n",
                     (int) iparticle, (double) H(0, 0),
                     (double) memory.H(0, 0));
              }
            }
          } else {
            Vector<Tbase> eps_reference;
            epsilon_at(occupations, eps_reference);
            H.resize(m_active, m_active);
            bool curvature_ok = true;
            for(Index l = 0; l < m_active && curvature_ok; l++) {
              auto probe = occupations;
              probe[active[l].first](active[l].second) += h;
              Vector<Tbase> eps_probe;
              epsilon_at(probe, eps_probe);
              H.col(l) = (eps_probe - eps_reference) / h;
              if(!H.col(l).allFinite()) curvature_ok = false;
            }
            initialize_with_orbitals(state_before.first, state_before.second);
            if(!curvature_ok) continue;
            H = ((H + H.transpose()) / Tbase(2)).eval();
          }
          memory.active = active;
          memory.H = H;
          memory.has_curvature = true;
          memory.eps_previous = eps_active;
          memory.has_pair = false;

          // Walk the active set. Solving the bordered system gives the
          // step to the point where the active orbitals share a
          // chemical potential; if that step drives an occupation past
          // zero or past its maximum, the model is telling us that
          // orbital belongs at the bound. Take the step as far as the
          // bound, pin the orbital there, and solve again over what is
          // left -- the standard primal active-set walk, which
          // terminates in at most one pass per active orbital.
          //
          // Scaling the whole step back instead, and stopping just
          // short so that nothing ever quite pins, is what this
          // replaces. That kept an orbital the model wanted empty in
          // the active set indefinitely, and left the walk to creep
          // toward a bound it was never allowed to reach.
          std::vector<bool> pinned(m_active, false);
          Vector<Tbase> dn = Vector<Tbase>::Zero(m_active);
          Vector<Tbase> eps_model = eps_active;
          Tbase mu_solved = 0;
          bool have_step = false;
          for(Index round = 0; round < m_active; round++) {
            std::vector<Index> free_idx;
            for(Index k = 0; k < m_active; k++)
              if(!pinned[k]) free_idx.push_back(k);
            const Index nf = (Index) free_idx.size();
            if(nf < 2) break;

            // [ H  -1 ] [ dn ]   [ -eps ]
            // [ 1^T 0 ] [ mu ] = [   0  ]
            Matrix<Tbase> system = Matrix<Tbase>::Zero(nf + 1, nf + 1);
            Vector<Tbase> rhs = Vector<Tbase>::Zero(nf + 1);
            for(Index a = 0; a < nf; a++) {
              for(Index b = 0; b < nf; b++)
                system(a, b) = H(free_idx[a], free_idx[b]);
              system(a, nf) = Tbase(-1);
              system(nf, a) = Tbase(1);
              rhs(a) = -eps_model(free_idx[a]);
            }
            const Vector<Tbase> solution = system.fullPivLu().solve(rhs);
            if(!solution.allFinite()) break;

            Vector<Tbase> direction = Vector<Tbase>::Zero(m_active);
            for(Index a = 0; a < nf; a++) direction(free_idx[a]) = solution(a);
            mu_solved = solution(nf);

            // How far along it before something hits a bound.
            Tbase alpha = 1;
            Index blocking = -1;
            for(Index a = 0; a < nf; a++) {
              const Index k = free_idx[a];
              const Tbase n_k = occupations[active[k].first](active[k].second)
                                + dn(k);
              const Tbase d_k = direction(k);
              Tbase limit = std::numeric_limits<Tbase>::infinity();
              if(d_k > 0)
                limit = (maximum_occupation_(active[k].first) - n_k) / d_k;
              else if(d_k < 0)
                limit = -n_k / d_k;
              if(limit < alpha) {
                alpha = std::max(Tbase(0), limit);
                blocking = k;
              }
            }

            dn += alpha * direction;
            eps_model += alpha * (H * direction);
            have_step = true;
            if(blocking < 0) break;

            pinned[blocking] = true;
            log_(5, "KKT refinement: particle %i pins %s orbital %zu at its"
                    " %s bound; re-solving over the %i that remain.\n",
                 (int) iparticle,
                 block_descriptions_[active[blocking].first].c_str(),
                 active[blocking].second,
                 direction(blocking) > 0 ? "upper" : "lower",
                 (int) (nf - 1));
          }
          if(!have_step || !dn.allFinite()) continue;

          // The trust radius remains as a guard on the model, not on
          // the bounds: those are now handled exactly above.
          Tbase scale = 1;
          const Tbase longest = dn.cwiseAbs().maxCoeff();
          if(longest > trust) scale = trust / longest;
          const Tbase step_length = scale * longest;

          auto trial = occupations;
          for(Index k = 0; k < m_active; k++)
            trial[active[k].first](active[k].second) += scale * dn(k);
          for(size_t b = 0; b < trial.size(); b++)
            for(Index k = 0; k < trial[b].size(); k++)
              trial[b](k) = std::min(maximum_occupation_(b),
                                     std::max(Tbase(0), trial[b](k)));

          // The relaxation here is the expensive part of the sweep, and
          // screening it with the perturbative estimate -- rejecting
          // the step when the estimated relaxed energy rises, which is
          // free because the estimate only re-reads the Fock matrix
          // just built -- does not pay: measured on iron M=0 it
          // changed the Fock count by 331 against 327, inside the
          // spread of two runs of the same code. It fires on one
          // rejection in eight. Most rejections happen where the
          // energy moves by 1e-9 or less, which is below the
          // estimate's own accuracy, so it predicts the wrong sign;
          // and a couple are rejected on the residual, which an energy
          // screen cannot see. The cost lives in the sweeps that are
          // accepted and grinding, not the ones turned away.
          initialize_with_orbitals(get_orbitals(), trial);
          relax_orbitals_at_fixed_occupations_(allowed);
          const Tbase E_after = get_energy();
          const Tbase residual_after = fermi_level_error();
          // Both tests are needed, and neither is comfortable.
          //
          // The residual is first order in the occupation step, the
          // energy second: near the stationary point a step changes
          // the residual by H*dn but the energy only by H*dn^2/2. The
          // energy signal therefore dies as the *square* of the
          // residual and drops below the noise floor while the
          // residual is still far above its own. With a curvature of
          // 0.3 and an energy reproducible to about 1e-9, the energy
          // comparison carries signal only while the residual exceeds
          // 2e-5, yet it is asked to veto steps at 3e-6. Those
          // verdicts are noise, and each one either wastes a
          // relaxation or quarters the trust radius, which then takes
          // two more sweeps to win back.
          //
          // Dropping the energy test is nonetheless wrong. It lets a
          // spin-restricted iron atom converge two orders further, to
          // 1e-8 in 260 Fock builds rather than 1e-6 in 302, but costs
          // the M = 5 case an order of magnitude, ending at 1e-5 where
          // the energy test holds it below 1e-6. The residual cannot
          // see which of several stationary points is the lower one;
          // only the energy can. Admitting a margin does not split the
          // difference either -- any margin wide enough to silence the
          // noise verdicts is wide enough to admit the steps that
          // wander, and reproduces the dropped-test numbers exactly.
          //
          // What this wants is a criterion that knows which question
          // it is answering: the energy while the step is large enough
          // to be heard and the stationary point is still in doubt,
          // the residual once the point is settled and only its
          // precision is at stake.
          // Accept a move that lowers the residual unless it costs
          // real energy. Uphill within the convergence threshold is
          // not a cost: the residual is first order in the occupation
          // step where the energy is second, so near the solution the
          // energy difference is smaller than the scale the solver
          // acts on anywhere else, while the residual still carries
          // thousands of times its own noise. Insisting on a strictly
          // downhill energy threw away the best step in the run --
          // the trace shows the residual offered 2.9e-6 -> 3.4e-8,
          // eighty-five fold, refused over an energy rise of 1e-10.
          // The budget is spent against where the refinement began,
          // not against the previous step. A per-step allowance is a
          // per-step licence to drift: each move stays inside it and
          // the sum walks off, which cost 3e-7 to 7e-7 on the iron
          // atoms -- above what the energy tolerance of their own
          // regression tests permits. Measured cumulatively the
          // budget bounds the total climb instead, and the steps
          // worth having cost around 1e-10, four orders inside it.
          if(residual_after < residual_before
             && E_after <= entry_energy + minimum_useful_descent_()) {
            // Double on success, quarter on failure. Tying the radius
            // to the length actually taken instead, so that it
            // contracts with the problem, measured worse on both
            // counts: the residual ended 6 to 50 times higher and
            // there were *more* rejections, not fewer.
            const Tbase grown =
              std::min(kkt_occupation_trust_radius_.get(), trust * Tbase(2));
            log_(5, "KKT occupation refinement: residual %e -> %e,"
                    " energy change %e; trust radius %e -> %e\n",
                 (double) residual_before, (double) residual_after,
                 (double) (E_after - E_before), (double) trust,
                 (double) grown);
            trust = grown;
            improved = true;
            any_step = true;
            // Keep the accepted iterate explicitly. The history is
            // ordered by energy, so an accepted uphill move is not its
            // head, and handing back the head would quietly return the
            // state this step was taken to leave. Right here the
            // history holds only iterates at these occupations -- the
            // step reset it -- so its head is this step's relaxed
            // state and nothing older.
            accepted_density = std::get<0>(orbital_history_[0]);
            accepted_fock = std::get<1>(orbital_history_[0]);
            // The half of the secant pair that the next sweep will
            // complete, once it has read the orbital energies this
            // step's relaxation produced.
            memory.step_previous = scale * dn;
            memory.has_pair = true;
            log_occupation_iterate_("Occupation refinement", sweep,
                                    E_after - E_before);
          } else {
            log_(5, "KKT occupation refinement: step rejected"
                    " (residual %e -> %e, energy change %e); trust radius"
                    " %e -> %e.\n",
                 (double) residual_before, (double) residual_after,
                 (double) (E_after - E_before), (double) trust,
                 (double) (trust / Tbase(4)));
            initialize_with_orbitals(state_before.first, state_before.second);
            // Shrink below the step just rejected, not by a blind
            // factor. The Newton step is usually shorter than the
            // radius, so dividing the radius alone leaves the step
            // unchanged and the next sweep proposes and rejects
            // exactly the same one: the trace showed a step repeated
            // five times over while the radius fell from 5e-3 to
            // 2e-5, some seventy-five Fock builds spent re-deriving a
            // verdict already in hand. Anchoring on the step length
            // guarantees the next proposal differs.
            trust = std::min(trust, step_length) / Tbase(4);
            any_step = true;   // a shrunk radius is worth another sweep
          }
        }
        if(!any_step) break;
      }

      // Put the history back, then re-enter the improved iterate as a
      // single new entry. add_entry sorts by energy, so the refined
      // state takes its place at the head if it earned it.
      if(improved) {
        // Hand back the accepted iterate and nothing else. The
        // history is ordered by energy and the iterate is its head,
        // so putting the old entries back files an accepted uphill
        // move behind whichever earlier one it climbed away from --
        // which then becomes the iterate again, and the refinement
        // has been undone by its own bookkeeping. On iron that cost
        // the whole cleanup: the state handed on was a mixture that
        // no longer satisfied converged(), so the caller refused the
        // swap and dropped a gain of 7e-7 Eh along with it.
        orbital_history_.clear();
        clear_diis_caches_();
        add_entry(accepted_density, accepted_fock);
      } else {
        restore_history_(snapshot);
      }
      return improved;
    }

    /// Report the iterate the way run() reports an SCF iteration.
    ///
    /// The cleanup and the occupation refinement move the occupations
    /// around after the SCF loop has stopped printing, so without this
    /// the phase that decides how a fractionally filled shell is
    /// divided is the one phase with no iteration history to read.
    /// Same quantities, same order, so the two can be followed as one
    /// trace.
    void log_occupation_iterate_(const char * tag, int step,
                                 Tbase energy_change) {
      if(verbosity_ < 5) return;
      log_(5, "%s step %i: %i Fock evaluations energy % .10f change % e\n",
           tag, step, (int) number_of_fock_evaluations_,
           (double) get_energy(), (double) energy_change);
      log_(5, "Occupations: particle-number error %e, Aufbau error %e,"
              " Fermi-level error %e\n",
           (double) particle_number_error(), (double) aufbau_error(),
           (double) fermi_level_error());
      const auto occupations = get_orbital_occupations();
      const auto occ_idx = occupied_orbitals(occupations);
      for(size_t l = 0; l < occ_idx.size(); l++)
        if(occ_idx[l].size())
          log_stream_(5) << block_descriptions_[l] + " occupations: "
                         << occupations[l].head(occ_idx[l].maxCoeff() + 1).transpose()
                         << std::endl;
      // The orbitals sharing the Fermi level are the ones whose
      // occupations are free, so name them: a fractional occupation is
      // only meaningful against the degeneracy it sits in.
      Orbitals<Torb> C_pseudo;
      OrbitalEnergies<Tbase> eps;
      pseudo_canonicalise_(get_orbitals(), get_fock_matrix(), occupations,
                           C_pseudo, eps);
      for(Index iparticle = 0;
          iparticle < number_of_blocks_per_particle_type_.size(); iparticle++) {
        std::ostringstream oss;
        size_t count = 0;
        for(const auto & level : order_orbitals_by_energy(eps, iparticle)) {
          const size_t iblock = std::get<1>(level);
          const size_t iorb = std::get<2>(level);
          const Tbase n = occupations[iblock](iorb);
          if(n <= occupation_change_threshold_) continue;
          if(maximum_occupation_(iblock) - n <= occupation_change_threshold_)
            continue;
          oss << " " << block_descriptions_[iblock] << "#" << iorb
              << " n=" << (double) n << " eps=" << (double) std::get<0>(level);
          count++;
        }
        if(count)
          log_(5, "Fractionally occupied (particle %i):%s\n", (int) iparticle,
               oss.str().c_str());
      }
    }

    /// Residual of the conditions the occupations have to satisfy at a
    /// minimum, in energy units.
    ///
    /// Minimising the energy over the occupations subject to a fixed
    /// particle number and 0 <= n <= n_max gives, with a chemical
    /// potential mu and Janak's dE/dn_k = eps_k,
    ///
    ///     eps_k  = mu      where 0 < n_k < n_max   (fractional)
    ///     eps_k >= mu      where n_k = 0           (empty)
    ///     eps_k <= mu      where n_k = n_max       (full)
    ///
    /// This returns the largest violation of any of the three over the
    /// particle types. ``aufbau_error`` tests the two inequalities in
    /// occupation units and exempts everything inside the Fermi-level
    /// cluster, fractional occupation being legitimate exactly there
    /// -- so the equality, which is the condition that fixes *where*
    /// inside the cluster the fractions sit, went unmeasured.
    ///
    /// Measuring it in energy units rather than occupation units is
    /// the point. Near a stationary point the energy is quadratic in
    /// the occupation displacement while these residuals are linear,
    /// so a criterion built on energy decrease is insensitive to
    /// exactly the displacements this catches. Measured on a
    /// spin-restricted iron atom, two runs of the same command settled
    /// at 4s/3d fractions 0.005 apart, 2e-6 Eh apart, both reporting
    /// convergence by every other diagnostic the solver carries.
    ///
    /// mu is taken as the mean orbital energy over the fractionally
    /// occupied orbitals. Where there are none it is not determined by
    /// the equality at all, and any value between the highest full and
    /// the lowest empty orbital satisfies the inequalities, so the
    /// residual is how far those two have crossed.
    Tbase fermi_level_error() const {
      if(orbital_history_.empty() || frozen_occupations_) return 0;

      Orbitals<Torb> C_pseudo;
      OrbitalEnergies<Tbase> eps;
      const auto occupations = get_orbital_occupations();
      pseudo_canonicalise_(get_orbitals(), get_fock_matrix(), occupations,
                           C_pseudo, eps);

      Tbase worst = 0;
      for(Index iparticle = 0;
          iparticle < number_of_blocks_per_particle_type_.size(); iparticle++) {
        std::vector<Tbase> fractional, empty, full;
        for(const auto & level : order_orbitals_by_energy(eps, iparticle)) {
          const Tbase energy = std::get<0>(level);
          const size_t iblock = std::get<1>(level);
          const size_t iorb = std::get<2>(level);
          const Tbase n = occupations[iblock](iorb);
          if(n <= occupation_change_threshold_)
            empty.push_back(energy);
          else if(maximum_occupation_(iblock) - n <= occupation_change_threshold_)
            full.push_back(energy);
          else
            fractional.push_back(energy);
        }

        if(fractional.empty()) {
          // mu is free between the bands; only a crossing is a
          // violation.
          if(!full.empty() && !empty.empty()) {
            const Tbase highest_full =
              *std::max_element(full.begin(), full.end());
            const Tbase lowest_empty =
              *std::min_element(empty.begin(), empty.end());
            worst = std::max(worst,
                             std::max(Tbase(0), highest_full - lowest_empty));
          }
          continue;
        }

        Tbase mu = 0;
        for(Tbase e : fractional) mu += e;
        mu /= (Tbase) fractional.size();

        for(Tbase e : fractional) worst = std::max(worst, std::abs(e - mu));
        for(Tbase e : empty) worst = std::max(worst, std::max(Tbase(0), mu - e));
        for(Tbase e : full) worst = std::max(worst, std::max(Tbase(0), e - mu));
      }
      return worst;
    }

    Tbase aufbau_error() const {
      if(orbital_history_.empty() || frozen_occupations_) return 0;

      Orbitals<Torb> C_pseudo;
      OrbitalEnergies<Tbase> eps;
      const auto occupations = get_orbital_occupations();
      pseudo_canonicalise_(get_orbitals(), get_fock_matrix(), occupations,
                           C_pseudo, eps);
      const auto aufbau = update_occupations(eps);

      Tbase worst = 0;
      for(Index iparticle = 0;
          iparticle < number_of_blocks_per_particle_type_.size(); iparticle++) {
        const auto levels = order_orbitals_by_energy(eps, iparticle);
        // Fermi level: the highest energy that Aufbau actually fills.
        bool have_fermi = false;
        Tbase eps_fermi = 0;
        for(const auto & level : levels) {
          const size_t iblock = std::get<1>(level);
          const size_t iorb = std::get<2>(level);
          if(aufbau[iblock](iorb) > occupation_change_threshold_) {
            eps_fermi = std::get<0>(level);
            have_fermi = true;
          }
        }
        if(!have_fermi) continue;

        for(const auto & level : levels) {
          const Tbase energy = std::get<0>(level);
          const size_t iblock = std::get<1>(level);
          const size_t iorb = std::get<2>(level);
          if(std::abs(energy - eps_fermi) <= optimal_damping_degeneracy_threshold_)
            continue;  // inside the Fermi-level cluster; free to be fractional
          const Tbase n = occupations[iblock](iorb);
          worst = std::max(worst, energy > eps_fermi
                                    ? std::abs(n)
                                    : std::abs(maximum_occupation_(iblock) - n));
        }
      }
      return worst;
    }

    /// Norm of the error vector of the ihist:th history entry -- the
    /// quantity converged() compares against the threshold.
    ///
    /// Defaults to the current iterate. The index is there because
    /// picking among stored iterates on their gradients is something
    /// the solver already does when a walk stalls, and something a
    /// caller may want for the same reason: whether the gradient of a
    /// given entry clears the threshold is not always a question the
    /// solver's own verdict answers, since the occupation cleanup can
    /// spend gradient on settling the occupations and the repair --
    /// worth about g^2/2H, far under what a Fock builder reproduces --
    /// cannot be verified by any line search.
    Tbase diis_error_norm(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Invalid entry!\n");
      return norm(diis_error_vector(ihist));
    }

    /// Check if we are converged
    bool converged() const {
        // Nothing has been iterated yet, so trivially not converged.
        // Guarding here rather than at every call site (including
        // print_settings) keeps the diagnostic safe to query at any
        // point in the solver's lifetime.
        if(orbital_history_.empty())
            return false;
        if(callback_convergence_function_) {

            // Data to pass to callback function
            std::map<std::string, std::any> callback_data;
            callback_data["dE"] = get_energy() - old_energy_;
            callback_data["diis_error"] = diis_error_norm();

            return callback_convergence_function_(callback_data);
        } else {
            return diis_error_norm() <= effective_convergence_threshold_();
        }
    }

    /// Whether the occupations are Aufbau to within
    /// ``aufbau_convergence_threshold_``. The other half of
    /// convergence: ``converged()`` asks whether the orbitals are at a
    /// stationary point, this asks whether the occupations are at one.
    ///
    /// Kept separate rather than folded into ``converged()`` because
    /// the two are established differently. The gradient criterion is
    /// a pure observation, but occupation space is only put right by
    /// ``aufbau_cleanup_step``, which needs the gradient to have
    /// converged first -- so a single predicate that demanded both
    /// would be circular, blocking on an error that nothing had yet
    /// been allowed to fix. ``run()`` therefore orders them: gradient,
    /// then cleanup, then this.
    bool occupations_converged() const {
        if(aufbau_convergence_threshold_ < Tbase(0)) return true;
        return aufbau_error() <= aufbau_convergence_threshold_;
    }

    /// Run the SCF
    ///
    /// Consumes the ``methods`` string setting, a ``+``-separated
    /// case-insensitive list drawn from ``"DIIS"`` (Pulay's
    /// A/EDIIS-bracketed direct inversion in the iterative subspace),
    /// ``"LCIIS"`` (Li & Yaron's least-squares commutator variant of
    /// the same extrapolation step -- it replaces the CDIIS
    /// coefficients and implies ``"DIIS"``, so asking for both is an
    /// error rather than a silent preference), ``"ODA"``
    /// (optimal-damping polytope step on the skeleton density
    /// matrices), and ``"CG"`` (preconditioned PR+ scaled steepest
    /// descent on orbital rotations at fixed occupations). The
    /// orbital-rotation slot also accepts ``"LBFGS"`` and ``"ARH"``
    /// (augmented Roothaan-Hall: the Roothaan-Hall diagonal made exact
    /// on the directions the density history spans).
    /// Configure via ``set("methods", ...)``; default is
    /// ``"DIIS + ODA + LBFGS"``. Examples:
    ///
    ///   ``"DIIS"``                pure A/EDIIS extrapolation
    ///   ``"LCIIS"``               least-squares commutator extrapolation
    ///   ``"ODA"``                 standalone polytope minimisation
    ///   ``"DIIS + ODA + LBFGS"``  full compound algorithm (default)
    ///   ``"DIIS + ODA + CG"``     PR+ CG in place of L-BFGS
    ///   ``"DIIS + ODA + ARH"``    augmented Roothaan-Hall rotations
    ///   ``"ODA + CG"``            DIIS-less compound
    ///
    /// State-transition rules: from DIIS we leave to ODA (or to CG when
    /// ODA is not allowed) on stall or large error; from ODA we hand
    /// to DIIS on integer occupations or to CG on fractional / failed
    /// occupations; from CG we burst ``orbital_rotation_steps_after_oda_`` (or the
    /// polytope dimension when that is left at zero) steps and then
    /// hand back to DIIS. The state-machine collapses gracefully when
    /// only a subset of the methods is allowed: ``"DIIS"`` alone
    /// keeps retrying DIIS until ``maximum_iterations_`` runs out;
    /// other subsets terminate early when every allowed method has
    /// failed in succession.
    void run() {
      // Dump the current settings once at the top of run() so a
      // verbosity 10 trace records exactly what the solver was
      // configured with. Route through log_stream_ so the message
      // ends up on the caller's log sink instead of unconditionally
      // on stdout, and skip the (non-trivial) catalog walk when the
      // gate is closed.
      if(auto ls = log_stream_(10); ls.enabled())
        print_settings(ls.stream());

      AllowedMethods allowed = parse_method_string(methods_);
      if(frozen_occupations_)
        allowed.oda = false;  // occupations are pinned; ODA cannot move them

      // Freeze the roundoff noise floor of the DIIS residual from the
      // initial Fock. The basis conditioning is dominated by the
      // one-electron part so this barely moves during the run.
      noise_floor_ = compute_noise_floor();
      if(verbosity_ > 0 && noise_safety_factor_ > 0 &&
         convergence_threshold_ < noise_safety_factor_ * noise_floor_) {
        log_(1, "Warning: convergence threshold %e is below %g x arithmetic "
               "noise floor %e Eh (epsilon=%e); clamping effective "
               "threshold to %e.\n",
               (double) convergence_threshold_.get(),
               (double) noise_safety_factor_.get(),
               (double) noise_floor_.get(),
               (double) std::numeric_limits<Tbase>::epsilon(),
               (double) (noise_safety_factor_ * noise_floor_));
      }

      enum class StepKind { DIIS, ODA, OrbitalRotation };

      // Initial state: prefer DIIS, then ODA, then the orbital-rotation
      // step (CG or LBFGS). The chosen state is guaranteed to be
      // allowed by the parser above.
      StepKind state = allowed.diis ? StepKind::DIIS
                     : allowed.oda  ? StepKind::ODA
                                    : StepKind::OrbitalRotation;

      auto pick_next = [&allowed](
          std::initializer_list<StepKind> preferences, StepKind fallback) {
        for(auto k : preferences) {
          if((k == StepKind::DIIS && allowed.diis) ||
             (k == StepKind::ODA  && allowed.oda)  ||
             (k == StepKind::OrbitalRotation && allowed.orbital_rotation()))
            return k;
        }
        return fallback;
      };

      old_energy_ = Tbase(0);
      int failed_iterations = 0;
      // Whether the "gradient converged, occupations have not" notice
      // has already been given at full volume this run.
      bool occupations_blocked_reported = false;
      // Whether the loop was left because every method had failed, as
      // opposed to running out of iterations.
      bool stopped_on_stall = false;
      // Number of orbital-rotation steps still owed by the current ODA -> orbital-rotation burst,
      // budgeted by orbital_rotation_steps_after_oda_ at the ODA transition.
      size_t orbital_rotation_steps_remaining = 0;

      // Burst-watch state: snapshot of the pseudo-canonical orbitals,
      // canonical-orbital energies, and equal-occupation sub-block
      // partition captured at the start of each ODA -> orbital-rotation
      // burst. After every rotation step the new pseudo-canonical
      // energies are compared against this snapshot to detect events
      // the rotation step itself cannot see: (i) a sign flip of
      // eps_i - eps_j for two differently-occupied orbitals (level
      // crossing across an occupation boundary), or (ii) an orbital
      // that has lost majority overlap with its initial equal-
      // occupation sub-block (qualitative change of orbital
      // character). Either trip ends the burst and hands control back
      // through the {DIIS, ODA} preference list so occupations can be
      // re-evaluated.
      Orbitals<Torb> burst_C_pseudo;
      OrbitalEnergies<Tbase> burst_eps;
      OrbitalOccupations<Tbase> burst_occ;
      // Per block: list of (i, j, eps_i - eps_j at burst start) for
      // every differently-occupied (i, j) pair, i < j. A pair trips
      // tripwire 1 when |delta_t - delta_0| exceeds
      // optimal_damping_degeneracy_threshold_, i.e. the gap has moved
      // by more than the energy window ODA uses to cluster orbitals.
      std::vector<std::vector<std::tuple<Index, Index, Tbase>>>
        burst_diff_occ_pairs;
      // Per block: index map orbital -> equal-occupation sub-block id
      // at burst start. Two orbitals share an id iff their occupations
      // differ by less than occupation_change_threshold_.
      std::vector<IndexVector> burst_subblock_id;
      // Minimum allowed sub-block-span overlap; falling below this for
      // any orbital ends the burst.
      const Tbase burst_subblock_overlap_floor = Tbase(0.8);

      auto init_burst_watch = [&]() {
        const auto C_now = get_orbitals();
        const auto F_now = get_fock_matrix();
        burst_occ = get_orbital_occupations();
        pseudo_canonicalise_(C_now, F_now, burst_occ, burst_C_pseudo, burst_eps);
        size_t nblk = burst_C_pseudo.size();
        burst_diff_occ_pairs.assign(nblk, {});
        burst_subblock_id.assign(nblk, IndexVector());
        for(size_t b = 0; b < nblk; b++) {
          if(burst_eps[b].size() == 0) continue;
          Index n_b = burst_eps[b].size();
          burst_subblock_id[b] = IndexVector::Zero(n_b);
          // Build sub-block ids by single-pass grouping on occupation.
          std::vector<bool> used(n_b, false);
          Index next_id = 0;
          for(Index i = 0; i < n_b; i++) {
            if(used[i]) continue;
            burst_subblock_id[b](i) = next_id;
            used[i] = true;
            for(Index j = i + 1; j < n_b; j++)
              if(!used[j] &&
                 std::abs(burst_occ[b](i) - burst_occ[b](j)) < occupation_change_threshold_) {
                burst_subblock_id[b](j) = next_id;
                used[j] = true;
              }
            next_id++;
          }
          // Differently-occupied pairs.
          for(Index i = 0; i < n_b; i++)
            for(Index j = i + 1; j < n_b; j++)
              if(burst_subblock_id[b](i) != burst_subblock_id[b](j)) {
                Tbase delta0 = burst_eps[b](i) - burst_eps[b](j);
                burst_diff_occ_pairs[b].emplace_back(i, j, delta0);
              }
        }
      };

      auto burst_watch_tripped = [&]() -> bool {
        if(burst_C_pseudo.empty()) return false;
        const auto C_now = get_orbitals();
        const auto F_now = get_fock_matrix();
        Orbitals<Torb> C_pseudo_now;
        OrbitalEnergies<Tbase> eps_now;
        pseudo_canonicalise_(C_now, F_now, burst_occ, C_pseudo_now, eps_now);

        // Tripwire 1: any differently-occupied pair's canonical-energy
        // gap has moved by more than ODA's degeneracy threshold.
        for(size_t b = 0; b < burst_diff_occ_pairs.size(); b++) {
          for(const auto & p : burst_diff_occ_pairs[b]) {
            Index i = std::get<0>(p);
            Index j = std::get<1>(p);
            Tbase delta0 = std::get<2>(p);
            Tbase deltat = eps_now[b](i) - eps_now[b](j);
            if(std::abs(deltat - delta0) > optimal_damping_degeneracy_threshold_) {
              log_(5, "Burst exit: block %zu orbitals %u, %u have a "
                     "canonical-energy gap shift %+e Eh (> threshold %e Eh).\n",
                     b, (unsigned) i, (unsigned) j,
                     (double) (deltat - delta0), (double) (optimal_damping_degeneracy_threshold_.get()));
              return true;
            }
          }
        }

        // Tripwire 2: any orbital has lost majority overlap with its
        // initial equal-occupation sub-block.
        for(size_t b = 0; b < burst_C_pseudo.size(); b++) {
          if(burst_C_pseudo[b].cols() == 0) continue;
          Matrix<Torb> ovl = C_pseudo_now[b].adjoint() * burst_C_pseudo[b];
          Index n_b = burst_C_pseudo[b].cols();
          for(Index i = 0; i < n_b; i++) {
            Index sid = burst_subblock_id[b](i);
            Tbase span_w = 0;
            for(Index j = 0; j < n_b; j++)
              if(burst_subblock_id[b](j) == sid)
                span_w += std::norm(ovl(i, j));
            if(span_w < burst_subblock_overlap_floor) {
              log_(5, "Burst exit: block %zu orbital %u has sub-block "
                     "overlap %.3f < %.3f.\n",
                     b, (unsigned) i, (double) (span_w), (double) (burst_subblock_overlap_floor));
              return true;
            }
          }
        }
        return false;
      };

      auto clear_burst_watch = [&]() {
        burst_C_pseudo.clear();
        burst_eps.clear();
        burst_occ.clear();
        burst_diff_occ_pairs.clear();
        burst_subblock_id.clear();
      };
      // For termination when DIIS is not in the allowed set: track
      // whether ODA / CG have failed since the most recent successful
      // step. The loop exits when every allowed non-DIIS method has
      // failed in succession.
      bool oda_failed = false, rotation_failed = false;
      for(size_t iteration=1; iteration <= maximum_iterations_; iteration++) {
        // Compute DIIS error
        Tbase diis_error = diis_error_norm();
        Tbase diis_max_error = diis_error_vector(0).template lpNorm<Eigen::Infinity>();
        Tbase dE = get_energy() - old_energy_;

        // Data to pass to callback function
        std::map<std::string, std::any> callback_data;
        callback_data["iter"] = iteration;
        callback_data["nfock"] = number_of_fock_evaluations_;
        callback_data["E"] = get_energy();
        callback_data["dE"] = get_energy() - old_energy_;
        callback_data["diis_error"] = diis_error;
        callback_data["diis_max_error"] = diis_max_error;

        log_(5, "\n\n");
        log_(1, "Iteration %i: %i Fock evaluations energy % .10f change % e DIIS error vector %s norm %e\n", (int) iteration, (int) number_of_fock_evaluations_, (double) (get_energy()), (double) (dE), error_norm_.get().c_str(), (double) (diis_error));
        log_(5, "History size %i\n",(int) orbital_history_.size());
        if(verbosity_>=5)
          log_(5, "Occupations: particle-number error %e, Aufbau error %e,"
                  " Fermi-level error %e\n",
               (double) (particle_number_error()), (double) (aufbau_error()),
               (double) (fermi_level_error()));
        if(verbosity_>=5) {
          const auto occupations = get_orbital_occupations();
          auto occ_idx(occupied_orbitals(occupations));
          for(size_t l=0;l<occ_idx.size();l++) {
            if(occ_idx[l].size())
              log_stream_(5) << block_descriptions_[l] + " occupations: " << occupations[l].head(occ_idx[l].maxCoeff()+1).transpose() << std::endl;
          }
        }
        if(converged()) {
          // When the collapsed skeleton set has been driving descent
          // for this SCF run, run one full-skeleton ODA step to
          // confirm the reduced fixed point is also stationary in
          // the full skeleton set. Skip the check when the last ODA
          // step already used the full skeletons (nothing new to
          // verify) or when ODA is disallowed. Sub-noise energy
          // drops are rejected on the same tolerance the trust-
          // region refit loop uses -- otherwise a "descent" of a few
          // times machine epsilon per iteration keeps the SCF
          // spinning at the arithmetic floor.
          if(allowed.oda && last_oda_via_collapsed_) {
            const Tbase E_before = get_energy();
            const bool descended = optimal_damping_step(/*force_full=*/true);
            const Tbase actual_descent = E_before - get_energy();
            const Tbase min_useful_descent = minimum_useful_descent_();
            if(descended && actual_descent > min_useful_descent) {
              log_(5, "Full-skeleton ODA found a %e Eh descent after "
                      "convergence; resuming SCF.\n", (double) (actual_descent));
              // Same post-ODA transition preference the state
              // machine uses after a normal ODA step: relax at the
              // new occupations before revisiting DIIS.
              state = pick_next({StepKind::OrbitalRotation, StepKind::DIIS},
                                StepKind::ODA);
              continue;
            }
            if(descended)
              log_(5, "Full-skeleton ODA descent %e below noise threshold %e; "
                      "treating as converged.\n",
                      (double) (actual_descent), (double) (min_useful_descent));
          }
          // The iterate is a mixed density, whose natural occupations
          // are only Aufbau up to the residual mixing. Report the
          // Aufbau filling of the converged Fock matrix instead, when
          // it does not cost energy.
          const bool cleanup_adopted =
            aufbau_cleanup_step(allowed, /*must_stay_converged=*/true);

          // Occupation space is judged after the cleanup, never
          // before: the cleanup is the step that puts the occupations
          // right, so testing ahead of it would block on an error
          // nothing has yet been able to fix.
          //
          // A cleanup that ran and was refused is the end of the road:
          // the same attempt would be made and refused on every
          // further pass, so blocking on the occupations would spin to
          // maximum_iterations. Report what is being
          // handed back and stop.
          if(!cleanup_adopted && !occupations_converged())
            log_(0, "Warning: converged occupations carry %e outside the "
                    "Fermi-level window, above the %e threshold, and the "
                    "Aufbau cleanup did not improve on them.\n",
                 (double) (aufbau_error()),
                 (double) (aufbau_convergence_threshold_.get()));

          if(cleanup_adopted && !occupations_converged()) {
            // Said once at the volume of the iteration line, then
            // demoted, so the extra iterations read as a stated
            // reason rather than as a stall.
            log_(occupations_blocked_reported ? 5 : 1,
                 "Orbital gradient converged, but %e of occupation sits "
                 "outside the Fermi-level window (threshold %e); "
                 "continuing.\n",
                 (double) (aufbau_error()),
                 (double) (aufbau_convergence_threshold_.get()));
            occupations_blocked_reported = true;
            // Occupations are ODA's business, so hand it the next step
            // when it is available.
            state = pick_next({StepKind::ODA, StepKind::OrbitalRotation},
                              StepKind::DIIS);
            continue;
          }

          // Say which it is. The cleanup is adopted on the energy and
          // does not consult the gradient, so it can settle the
          // occupations at the cost of the criterion that was met
          // before it ran. Announcing convergence regardless would
          // leave the caller holding a state its own converged()
          // denies, a contradiction the driver sees as an SCF that
          // both converged and did not.
          //
          // Refusing the swap instead is the worse trade, measured:
          // it is what makes the spin-restricted atom report
          // occupations 0.13 outside the Fermi-level window as
          // converged. The settled occupations are worth having; the
          // claim about the gradient is what gives.
          if(converged()) {
            log_(1, "Converged to energy % .10f!\n", (double) (get_energy()));
          } else {
            log_(1, "Settled at energy % .10f. The occupation cleanup left"
                    " the gradient at %e, against a threshold of %e: the"
                    " repair costs about g^2/2H, below what the Fock builder"
                    " resolves, so it cannot be recovered by a line"
                    " search.\n",
                 (double) get_energy(), (double) diis_error_norm(),
                 (double) effective_convergence_threshold_());
          }

          // Print out info
          callback_data["step"] = std::string("Converged");
          if(callback_function_)
            callback_function_(callback_data);
          break;
        }

        // Pre-step transition: bail out of DIIS if it is stalling or
        // the DIIS error is so large that A/EDIIS can't be trusted.
        // Only meaningful when at least one non-DIIS method is allowed.
        if(state == StepKind::DIIS && (allowed.oda || allowed.orbital_rotation())) {
          bool stalled =
            (diis_max_error >= optimal_damping_threshold_) ||
            (failed_iterations >= oda_restart_steps_);
          if(stalled) {
            StepKind next = pick_next({StepKind::ODA, StepKind::OrbitalRotation}, StepKind::DIIS);
            if(next != state) {
              if(verbosity_>=5) {
                const char * nname = next == StepKind::ODA ? "ODA"
                                   : orbital_rotation_token_(allowed);
                if(diis_max_error >= optimal_damping_threshold_)
                  log_(5, "Switching DIIS -> %s: DIIS max error %e exceeds threshold %e\n",
                         nname, (double) (diis_max_error), (double) (optimal_damping_threshold_.get()));
                else
                  log_(5, "Switching DIIS -> %s: %i consecutive failed DIIS iterations\n",
                         nname, failed_iterations);
              }
              state = next;
            }
          }
        }

        old_energy_ = get_energy();

        if(state == StepKind::ODA) {
          log_(5, "Optimal damping step\n");
          callback_data["step"] = std::string("ODA");
          if(callback_function_)
            callback_function_(callback_data);
          bool oda_ok = optimal_damping_step();
          // Size the orbital-rotation burst: explicit orbital_rotation_steps_after_oda_ if the
          // user has set it, otherwise the count of orbital-rotation
          // DOFs at the new iterate that live inside a degenerate
          // group (sum_p sum_b sum_g N_g (K_g - N_g) at a polytope
          // vertex; up to K_g (K_g - 1)/2 at an interior point). One
          // orbital-rotation step is taken as a floor so a trivial polytope still
          // gets at least the Roothaan relaxation pass after each
          // ODA call.
          size_t orbital_rotation_burst = orbital_rotation_steps_after_oda_ > 0
                              ? orbital_rotation_steps_after_oda_
                              : std::max<size_t>(last_active_rotation_count_, 1);
          if(oda_ok) {
            failed_iterations = 0;
            oda_failed = rotation_failed = false;
            if(has_integer_occupations()) {
              // Polytope optimum sits on a vertex; hand back to DIIS
              // if available, else fall through the preference list.
              state = pick_next({StepKind::DIIS, StepKind::OrbitalRotation}, StepKind::ODA);
            } else {
              // Fractional polytope-interior optimum; relax the orbital
              // rotations (CG or L-BFGS) before DIIS gets its turn.
              state = pick_next({StepKind::OrbitalRotation, StepKind::DIIS}, StepKind::ODA);
            }
            orbital_rotation_steps_remaining = (state == StepKind::OrbitalRotation) ? orbital_rotation_burst : 0;
          } else {
            // Polytope minimum says we can't descend in occupation
            // space; try orbital rotations next, or DIIS if the
            // orbital-rotation branch isn't allowed.
            oda_failed = true;
            state = pick_next({StepKind::OrbitalRotation, StepKind::DIIS}, StepKind::ODA);
            orbital_rotation_steps_remaining = (state == StepKind::OrbitalRotation) ? orbital_rotation_burst : 0;
          }
          if(state == StepKind::OrbitalRotation && orbital_rotation_steps_remaining > 1)
            init_burst_watch();
          else
            clear_burst_watch();
        } else if(state == StepKind::OrbitalRotation) {
          // CG vs L-BFGS vs ARH: exactly one of the three is enabled,
          // the parser having rejected a request for more than one.
          log_(5, "%s step (%i remaining in burst)\n",
               orbital_rotation_name_(allowed),
               (int) orbital_rotation_steps_remaining);
          callback_data["step"] = std::string(orbital_rotation_token_(allowed));
          if(callback_function_)
            callback_function_(callback_data);
          bool rotation_ok = orbital_rotation_step_(allowed);
          if(rotation_ok) {
            failed_iterations = 0;
            oda_failed = rotation_failed = false;
          } else {
            rotation_failed = true;
          }
          if(orbital_rotation_steps_remaining > 0)
            orbital_rotation_steps_remaining--;
          // Stay in the orbital-rotation state only if we still owe
          // steps from the last ODA AND the line search just succeeded
          // AND the burst-watch has not flagged a level crossing or
          // qualitative orbital change. A failed line search means
          // there is no more descent in the orbital-rotation subspace
          // at the current occupations, and the next step would just
          // fail too; a tripped watch means the occupations themselves
          // are no longer the right Aufbau filling and ODA must be
          // re-consulted. In either case hand back through the
          // preference list (DIIS first, then ODA).
          bool burst_tripped =
            (orbital_rotation_steps_remaining > 0 && rotation_ok)
              ? burst_watch_tripped()
              : false;
          if(orbital_rotation_steps_remaining > 0 && rotation_ok && !burst_tripped) {
            state = StepKind::OrbitalRotation;
          } else {
            orbital_rotation_steps_remaining = 0;
            clear_burst_watch();
            state = pick_next({StepKind::DIIS, StepKind::ODA}, StepKind::OrbitalRotation);
          }
        } else {
          // DIIS step. Compute mixing factor (Garza and Scuseria, 2012).
          Tbase aediis_coeff;
          if(diis_error < diis_threshold_) {
            aediis_coeff = Tbase(0);
          } else if(diis_error < diis_epsilon_) {
            aediis_coeff = (diis_error-diis_threshold_)/(diis_epsilon_-diis_threshold_);
          } else {
            aediis_coeff = Tbase(1);
          }
          Vector<Tbase> weights;
          std::string step;
          std::tie(weights, step) = minimal_error_sampling_algorithm_weights(aediis_coeff);
          log_(5, "%s step\n",step.c_str());
          log_stream_(10) << "Extrapolation weights: " << weights.transpose() << std::endl;

          callback_data["step"] = step;
          if(callback_function_)
            callback_function_(callback_data);

          if(!attempt_extrapolation(weights)) {
            log_(10, "Warning: did not go down in energy!\n");
            failed_iterations++;
          } else {
            failed_iterations=0;
            oda_failed = rotation_failed = false;
            // The extrapolation moved the iterate, so the recorded
            // rotation trajectory no longer leads to it.
            clear_orbital_rotation_history_();
          }
          // Stay in DIIS; the pre-step check at the top of the next
          // iteration will move us to ODA / CG if DIIS keeps stalling.
          state = StepKind::DIIS;
        }

        // Early termination: if DIIS is not in the allowed set, exit
        // as soon as every allowed non-DIIS method has failed since
        // the last successful step. With DIIS available, DIIS keeps
        // retrying until ``maximum_iterations_`` and the loop never
        // exits early.
        if(!allowed.diis) {
          bool all_failed =
            (!allowed.oda || oda_failed) &&
            (!allowed.orbital_rotation() || rotation_failed);
          if(all_failed) {
            // Before giving up, stand on the best iterate computed
            // rather than the cheapest one to reach for.
            //
            // The history is ordered by energy and the iterate is its
            // head, but convergence is judged on the gradient, and the
            // two part company exactly here: the step stalls because
            // the energy has gone flat to the arithmetic, and among
            // iterates whose energies are then indistinguishable the
            // gradients still differ. A spin-restricted La+ cation
            // with PBE stalls at a head gradient of 1.06e-6 against a
            // threshold of 1e-6 while holding an entry at 9.6e-7 --
            // converged, computed, and passed over because it costs a
            // fraction of a microhartree.
            //
            // Choosing among entries already in hand, and only when
            // the walk has stopped anyway, so nothing that still had
            // somewhere to go is affected.
            if(orbital_history_.size() > 1) {
              size_t best_index = 0;
              Tbase best_gradient = diis_error_norm();
              for(size_t ihist = 1; ihist < orbital_history_.size(); ihist++) {
                const Tbase gradient = diis_error_norm(ihist);
                if(gradient < best_gradient) {
                  best_gradient = gradient;
                  best_index = ihist;
                }
              }
              if(best_index != 0) {
                log_(1, "Stalled at a gradient of %e, but history entry %zu"
                        " stands at %e; taking that one.\n",
                     (double) diis_error, best_index, (double) best_gradient);
                const auto density = std::get<0>(orbital_history_[best_index]);
                const auto fock = std::get<1>(orbital_history_[best_index]);
                orbital_history_.clear();
                clear_diis_caches_();
                add_entry(density, fock);
                diis_error = best_gradient;
              }
            }
            log_(1, "All allowed SCF methods failed at iteration %i; stopping with DIIS error vector %s norm %e.\n",
                   (int) iteration, error_norm_.get().c_str(), (double) (diis_error));
            stopped_on_stall = true;
            callback_data["step"] = std::string("Stalled");
            if(callback_function_)
              callback_function_(callback_data);
            break;
          }
        }
        // Do cleanup
        cleanup();
      }

      // A run that stalled a few percent above the threshold is not
      // meaningfully different from one that reached it, and without
      // this the two would hand back utterly different things: an
      // Aufbau occupation vector from the converged exit, the raw mixed
      // density with its whole Rydberg tail from the stalled one. The
      // step is adopted only if it does not raise the energy, so it
      // cannot make the reported answer worse.
      //
      // Running out of iterations is a different matter. There the
      // caller asked for a bounded amount of work, and the iterate is
      // wherever the SCF happened to be rather than near a fixed point,
      // so the cleanup's relaxation would be a full orbital
      // optimisation outside the SCF's own convergence control -- easily
      // more work than the iterations that were asked for.
      // ``maximum_iterations`` should mean what it says.
      // Gated on the stall, not on having failed: standing on the
      // best-gradient iterate can leave the run converged after all,
      // and the occupations still want settling either way. The guard
      // is set to whatever we are now claiming, so an accepted swap
      // cannot cost a convergence just recovered.
      if(stopped_on_stall)
        aufbau_cleanup_step(allowed, /*must_stay_converged=*/converged());
    }

    /// Get the SCF solution
    DensityMatrix<Torb, Tbase> get_solution(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]);
    }

    /// Get the orbitals
    Orbitals<Torb> get_orbitals(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).first;
    }

    /// Get the orbital occupations
    OrbitalOccupations<Tbase> get_orbital_occupations(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).second;
    }

    /// Get the Fock matrix builder return
    FockBuilderReturn<Torb, Tbase> get_fock_build(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]);
    }

    /// Get the Fock matrix for the ihist:th entry
    FockMatrix<Torb> get_fock_matrix(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]).second;
    }


    /// Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search
    void brute_force_search_for_lowest_configuration() {
      // Make sure we have a solution
      if(orbital_history_.size() == 0)
        run();
      else {
        Tbase diis_error = diis_error_norm();
        if(diis_error >= diis_threshold_)
          run();
      }

      // Get the reference orbitals and orbital occupations
      auto reference_solution = orbital_history_[0];
      auto reference_orbitals = get_orbitals();
      auto reference_occupations = get_orbital_occupations();
      auto reference_energy = get_energy();
      auto reference_fock = get_fock_matrix();

      // We also need the orbital energies below
      auto diagonalized_fock = compute_orbitals(reference_fock);
      const auto & orbital_energies = diagonalized_fock.second;

      // The brute-force search sweeps occupations, so it must thaw
      // and silence for the duration -- save the caller's settings and
      // restore them on return through the RAII guard.
      struct SettingsGuard {
        Setting<int> * verb; int old_verb;
        Setting<int> * frozen; int old_frozen;
        ~SettingsGuard() { *verb = old_verb; *frozen = old_frozen; }
      };
      SettingsGuard guard{&verbosity_, verbosity_,
                          &frozen_occupations_, frozen_occupations_};
      verbosity_ = 0;
      frozen_occupations_ = 0;
      while(true) {
        // Count the number of particles in each block
        Vector<Tbase> number_of_particles_per_block = Vector<Tbase>::Zero(number_of_blocks_);
        for(Index iblock=0; iblock<number_of_particles_per_block.size(); iblock++) {
          if(empty_block(iblock))
            continue;
          number_of_particles_per_block[iblock] = reference_occupations[iblock].sum();
        }
        log_stream_(0) << "Number of particles per block: " << number_of_particles_per_block.transpose() << std::endl;

        // List of occupations and resulting energies
        std::vector<std::pair<Vector<Tbase>,Tbase>> list_of_energies;

        // Loop over particle types. We have a double loop, since finding the lowest state in UHF probably requires this
        for(Index iparticle=0; iparticle<number_of_blocks_per_particle_type_.size(); iparticle++) {
          size_t iblock_start = particle_block_offset(iparticle);
          size_t iblock_end = iblock_start + number_of_blocks_per_particle_type_(iparticle);

          // One-particle moves
          for(size_t iblock_source = iblock_start; iblock_source < iblock_end; iblock_source++)
            for(size_t iblock_target = iblock_start; iblock_target < iblock_end; iblock_target++) {
              if(iblock_source == iblock_target)
                continue;

              // Maximum number to move
              Tbase num_i_source = number_of_particles_per_block[iblock_source];
              Tbase i_target_capacity = reference_occupations[iblock_target].size()*maximum_occupation_[iblock_target];
              Tbase i_target_capacity_left = i_target_capacity - reference_occupations[iblock_target].sum();
              int num_i_max = std::ceil(std::min(num_i_source, i_target_capacity_left));
              num_i_max = std::min(num_i_max, (int) std::round(std::min(maximum_occupation_[iblock_source], maximum_occupation_[iblock_target])));

              // Generate trials by moving particles
              for(int imove=1; imove<=num_i_max; imove++) {
                // Modify the occupations
                auto trial_number(number_of_particles_per_block);
                Tbase i_moved = std::min((Tbase) imove, trial_number(iblock_source));
                trial_number(iblock_source) -= i_moved;
                trial_number(iblock_target) += i_moved;

                if(trial_number(iblock_source) < Tbase(0) or trial_number(iblock_target) > i_target_capacity)
                  continue;

                fixed_number_of_particles_per_block_ = trial_number;

                log_(0, "isource = %i itarget = %i imoved = %f\n", (int)iblock_source, (int)iblock_target, (double) (i_moved));
                log_stream_(0) << "trial number of particles: " << trial_number.transpose() << std::endl;
                log_flush_();

                // Determine full orbital occupations from the specified data. Because we've fixed the number of particles in each block, it doesn't matter that the orbital energies aren't correct
                auto trial_occupations = update_occupations(orbital_energies);
                initialize_with_orbitals(reference_orbitals, trial_occupations);
                try {
                  run();
                } catch(...) {};
                // Add the result to the list
                list_of_energies.push_back(std::make_pair(trial_number, get_energy()));
                // Reset the restriction
                Vector<Tbase> dummy;
                fixed_number_of_particles_per_block_ = dummy;
              }
            }

          for(Index jparticle=0; jparticle<=iparticle; jparticle++) {
            size_t jblock_start = particle_block_offset(jparticle);
            size_t jblock_end = jblock_start + number_of_blocks_per_particle_type_(jparticle);

            // Loop over blocks of particles
            for(size_t iblock_source = iblock_start; iblock_source < iblock_end; iblock_source++)
              for(size_t iblock_target = iblock_start; iblock_target < iblock_end; iblock_target++) {

                bool same_particle = (iparticle == jparticle);
                size_t jblock_source_end = same_particle ? iblock_source+1 : jblock_end;
                size_t jblock_target_end = same_particle ? iblock_target+1 : jblock_end;
                log_(0, "iparticle= %i jparticle= %i isource=%i itarget=%i\n",(int)iparticle,(int)jparticle,(int)iblock_source,(int)iblock_target);

                for(size_t jblock_source = jblock_start; jblock_source < jblock_source_end; jblock_source++)
                  for(size_t jblock_target = jblock_start; jblock_target < jblock_target_end; jblock_target++) {
                    // Skip trivial cases
                    if(iblock_source == iblock_target and jblock_source == jblock_target)
                      continue;
                    if(iblock_source == jblock_target and jblock_source == iblock_target)
                      continue;
                    // Skip one-particle cases
                    if(iblock_source == jblock_source and iblock_target == jblock_target)
                      continue;

                    // Maximum number to move
                    Tbase num_i_source = number_of_particles_per_block[iblock_source];
                    Tbase i_target_capacity = reference_occupations[iblock_target].size()*maximum_occupation_[iblock_target];
                    Tbase i_target_capacity_left = i_target_capacity - reference_occupations[iblock_target].sum();
                    int num_i_max = std::ceil(std::min(num_i_source, i_target_capacity_left));
                    num_i_max = std::min(num_i_max, (int) std::round(std::min(maximum_occupation_[iblock_source], maximum_occupation_[iblock_target])));

                    Tbase num_j_source = number_of_particles_per_block[jblock_source];
                    Tbase j_target_capacity = reference_occupations[jblock_target].size()*maximum_occupation_[jblock_target];
                    Tbase j_target_capacity_left = j_target_capacity - reference_occupations[jblock_target].sum();
                    int num_j_max = std::ceil(std::min(num_j_source, j_target_capacity_left));
                    num_j_max = std::min(num_j_max, (int) std::round(std::min(maximum_occupation_[jblock_source], maximum_occupation_[jblock_target])));

                    log_(0, "i: source %f capacity left %f num max %i\n",(double) (num_i_source),(double) (i_target_capacity_left),num_i_max);
                    log_(0, "j: source %f capacity left %f num max %i\n",(double) (num_j_source),(double) (j_target_capacity_left),num_j_max);
                    log_flush_();

                    // Generate trials by moving particles
                    for(int imove=1; imove<=num_i_max; imove++)
                      for(int jmove=1; jmove<=num_j_max; jmove++) {
                        // These also lead to degeneracies
                        if(iblock_source == iblock_target and imove > 0)
                          continue;
                        if(iblock_source == iblock_target and jmove == 0)
                          continue;
                        if(jblock_source == jblock_target and jmove > 0)
                          continue;
                        if(jblock_source == jblock_target and imove == 0)
                          continue;

                        // Modify the occupations
                        auto trial_number(number_of_particles_per_block);
                        Tbase i_moved = std::min((Tbase) imove, trial_number(iblock_source));
                        trial_number(iblock_source) -= i_moved;
                        trial_number(iblock_target) += i_moved;
                        Tbase j_moved = std::min((Tbase) jmove, trial_number(jblock_source));
                        trial_number(jblock_source) -= j_moved;
                        trial_number(jblock_target) += j_moved;

                        if(trial_number(iblock_source) < Tbase(0) or trial_number(jblock_source) < Tbase(0))
                          continue;
                        if(trial_number(iblock_target) > i_target_capacity)
                          continue;
                        if(trial_number(jblock_target) > j_target_capacity)
                          continue;

                        fixed_number_of_particles_per_block_ = trial_number;

                        log_(0, "isource = %i itarget = %i imoved = %f\n", (int)iblock_source, (int)iblock_target, (double) (i_moved));
                        log_(0, "jsource = %i jtarget = %i jmoved = %f\n", (int)jblock_source, (int)jblock_target, (double) (j_moved));
                        log_stream_(0) << "trial number of particles: " << trial_number.transpose() << std::endl;
                        log_flush_();

                        // Determine full orbital occupations from the specified data. Because we've fixed the number of particles in each block, it doesn't matter that the orbital energies aren't correct
                        auto trial_occupations = update_occupations(orbital_energies);
                        initialize_with_orbitals(reference_orbitals, trial_occupations);
                        try {
                          run();
                        } catch(...) {};
                        // Add the result to the list
                        list_of_energies.push_back(std::make_pair(trial_number, get_energy()));
                        // Reset the restriction
                        Vector<Tbase> dummy;
                        fixed_number_of_particles_per_block_ = dummy;
                      }
                  }
              }
          }
        }

        // Sort the list in ascending order
        std::sort(list_of_energies.begin(), list_of_energies.end(), [](const std::pair<Vector<Tbase>,Tbase> & a, const std::pair<Vector<Tbase>,Tbase> & b) {return a.second < b.second;});

        log_(0, "Configurations\n");
        for(size_t iconf=0;iconf<list_of_energies.size();iconf++) {
          log_(0, "%4i E= % .10f with occupations\n",(int) iconf, (double) (list_of_energies[iconf].second));
          log_stream_(0) << list_of_energies[iconf].first.transpose() << std::endl;
        }

        if(list_of_energies[0].second < reference_energy) {
          log_(0, "Energy changed by %e by improved reference\n", (double) (list_of_energies[0].second - reference_energy));

          // Update the reference
          fixed_number_of_particles_per_block_ = list_of_energies[0].first;
          auto trial_occupations = update_occupations(orbital_energies);
          initialize_with_orbitals(reference_orbitals, trial_occupations);
          run();

          reference_solution = orbital_history_[0];
          reference_orbitals = get_orbitals();
          reference_occupations = get_orbital_occupations();
          reference_energy = get_energy();
          reference_fock = get_fock_matrix();
        } else {
          // Restore the reference calculation
          initialize_with_orbitals(reference_orbitals, reference_occupations);
          run();
          log_(0, "Search converged!\n");
          break;
        }
      }
    }

    void callback_function(std::function<void(const std::map<std::string,std::any> &)> callback_function = nullptr) {
      callback_function_ = callback_function;
    }

    void callback_convergence_function(std::function<bool(const std::map<std::string,std::any> &)> callback_convergence_function = nullptr) {
      callback_convergence_function_ = callback_convergence_function;
    }

    /// Register a log sink. The callback receives ``(level, message)``
    /// where ``level`` is the minimum verbosity_ at which the message
    /// would normally print and ``message`` is the finished, formatted
    /// text (newlines included). Pass a default-constructed
    /// std::function (or nullptr) to restore the stdout default.
    void logger(std::function<void(int, const std::string &)> sink = nullptr) {
      logger_ = std::move(sink);
    }

    /// True iff a caller-supplied log sink is currently installed.
    bool has_logger() const {
      return static_cast<bool>(logger_);
    }
  };
}
