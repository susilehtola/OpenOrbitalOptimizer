

# File scfsolver.hpp

[**File List**](files.md) **>** [**openorbitaloptimizer**](dir_3072c93c56dfbbd2cb4eee0809487533.md) **>** [**scfsolver.hpp**](scfsolver_8hpp.md)

[Go to the documentation of this file](scfsolver_8hpp.md)


```C++
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

    template<typename T>
    void project_onto_unit_simplex(Vector<T> & v) {
      for(Index i = 0; i < v.size(); i++)
        if(v(i) < T(0)) v(i) = T(0);
      T s = v.sum();
      if(s > T(1))
        v /= s;
    }

    template<typename T, size_t N>
    T evaluate_polynomial(const std::array<T, N> & coeffs, T x) {
      T r = coeffs[N - 1];
      for (size_t i = N - 1; i-- > 0;)
        r = r * x + coeffs[i];
      return r;
    }

  }

  template<typename Torb, typename Tbase> class SCFSolver {
    static_assert(std::is_same_v<Torb, Tbase> ||
                  std::is_same_v<Torb, std::complex<Tbase>>,
                  "SCFSolver<Torb, Tbase>: Torb must be either Tbase or "
                  "std::complex<Tbase>");
    /* Input data section */
    IndexVector number_of_blocks_per_particle_type_;
    Vector<Tbase> maximum_occupation_;
    Vector<Tbase> number_of_particles_;
    FockBuilder<Torb, Tbase> fock_builder_;
    BatchedFockBuilder<Torb, Tbase> batched_fock_builder_;
    std::vector<std::string> block_descriptions_;
    std::function<void(const std::map<std::string,std::any> & data)> callback_function_;
    std::function<bool(const std::map<std::string,std::any> & data)> callback_convergence_function_;
    std::function<void(int level, const std::string & msg)> logger_;

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

    class SettingBase {
    public:
      SettingBase(const char * key, const char * doc, bool writable)
        : key_(key), doc_(doc), writable_(writable) {}
      virtual ~SettingBase() = default;

      const char * key() const { return key_; }
      const char * doc() const { return doc_; }
      bool writable() const        { return writable_; }

      virtual const char * type() const = 0;
      virtual void print_value(std::ostream & os) const = 0;

    private:
      const char * key_;
      const char * doc_;
      bool writable_;
    };

    class SettingRegistry {
    public:
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

    template<typename T>
    class Setting : public SettingBase {
    public:
      using Hook = T (SCFSolver::*)(const T &) const;

      using Source = T (SCFSolver::*)() const;

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

    friend int operator++(Setting<int> & s, int) {
      int old = s.get();
      s = old + 1;
      return old;
    }


    SettingRegistry settings_;

    Setting<Tbase> convergence_threshold_{
        settings_, "convergence_threshold",
        "DIIS-error convergence threshold", Tbase(1e-7)};

    Setting<Tbase> aufbau_convergence_threshold_{
        settings_, "aufbau_convergence_threshold",
        "occupation outside the Fermi-level window allowed at convergence; "
        "negative disables the check",
        Tbase(1e-6)};

    Setting<Tbase> noise_safety_factor_{
        settings_, "noise_safety_factor",
        "K in effective threshold max(convergence_threshold, K * noise_floor)",
        Tbase(10)};

    Setting<std::string> error_norm_{
        settings_, "error_norm",
        "DIIS error norm; one of rms, fro, inf, 1, 2",
        "rms",
        true,
        &SCFSolver::canonicalise_error_norm_};

    Setting<std::string> methods_{
        settings_, "methods",
        "SCF method mix consumed by run(); e.g. \"DIIS + ODA + LBFGS\", \"DIIS\", \"LCIIS + ODA + CG\", \"ODA + CG\", \"DIIS + ODA + CG\"",
        "DIIS + ODA + LBFGS",
        true,
        &SCFSolver::canonicalise_methods_};

    Setting<Tbase> diis_epsilon_{
        settings_, "diis_epsilon",
        "pure-DIIS blend cutoff", Tbase(1e-1)};

    Setting<Tbase> diis_threshold_{
        settings_, "diis_threshold",
        "A/EDIIS blend cutoff (Garza-Scuseria)", Tbase(1e-4)};

    Setting<Tbase> diis_diagonal_damping_{
        settings_, "diis_diagonal_damping",
        "DIIS matrix diagonal damping", Tbase(0.02)};

    Setting<Tbase> diis_restart_factor_{
        settings_, "diis_restart_factor",
        "DIIS history restart factor", Tbase(1e-4)};

    Setting<int> lciis_maximum_history_{
        settings_, "lciis_maximum_history",
        "history entries LCIIS extrapolates over (0 = no separate cap)", 6};

    Setting<int> lciis_maximum_iterations_{
        settings_, "lciis_maximum_iterations",
        "max Newton iterations in the LCIIS quartic minimisation", 50};

    Setting<Tbase> lciis_convergence_threshold_{
        settings_, "lciis_convergence_threshold",
        "convergence threshold on the LCIIS Newton step norm", Tbase(1e-10)};

    Setting<Tbase> optimal_damping_threshold_{
        settings_, "optimal_damping_threshold",
        "DIIS error above which ODA takes over", Tbase(1)};

    Setting<Tbase> optimal_damping_degeneracy_threshold_{
        settings_, "optimal_damping_degeneracy_threshold",
        "ODA orbital-degeneracy window (Eh)", Tbase(1e-2)};

    Setting<int> max_oda_refits_{
        settings_, "max_oda_refits",
        "max trust-region refits inside optimal_damping_step after acceptance (0 disables)",
        3};

    Setting<int> maximum_iterations_{
        settings_, "maximum_iterations",
        "outer SCF iteration cap", 128};

    Setting<int> maximum_history_length_{
        settings_, "maximum_history_length",
        "DIIS and L-BFGS history depth", 10};

    Setting<int> oda_restart_steps_{
        settings_, "oda_restart_steps",
        "steps of no DIIS progress before switching to ODA", 5};

    Setting<int> orbital_rotation_steps_after_oda_{
        settings_, "orbital_rotation_steps_after_oda",
        "orbital-rotation steps after each ODA (0 = use last_active_rotation_count)",
        0};

    Setting<Tbase> minimal_gradient_projection_{
        settings_, "minimal_gradient_projection",
        "minimum preconditioned-CG projection on gradient", Tbase(1e-4)};

    Setting<Tbase> initial_level_shift_{
        settings_, "initial_level_shift",
        "orbital-rotation preconditioner floor", Tbase(1e-3)};

    Setting<Tbase> level_shift_factor_{
        settings_, "level_shift_factor",
        "level-shift diminution factor", Tbase(2)};

    Setting<Tbase> occupied_threshold_{
        settings_, "occupied_threshold",
        "occupied-orbital detection cutoff", Tbase(1e-6)};

    Setting<Tbase> occupation_change_threshold_{
        settings_, "occupation_change_threshold",
        "occupation-equality tolerance", Tbase(1e-6)};

    Setting<Tbase> density_restart_factor_{
        settings_, "density_restart_factor",
        "history density-diff restart factor", Tbase(1e-4)};

    Setting<int> frozen_occupations_{
        settings_, "frozen_occupations",
        "pin occupations across SCF (0 or 1)", 0};

    Setting<int> verbosity_{
        settings_, "verbosity",
        "0..30", 5};

    Setting<Tbase> noise_floor_{
        settings_, "noise_floor",
        "frozen roundoff floor of DIIS error, populated by run()",
        Tbase(0),
        false};

    Setting<int> number_of_fock_evaluations_{
        settings_, "number_of_fock_evaluations",
        "Fock-evaluation counter (reset on initialize_with_*)", 0, false};

    Setting<int> last_polytope_dimension_{
        settings_, "last_polytope_dimension",
        "ODA polytope dimension of the most recent optimal_damping_step",
        0,
        false};

    Setting<int> last_active_rotation_count_{
        settings_, "last_active_rotation_count",
        "active rotations counted by the most recent ODA step", 0, false};

    Setting<int> converged_{
        settings_, "converged",
        "0 or 1 -- re-evaluates the convergence rule now",
        0,
        false,
        nullptr,
        &SCFSolver::converged_as_int_};

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

    std::vector<SettingBase *> all_settings_() {
      return settings_.settings();
    }

    std::vector<const SettingBase *> all_settings_() const {
      return settings_.settings();
    }


    /* Internal data section */
    size_t number_of_blocks_;
    OrbitalHistory<Torb, Tbase> orbital_history_;
    OrbitalOccupations<Tbase> orbital_occupations_;

    mutable size_t next_history_index_ = 0;

    mutable std::map<size_t, std::vector<Matrix<Torb>>> diis_commutator_cache_;

    mutable std::map<std::pair<size_t, size_t>, Tbase> trace_DF_cache_;
    mutable std::map<std::pair<size_t, size_t>, Tbase> diis_matrix_cache_;
    mutable std::map<std::pair<size_t, size_t>, Tbase> density_diff_cache_;

    bool last_oda_via_collapsed_ = false;

    Tbase old_energy_ = Tbase(0);

    Vector<Tbase> previous_orbital_gradient_;
    Vector<Tbase> previous_orbital_direction_;
    std::vector<OrbitalRotation> previous_orbital_dofs_;

    struct LBFGSState {
      std::deque<Vector<Tbase>> s;
      std::deque<Vector<Tbase>> y;
      std::deque<Tbase> rho;
      Vector<Tbase> pending_s;
      Vector<Tbase> pending_g;
      std::vector<OrbitalRotation> history_dofs;
    };
    LBFGSState lbfgs_;

    using SkeletonOccupations =
        std::vector<std::vector<std::vector<Vector<Tbase>>>>;

    struct AllowedMethods {
      bool diis = false, oda = false, cg = false, lbfgs = false;
      bool lciis = false;
      bool orbital_rotation() const { return cg || lbfgs; }
      bool any() const { return diis || oda || orbital_rotation(); }
    };

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
        else throw std::logic_error("Unknown method '" + token
            + "' in methods string '" + methods
            + "' (allowed: DIIS, LCIIS, ODA, CG, LBFGS)");
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

      // Same story one level down: CG and L-BFGS are two ways to take
      // the one orbital-rotation step, and the step picks a single
      // one, so listing both would silently drop whichever lost.
      if(allowed.cg && allowed.lbfgs)
        throw std::logic_error("Both CG and LBFGS requested in methods"
            " string '" + methods + "': they are alternative"
            " implementations of the one orbital-rotation step, not"
            " steps that can both run, so the combination is ambiguous."
            " Ask for exactly one of them.");

      if(!allowed.any())
        throw std::logic_error("No methods enabled in '" + methods + "'");
      return allowed;
    }

    static std::string to_upper_copy(const std::string & s) {
      std::string out = s;
      std::transform(out.begin(), out.end(), out.begin(),
                     [](unsigned char c){ return std::toupper(c); });
      return out;
    }

    /* Internal functions */
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

    void log_flush_() const {
      if(!logger_) std::fflush(stdout);
    }

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
      bool enabled() const { return enabled_; }
      std::ostream & stream() { return oss_; }
    private:
      const SCFSolver * solver_;
      int level_;
      bool enabled_;
      std::ostringstream oss_;
    };

    LogStream log_stream_(int level) const {
      return LogStream(this, level);
    }

    bool empty_block(size_t iblock) const {
      // Check if Fock matrix has zero dimension
      if(iblock>=std::get<0>(orbital_history_[0]).first.size())
        throw std::logic_error("Trying to check empty block for nonexistent index!\n");
      return std::get<1>(orbital_history_[0]).second[iblock].size() == 0;
    }

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

    Matrix<Torb> get_density_matrix_block(size_t ihist, size_t iblock) const {
      return build_density_block_(
          get_orbital_block(ihist, iblock),
          get_orbital_occupation_block(ihist, iblock),
          maximum_occupation_(iblock));
    }

    Tbase tr_of_product_(const Matrix<Torb> & A,
                         const Matrix<Torb> & B) const {
      return std::real((A.array() * B.transpose().array()).sum());
    }

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

    OrbitalBlock<Torb> get_orbital_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbitals for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbitals for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).first[iblock];
    }

    OrbitalBlockOccupations<Tbase> get_orbital_occupation_block(size_t ihist, size_t iblock) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent history member!\n");
      if(iblock>=std::get<0>(orbital_history_[ihist]).first.size())
        throw std::logic_error("Trying to access orbital occupations for nonexistent block index!\n");
      return std::get<0>(orbital_history_[ihist]).second[iblock];
    }

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

    size_t get_index(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Trying to access index for nonexistent history member!\n");
      return std::get<2>(orbital_history_[ihist]);
    }

    size_t largest_index() const {
      size_t index = get_index(0);
      for(size_t i=1;i<orbital_history_.size();i++) {
        index = std::max(index, get_index(i));
      }
      return index;
    }

    IndexVector matrix_dimension() const {
      const auto & fock = std::get<1>(orbital_history_[0]).second;
      IndexVector dim(fock.size());
      for(size_t i=0;i<fock.size();i++)
        dim(i) = fock[i].cols();
      return dim;
    }

    Matrix<Torb> get_fock_matrix_block(size_t ihist, size_t iblock) const {
      return std::get<1>(orbital_history_[ihist]).second[iblock];
    }

    Vector<Tbase> vectorise(const Matrix<Torb> & mat) const {
      return vectorise_real_imag(mat);
    }

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

    void clear_diis_caches_() const {
      diis_commutator_cache_.clear();
      trace_DF_cache_.clear();
      diis_matrix_cache_.clear();
      density_diff_cache_.clear();
    }

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

    static std::pair<size_t, size_t> sorted_pair_(size_t a, size_t b) {
      return a <= b ? std::make_pair(a, b) : std::make_pair(b, a);
    }

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

    Matrix<Torb> diis_residual(size_t ihist, size_t iblock) const {
      // FPS - SPF (S = I) commutator lives on the cache with the
      // ``FP - PF`` sign convention. Project into the current
      // reference natural orbitals so the L^infty norm stays
      // basis-independent, matching the pre-cache semantics.
      const Matrix<Torb> & X = diis_commutator_cached_(ihist, iblock);
      auto C = get_orbital_block(0, iblock);
      return C.adjoint() * X * C;
    }

    std::vector<Matrix<Torb>> diis_residual(size_t ihist) const {
      std::vector<Matrix<Torb>> residuals(number_of_blocks_);
      for(size_t iblock=0; iblock<number_of_blocks_; iblock++) {
        if(empty_block(iblock))
          continue;
        residuals[iblock] = diis_residual(ihist, iblock);
      }
      return residuals;
    }

    Vector<Tbase> diis_error_vector(size_t ihist, size_t iblock) const {
      return vectorise(diis_residual(ihist, iblock));
    }

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

    Tbase effective_convergence_threshold_() const {
      return std::max<Tbase>(convergence_threshold_,
                             noise_safety_factor_ * noise_floor_);
    }

    Tbase minimum_useful_descent_() const {
      return effective_convergence_threshold_() / Tbase(10);
    }

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
    size_t lciis_subspace_size_() const {
      size_t M = orbital_history_.size();
      if(lciis_maximum_history_ > 0)
        M = std::min(M, (size_t) lciis_maximum_history_);
      return M;
    }

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

      Matrix<Tbase> search_directions = Matrix<Tbase>::Identity(b.size(), b.size());

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

    Vector<Tbase> ediis_linear_term() const {
      Vector<Tbase> ret = Vector<Tbase>::Zero(orbital_history_.size());
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++) {
        ret(ihist) = get_energy(ihist);
      }
      return ret;
    }

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

    Vector<Tbase> adiis_weights() const {
      return aediis_weights(adiis_linear_term(), adiis_quadratic_term());
    }

    Vector<Tbase> ediis_weights() const {
      return aediis_weights(ediis_linear_term(), ediis_quadratic_term());
    }

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

    void conserve_block_occupations_(Vector<Tbase> & occupations,
                                     Tbase target) const {
      for(Index k = 0; k < occupations.size(); k++)
        if(occupations(k) < Tbase(0))
          occupations(k) = Tbase(0);
      Tbase sum = occupations.sum();
      if(sum > std::numeric_limits<Tbase>::min())
        occupations *= target / sum;
    }

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

    bool optimal_damping_step(bool force_full = false) {
      std::pair<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>>
        best_evaluated;
      SkeletonOccupations skeletons;
      return optimal_damping_step_(force_full, /*exclude_reference=*/false,
                                   best_evaluated, skeletons);
    }

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

    void cleanup() {
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

    Vector<Tbase> preconditioned_sd_direction_(
        const RotationStepContext & ctx, Tbase sigma) const {
      Vector<Tbase> d(ctx.n_par);
      for(size_t k = 0; k < ctx.n_par; k++)
        d(k) = -ctx.g(k) / (sigma + std::max(Tbase(0), ctx.h(k)));
      return d;
    }

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

    Tbase t_max_for_K_(const Orbitals<Torb> & K) const {
      Tbase t_max = std::numeric_limits<Tbase>::max();
      for(size_t b = 0; b < K.size(); b++) {
        if(K[b].size() == 0) continue;
        Matrix<std::complex<Tbase>> KI = K[b].template cast<std::complex<Tbase>>() * std::complex<Tbase>(Tbase{0}, Tbase{-1});
        // Hermitize to suppress eig_sym roundoff warnings; -iK is
        // analytically Hermitian for anti-Hermitian K.
        KI = std::complex<Tbase>(Tbase{(Tbase(1)/Tbase(2))}) * (KI + KI.adjoint().eval());
        Eigen::SelfAdjointEigenSolver<Matrix<std::complex<Tbase>>> es(KI);
        Vector<Tbase> ev = es.eigenvalues();
        if(ev.size() > 0) {
          Tbase max_abs = ev.array().abs().maxCoeff();
          if(max_abs > 0) t_max = std::min(t_max, Tbase(M_PI / 2) / max_abs);
        }
      }
      return t_max;
    }

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

    void clear_orbital_rotation_history_() {
      previous_orbital_gradient_.resize(0);
      previous_orbital_direction_.resize(0);
      previous_orbital_dofs_.clear();
      lbfgs_ = LBFGSState();
    }

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

    static constexpr Tbase lbfgs_minimum_relative_descent_ = Tbase(0.5);

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

    std::vector<IndexVector> occupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> occ_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        occ_idx[l]=find_indices_where(occupations[l], [this](Tbase v){ return v >= occupied_threshold_; });
      }
      return occ_idx;
    }

    std::vector<IndexVector> unoccupied_orbitals(const OrbitalOccupations<Tbase> & occupations) {
      std::vector<IndexVector> virt_idx(occupations.size());
      for(size_t l=0;l<occupations.size();l++) {
        virt_idx[l]=find_indices_where(occupations[l], [this](Tbase v){ return v < occupied_threshold_; });
      }
      return virt_idx;
    }

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

    struct OptionInfo {
      const char * key;
      const char * type;      
      bool         writable;  
      const char * doc;       
    };

  private:
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

    template<typename T>
    void assign_setting_(const std::string & key, const T & v,
                         const char * what) {
      Setting<T> * s = find_setting_<T>(key, what);
      if(!s->writable())
        throw std::invalid_argument(std::string("SCFSolver::") + what
            + ": '" + key + "' is a read-only diagnostic");
      *s = s->hook() ? (this->*(s->hook()))(v) : v;
    }

    std::string canonicalise_error_norm_(const std::string & v) const {
      Vector<Tbase> test = Vector<Tbase>::Ones(1);
      (void) norm(test, v);
      return v;
    }

    int converged_as_int_() const {
      return converged() ? 1 : 0;
    }

    std::string canonicalise_methods_(const std::string & v) const {
      (void) parse_method_string(v);
      return to_upper_copy(v);
    }

  public:
    static const std::vector<OptionInfo> & options() {
      static const std::vector<OptionInfo> catalog = prototype_().build_catalog_();
      return catalog;
    }

  private:
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

    void set(const std::string & key, const char * value) {
      set_string(key, std::string(value));
    }


    void set_real(const std::string & key, Tbase v) {
      assign_setting_<Tbase>(key, v, "set_real");
    }

    void set_int(const std::string & key, int v) {
      assign_setting_<int>(key, v, "set_int");
    }

    void set_string(const std::string & key, const std::string & v) {
      assign_setting_<std::string>(key, v, "set_string");
    }

    Tbase get_real(const std::string & key) const {
      const Setting<Tbase> * s = find_setting_<Tbase>(key, "get_real");
      return s->source() ? (this->*(s->source()))() : s->get();
    }

    int get_int(const std::string & key) const {
      const Setting<int> * s = find_setting_<int>(key, "get_int");
      return s->source() ? (this->*(s->source()))() : s->get();
    }

    std::string get_string(const std::string & key) const {
      const Setting<std::string> * s = find_setting_<std::string>(key, "get_string");
      return s->source() ? (this->*(s->source()))() : s->get();
    }

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

    static std::string citation() {
      return "Susi Lehtola and Lori A. Burns, "
             "\"OpenOrbitalOptimizer -- a reusable open source library "
             "for self-consistent field calculations\", "
             "J. Phys. Chem. A 129, 5651 (2025). "
             "doi:10.1021/acs.jpca.5c02110";
    }

    static void print_citation(std::ostream & os = std::cout) {
      os << "If you use OpenOrbitalOptimizer, please cite:\n"
         << "  " << citation() << "\n";
      os.flush();
    }

    // === End settings façade ==============================================

    void set_batched_fock_builder(BatchedFockBuilder<Torb, Tbase> builder) {
      batched_fock_builder_ = std::move(builder);
    }

    bool has_batched_fock_builder() const {
      return batched_fock_builder_ != nullptr;
    }

    Tbase get_energy(size_t ihist=0) const {
      if(ihist>=orbital_history_.size())
        throw std::logic_error("Invalid entry!\n");
      return std::get<1>(orbital_history_[ihist]).first;
    }


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

    OrbitalHistoryEntry<Torb, Tbase> make_history_entry(const DensityMatrix<Torb, Tbase> & density_matrix, const FockBuilderReturn<Torb, Tbase> & fock) const {
      return std::make_tuple(density_matrix, fock, next_history_index_++);
    }

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

    void print_history() const {
      // Unconditional (caller invokes this explicitly for diagnostics).
      log_(0, "Orbital history\n");
      for(size_t ihist=0;ihist<orbital_history_.size();ihist++)
        log_(0, "%2i % .9f % e % i\n",(int) ihist, (double) (get_energy(ihist)), (double) (get_energy(ihist)-get_energy()), (int) get_index(ihist));
    }

    void reset_history() {
      while(orbital_history_.size()>1)
        orbital_history_.pop_back();
      clear_diis_caches_();
    }

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

    Index particle_block_offset(size_t iparticle) const {
      return (iparticle>0) ? number_of_blocks_per_particle_type_.head(iparticle).sum() : 0;
    }

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

    std::vector<std::tuple<Tbase, size_t, size_t>> order_orbitals_by_energy(const OrbitalEnergies<Tbase> & orbital_energies, size_t iparticle) const {
      size_t block_offset = particle_block_offset(iparticle);
      std::vector<std::tuple<Tbase, size_t, size_t>> all_energies;
      for(size_t iblock = block_offset; iblock < block_offset + (size_t)number_of_blocks_per_particle_type_(iparticle); iblock++)
        for(Index iorb = 0; iorb < orbital_energies[iblock].size(); iorb++)
          all_energies.push_back(std::make_tuple(orbital_energies[iblock](iorb), iblock, iorb));
      std::stable_sort(all_energies.begin(), all_energies.end(), [](const std::tuple<Tbase, size_t, size_t> & a, const std::tuple<Tbase, size_t, size_t> & b) {return std::get<0>(a) < std::get<0>(b);});
      return all_energies;
    }

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
        if(!(allowed.lbfgs ? lbfgs_step() : scaled_steepest_descent_step()))
          break;
    }

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
      if(E_origin < best_energy) {
        best_energy = E_origin;
        best_state = get_solution();
      }

      Matrix<Tbase> hessian(npars, npars);
      for(size_t j = 0; j < npars; j++) {
        Vector<Tbase> vertex = Vector<Tbase>::Zero(npars);
        vertex(j) = Tbase(1);
        const Tbase E_vertex = sample(vertex);
        if(E_vertex < best_energy) {
          best_energy = E_vertex;
          best_state = get_solution();
        }
        hessian.col(j) = relaxed_occupation_gradient_(skeletons, axis) - gradient;
      }
      // The Hessian is symmetric; the finite differences are not, quite.
      hessian = ((hessian + hessian.transpose()) / Tbase(2)).eval();

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
      const size_t maximum_refinements = 4;
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

        const Tbase E_star = sample(lambda_star);
        log_stream_(5) << "Relaxed polytope: lambda = "
                       << lambda_star.transpose() << ", E = " << E_star
                       << std::endl;
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

    bool aufbau_cleanup_step(const AllowedMethods & allowed,
                             bool must_stay_converged) {
      if(frozen_occupations_) {
        log_(5, "Aufbau cleanup skipped: occupations are frozen.\n");
        return false;
      }
      log_(5, "Aufbau cleanup: choosing occupations, then relaxing at them.\n");

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

      // Accept unless the Aufbau state is worse by an amount the SCF
      // itself would act on. Demanding a strict improvement throws away
      // exactly-Aufbau occupations to defend an energy difference
      // smaller than the descent the solver already calls noise, which
      // is the wrong trade: the occupations are the point of the step,
      // and an energy difference below that scale is not evidence of a
      // preference for the mixed density.
      const Tbase tolerance = minimum_useful_descent_();
      if(aufbau_energy <= reference_energy + tolerance
         && !(must_stay_converged && !converged())) {
        log_(5, "Aufbau cleanup accepted, energy change %e\n",
             (double) (aufbau_energy - reference_energy));
        return true;
      }

      if(must_stay_converged && aufbau_energy <= reference_energy + tolerance)
        log_(5, "Aufbau cleanup would cost the convergence criterion; "
                "keeping the converged density.\n");
      else

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
            callback_data["diis_error"] = norm(diis_error_vector(0));

            return callback_convergence_function_(callback_data);
        } else {
            return norm(diis_error_vector(0))
                <= effective_convergence_threshold_();
        }
    }

    bool occupations_converged() const {
        if(aufbau_convergence_threshold_ < Tbase(0)) return true;
        return aufbau_error() <= aufbau_convergence_threshold_;
    }

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
        Tbase diis_error = norm(diis_error_vector(0));
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
          log_(5, "Occupations: particle-number error %e, Aufbau error %e\n",
               (double) (particle_number_error()), (double) (aufbau_error()));
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

          log_(1, "Converged to energy % .10f!\n", (double) (get_energy()));

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
                                   : (allowed.lbfgs ? "L-BFGS" : "CG");
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
          // CG vs L-BFGS: exactly one of the two is enabled, the
          // parser having rejected a request for both.
          const bool use_lbfgs = allowed.lbfgs;
          log_(5, "%s step (%i remaining in burst)\n",
               use_lbfgs ? "L-BFGS" : "Scaled steepest descent",
               (int) orbital_rotation_steps_remaining);
          callback_data["step"] = std::string(use_lbfgs ? "LBFGS" : "CG");
          if(callback_function_)
            callback_function_(callback_data);
          bool rotation_ok = use_lbfgs ? lbfgs_step() : scaled_steepest_descent_step();
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
      if(!converged() && stopped_on_stall)
        aufbau_cleanup_step(allowed, /*must_stay_converged=*/false);
    }

    DensityMatrix<Torb, Tbase> get_solution(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]);
    }

    Orbitals<Torb> get_orbitals(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).first;
    }

    OrbitalOccupations<Tbase> get_orbital_occupations(size_t ihist=0) const {
      return std::get<0>(orbital_history_[ihist]).second;
    }

    FockBuilderReturn<Torb, Tbase> get_fock_build(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]);
    }

    FockMatrix<Torb> get_fock_matrix(size_t ihist=0) const {
      return std::get<1>(orbital_history_[ihist]).second;
    }


    void brute_force_search_for_lowest_configuration() {
      // Make sure we have a solution
      if(orbital_history_.size() == 0)
        run();
      else {
        Tbase diis_error = norm(diis_error_vector(0));
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

    void logger(std::function<void(int, const std::string &)> sink = nullptr) {
      logger_ = std::move(sink);
    }

    bool has_logger() const {
      return static_cast<bool>(logger_);
    }
  };
}
```


