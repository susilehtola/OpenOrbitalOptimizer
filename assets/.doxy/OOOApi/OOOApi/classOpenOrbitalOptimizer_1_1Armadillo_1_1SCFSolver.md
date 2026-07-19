

# Class OpenOrbitalOptimizer::Armadillo::SCFSolver

**template &lt;class Torb, class Tbase&gt;**



[**ClassList**](annotated.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**Armadillo**](namespaceOpenOrbitalOptimizer_1_1Armadillo.md) **>** [**SCFSolver**](classOpenOrbitalOptimizer_1_1Armadillo_1_1SCFSolver.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**SCFSolver**](#function-scfsolver) (const arma::uvec & number\_of\_blocks\_per\_particle\_type, const arma::Col&lt; Tbase &gt; & maximum\_occupation, const arma::Col&lt; Tbase &gt; & number\_of\_particles, const FockBuilder&lt; Torb, Tbase &gt; & fock\_builder, const std::vector&lt; std::string &gt; & block\_descriptions) <br> |
|  void | [**brute\_force\_search\_for\_lowest\_configuration**](#function-brute_force_search_for_lowest_configuration) () <br> |
|  DiagonalizedFockMatrix&lt; Torb, Tbase &gt; | [**compute\_orbitals**](#function-compute_orbitals) (const FockMatrix&lt; Torb &gt; & fock) const<br> |
|  bool | [**converged**](#function-converged) () const<br> |
|  void | [**convergence\_threshold**](#function-convergence_threshold-12) (Tbase t) <br> |
|  Tbase | [**convergence\_threshold**](#function-convergence_threshold-22) () const<br> |
|  void | [**diis\_diagonal\_damping**](#function-diis_diagonal_damping-12) (Tbase e) <br> |
|  Tbase | [**diis\_diagonal\_damping**](#function-diis_diagonal_damping-22) () const<br> |
|  void | [**diis\_epsilon**](#function-diis_epsilon-12) (Tbase e) <br> |
|  Tbase | [**diis\_epsilon**](#function-diis_epsilon-22) () const<br> |
|  void | [**diis\_restart\_factor**](#function-diis_restart_factor-12) (Tbase e) <br> |
|  Tbase | [**diis\_restart\_factor**](#function-diis_restart_factor-22) () const<br> |
|  void | [**diis\_threshold**](#function-diis_threshold-12) (Tbase e) <br> |
|  Tbase | [**diis\_threshold**](#function-diis_threshold-22) () const<br> |
|  void | [**error\_norm**](#function-error_norm) (const std::string & n) <br> |
|  void | [**fixed\_number\_of\_particles\_per\_block**](#function-fixed_number_of_particles_per_block) (const arma::Col&lt; Tbase &gt; & v) <br> |
|  bool | [**frozen\_occupations**](#function-frozen_occupations-12) () const<br> |
|  void | [**frozen\_occupations**](#function-frozen_occupations-22) (bool b) <br> |
|  Tbase | [**get\_energy**](#function-get_energy) (size\_t ihist=0) const<br> |
|  FockBuilderReturn&lt; Torb, Tbase &gt; | [**get\_fock\_build**](#function-get_fock_build) (size\_t ihist=0) const<br> |
|  FockMatrix&lt; Torb &gt; | [**get\_fock\_matrix**](#function-get_fock_matrix) (size\_t ihist=0) const<br> |
|  OrbitalOccupations&lt; Tbase &gt; | [**get\_orbital\_occupations**](#function-get_orbital_occupations) (size\_t ihist=0) const<br> |
|  Orbitals&lt; Torb &gt; | [**get\_orbitals**](#function-get_orbitals) (size\_t ihist=0) const<br> |
|  DensityMatrix&lt; Torb, Tbase &gt; | [**get\_solution**](#function-get_solution) (size\_t ihist=0) const<br> |
|  void | [**initialize\_with\_fock**](#function-initialize_with_fock) (const FockMatrix&lt; Torb &gt; & fock\_guess) <br> |
|  void | [**initialize\_with\_orbitals**](#function-initialize_with_orbitals) (const Orbitals&lt; Torb &gt; & orbitals, const OrbitalOccupations&lt; Tbase &gt; & occupations) <br> |
|  size\_t | [**last\_active\_rotation\_count**](#function-last_active_rotation_count) () const<br> |
|  size\_t | [**last\_polytope\_dimension**](#function-last_polytope_dimension) () const<br> |
|  void | [**maximum\_history\_length**](#function-maximum_history_length-12) (int n) <br> |
|  int | [**maximum\_history\_length**](#function-maximum_history_length-22) () const<br> |
|  void | [**maximum\_iterations**](#function-maximum_iterations-12) (size\_t n) <br> |
|  size\_t | [**maximum\_iterations**](#function-maximum_iterations-22) () const<br> |
|  size\_t | [**number\_of\_fock\_evaluations**](#function-number_of_fock_evaluations) () const<br> |
|  void | [**oda\_restart\_steps**](#function-oda_restart_steps-12) (int n) <br> |
|  int | [**oda\_restart\_steps**](#function-oda_restart_steps-22) () const<br> |
|  void | [**optimal\_damping\_degeneracy\_threshold**](#function-optimal_damping_degeneracy_threshold-12) (Tbase e) <br> |
|  Tbase | [**optimal\_damping\_degeneracy\_threshold**](#function-optimal_damping_degeneracy_threshold-22) () const<br> |
|  void | [**optimal\_damping\_threshold**](#function-optimal_damping_threshold-12) (Tbase e) <br> |
|  Tbase | [**optimal\_damping\_threshold**](#function-optimal_damping_threshold-22) () const<br> |
|  void | [**orbital\_rotation\_steps\_after\_oda**](#function-orbital_rotation_steps_after_oda-12) (size\_t n) <br> |
|  size\_t | [**orbital\_rotation\_steps\_after\_oda**](#function-orbital_rotation_steps_after_oda-22) () const<br> |
|  void | [**print\_history**](#function-print_history) () const<br> |
|  void | [**reset\_history**](#function-reset_history) () <br> |
|  void | [**run**](#function-run) (const std::string & methods="DIIS + ODA + CG") <br> |
|  void | [**run\_optimal\_damping**](#function-run_optimal_damping) () <br>_Backwards-compatible alias for the pre-oda-merge behavior._  |
|  OrbitalOccupations&lt; Tbase &gt; | [**update\_occupations**](#function-update_occupations) (const OrbitalEnergies&lt; Tbase &gt; & orbital\_energies) const<br> |
|  void | [**verbosity**](#function-verbosity-12) (int v) <br> |
|  int | [**verbosity**](#function-verbosity-22) () const<br> |




























## Public Functions Documentation




### function SCFSolver 

```C++
inline OpenOrbitalOptimizer::Armadillo::SCFSolver::SCFSolver (
    const arma::uvec & number_of_blocks_per_particle_type,
    const arma::Col< Tbase > & maximum_occupation,
    const arma::Col< Tbase > & number_of_particles,
    const FockBuilder< Torb, Tbase > & fock_builder,
    const std::vector< std::string > & block_descriptions
) 
```




<hr>



### function brute\_force\_search\_for\_lowest\_configuration 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::brute_force_search_for_lowest_configuration () 
```




<hr>



### function compute\_orbitals 

```C++
inline DiagonalizedFockMatrix< Torb, Tbase > OpenOrbitalOptimizer::Armadillo::SCFSolver::compute_orbitals (
    const FockMatrix< Torb > & fock
) const
```




<hr>



### function converged 

```C++
inline bool OpenOrbitalOptimizer::Armadillo::SCFSolver::converged () const
```




<hr>



### function convergence\_threshold [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::convergence_threshold (
    Tbase t
) 
```




<hr>



### function convergence\_threshold [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::convergence_threshold () const
```




<hr>



### function diis\_diagonal\_damping [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_diagonal_damping (
    Tbase e
) 
```




<hr>



### function diis\_diagonal\_damping [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_diagonal_damping () const
```




<hr>



### function diis\_epsilon [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_epsilon (
    Tbase e
) 
```




<hr>



### function diis\_epsilon [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_epsilon () const
```




<hr>



### function diis\_restart\_factor [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_restart_factor (
    Tbase e
) 
```




<hr>



### function diis\_restart\_factor [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_restart_factor () const
```




<hr>



### function diis\_threshold [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_threshold (
    Tbase e
) 
```




<hr>



### function diis\_threshold [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::diis_threshold () const
```




<hr>



### function error\_norm 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::error_norm (
    const std::string & n
) 
```




<hr>



### function fixed\_number\_of\_particles\_per\_block 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::fixed_number_of_particles_per_block (
    const arma::Col< Tbase > & v
) 
```




<hr>



### function frozen\_occupations [1/2]

```C++
inline bool OpenOrbitalOptimizer::Armadillo::SCFSolver::frozen_occupations () const
```




<hr>



### function frozen\_occupations [2/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::frozen_occupations (
    bool b
) 
```




<hr>



### function get\_energy 

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::get_energy (
    size_t ihist=0
) const
```




<hr>



### function get\_fock\_build 

```C++
inline FockBuilderReturn< Torb, Tbase > OpenOrbitalOptimizer::Armadillo::SCFSolver::get_fock_build (
    size_t ihist=0
) const
```




<hr>



### function get\_fock\_matrix 

```C++
inline FockMatrix< Torb > OpenOrbitalOptimizer::Armadillo::SCFSolver::get_fock_matrix (
    size_t ihist=0
) const
```




<hr>



### function get\_orbital\_occupations 

```C++
inline OrbitalOccupations< Tbase > OpenOrbitalOptimizer::Armadillo::SCFSolver::get_orbital_occupations (
    size_t ihist=0
) const
```




<hr>



### function get\_orbitals 

```C++
inline Orbitals< Torb > OpenOrbitalOptimizer::Armadillo::SCFSolver::get_orbitals (
    size_t ihist=0
) const
```




<hr>



### function get\_solution 

```C++
inline DensityMatrix< Torb, Tbase > OpenOrbitalOptimizer::Armadillo::SCFSolver::get_solution (
    size_t ihist=0
) const
```




<hr>



### function initialize\_with\_fock 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::initialize_with_fock (
    const FockMatrix< Torb > & fock_guess
) 
```




<hr>



### function initialize\_with\_orbitals 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::initialize_with_orbitals (
    const Orbitals< Torb > & orbitals,
    const OrbitalOccupations< Tbase > & occupations
) 
```




<hr>



### function last\_active\_rotation\_count 

```C++
inline size_t OpenOrbitalOptimizer::Armadillo::SCFSolver::last_active_rotation_count () const
```




<hr>



### function last\_polytope\_dimension 

```C++
inline size_t OpenOrbitalOptimizer::Armadillo::SCFSolver::last_polytope_dimension () const
```




<hr>



### function maximum\_history\_length [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::maximum_history_length (
    int n
) 
```




<hr>



### function maximum\_history\_length [2/2]

```C++
inline int OpenOrbitalOptimizer::Armadillo::SCFSolver::maximum_history_length () const
```




<hr>



### function maximum\_iterations [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::maximum_iterations (
    size_t n
) 
```




<hr>



### function maximum\_iterations [2/2]

```C++
inline size_t OpenOrbitalOptimizer::Armadillo::SCFSolver::maximum_iterations () const
```




<hr>



### function number\_of\_fock\_evaluations 

```C++
inline size_t OpenOrbitalOptimizer::Armadillo::SCFSolver::number_of_fock_evaluations () const
```




<hr>



### function oda\_restart\_steps [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::oda_restart_steps (
    int n
) 
```




<hr>



### function oda\_restart\_steps [2/2]

```C++
inline int OpenOrbitalOptimizer::Armadillo::SCFSolver::oda_restart_steps () const
```




<hr>



### function optimal\_damping\_degeneracy\_threshold [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::optimal_damping_degeneracy_threshold (
    Tbase e
) 
```




<hr>



### function optimal\_damping\_degeneracy\_threshold [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::optimal_damping_degeneracy_threshold () const
```




<hr>



### function optimal\_damping\_threshold [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::optimal_damping_threshold (
    Tbase e
) 
```




<hr>



### function optimal\_damping\_threshold [2/2]

```C++
inline Tbase OpenOrbitalOptimizer::Armadillo::SCFSolver::optimal_damping_threshold () const
```




<hr>



### function orbital\_rotation\_steps\_after\_oda [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::orbital_rotation_steps_after_oda (
    size_t n
) 
```




<hr>



### function orbital\_rotation\_steps\_after\_oda [2/2]

```C++
inline size_t OpenOrbitalOptimizer::Armadillo::SCFSolver::orbital_rotation_steps_after_oda () const
```




<hr>



### function print\_history 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::print_history () const
```




<hr>



### function reset\_history 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::reset_history () 
```




<hr>



### function run 

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::run (
    const std::string & methods="DIIS + ODA + CG"
) 
```




<hr>



### function run\_optimal\_damping 

_Backwards-compatible alias for the pre-oda-merge behavior._ 
```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::run_optimal_damping () 
```




<hr>



### function update\_occupations 

```C++
inline OrbitalOccupations< Tbase > OpenOrbitalOptimizer::Armadillo::SCFSolver::update_occupations (
    const OrbitalEnergies< Tbase > & orbital_energies
) const
```




<hr>



### function verbosity [1/2]

```C++
inline void OpenOrbitalOptimizer::Armadillo::SCFSolver::verbosity (
    int v
) 
```




<hr>



### function verbosity [2/2]

```C++
inline int OpenOrbitalOptimizer::Armadillo::SCFSolver::verbosity () const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/armadillo_compat.hpp`

