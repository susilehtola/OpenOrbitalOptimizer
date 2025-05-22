

# Class OpenOrbitalOptimizer::SCFSolver

**template &lt;typename Torb, typename Tbase&gt;**



[**ClassList**](annotated.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md)



_SCF solver class._ 

* `#include <scfsolver.hpp>`





































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**SCFSolver**](#function-scfsolver) (const arma::uvec & number\_of\_blocks\_per\_particle\_type, const arma::Col&lt; Tbase &gt; & maximum\_occupation, const arma::Col&lt; Tbase &gt; & number\_of\_particles, const FockBuilder&lt; Torb, Tbase &gt; & fock\_builder, const std::vector&lt; std::string &gt; & block\_descriptions) <br>_Constructor._  |
|  bool | [**add\_entry**](#function-add_entry-12) (const DensityMatrix&lt; Torb, Tbase &gt; & density) <br>_Add entry to history, return value is True if energy was lowered._  |
|  bool | [**add\_entry**](#function-add_entry-22) (const DensityMatrix&lt; Torb, Tbase &gt; & density, const FockBuilderReturn&lt; Torb, Tbase &gt; & fock) <br>_Add entry to history, return value is True if energy was lowered._  |
|  void | [**brute\_force\_search\_for\_lowest\_configuration**](#function-brute_force_search_for_lowest_configuration) () <br>_Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search._  |
|  void | [**callback\_function**](#function-callback_function) (std::function&lt; void(const std::map&lt; std::string, std::any &gt; &)&gt; callback\_function=nullptr) <br> |
|  DiagonalizedFockMatrix&lt; Torb, Tbase &gt; | [**compute\_orbitals**](#function-compute_orbitals) (const FockMatrix&lt; Torb &gt; & fock) const<br>_Computes orbitals and orbital energies by diagonalizing the Fock matrix._  |
|  bool | [**converged**](#function-converged) () const<br>_Check if we are converged._  |
|  Tbase | [**convergence\_threshold**](#function-convergence_threshold-12) () const<br>_Get convergence threshold._  |
|  void | [**convergence\_threshold**](#function-convergence_threshold-22) (Tbase convergence\_threshold) <br>_Set verbosity._  |
|  Tbase | [**density\_matrix\_difference**](#function-density_matrix_difference) (size\_t ihist, size\_t jhist) <br>_Density matrix difference norm._  |
|  arma::Col&lt; Tbase &gt; | [**determine\_number\_of\_particles\_by\_aufbau**](#function-determine_number_of_particles_by_aufbau) (const OrbitalEnergies&lt; Tbase &gt; & orbital\_energies) const<br>_Determine number of particles in each block._  |
|  Tbase | [**diis\_diagonal\_damping**](#function-diis_diagonal_damping-12) () const<br>_Damping factor for DIIS diagonal._  |
|  void | [**diis\_diagonal\_damping**](#function-diis_diagonal_damping-22) (Tbase eps) <br>_Damping factor for DIIS diagonal._  |
|  Tbase | [**diis\_epsilon**](#function-diis_epsilon-12) () const<br>_When to start mixing in DIIS._  |
|  void | [**diis\_epsilon**](#function-diis_epsilon-22) (Tbase eps) <br>_When to start mixing in DIIS._  |
|  Tbase | [**diis\_restart\_factor**](#function-diis_restart_factor-12) () const<br>_DIIS restart criterion._  |
|  void | [**diis\_restart\_factor**](#function-diis_restart_factor-22) (Tbase eps) <br>_DIIS restart criterion._  |
|  Tbase | [**diis\_threshold**](#function-diis_threshold-12) () const<br>_When to switch over to DIIS._  |
|  void | [**diis\_threshold**](#function-diis_threshold-22) (Tbase eps) <br>_When to switch over to DIIS._  |
|  std::string | [**error\_norm**](#function-error_norm-12) () const<br>_Get the used error norm._  |
|  void | [**error\_norm**](#function-error_norm-22) (const std::string & error\_norm) <br>_Set the used error norm._  |
|  void | [**fixed\_number\_of\_particles\_per\_block**](#function-fixed_number_of_particles_per_block) (const arma::Col&lt; Tbase &gt; & number\_of\_particles\_per\_block) <br>_Fix the number of occupied orbitals per block._  |
|  bool | [**frozen\_occupations**](#function-frozen_occupations-12) () const<br>_Get frozen occupations._  |
|  void | [**frozen\_occupations**](#function-frozen_occupations-22) (bool frozen) <br>_Set frozen occupations._  |
|  Tbase | [**get\_energy**](#function-get_energy) (size\_t ihist=0) const<br>_Get the energy for the n:th entry._  |
|  FockBuilderReturn&lt; Torb, Tbase &gt; | [**get\_fock\_build**](#function-get_fock_build) (size\_t ihist=0) const<br>_Get the Fock matrix builder return._  |
|  FockMatrix&lt; Torb &gt; | [**get\_fock\_matrix**](#function-get_fock_matrix) (size\_t ihist=0) const<br>_Get the Fock matrix for the ihist:th entry._  |
|  OrbitalOccupations&lt; Tbase &gt; | [**get\_orbital\_occupations**](#function-get_orbital_occupations) (size\_t ihist=0) const<br>_Get the orbital occupations._  |
|  Orbitals&lt; Torb &gt; | [**get\_orbitals**](#function-get_orbitals) (size\_t ihist=0) const<br>_Get the orbitals._  |
|  DensityMatrix&lt; Torb, Tbase &gt; | [**get\_solution**](#function-get_solution) (size\_t ihist=0) const<br>_Get the SCF solution._  |
|  void | [**initialize\_with\_fock**](#function-initialize_with_fock) (const FockMatrix&lt; Torb &gt; & fock\_guess) <br>_Initialize the solver with a guess Fock matrix._  |
|  void | [**initialize\_with\_orbitals**](#function-initialize_with_orbitals) (const Orbitals&lt; Torb &gt; & orbitals, const OrbitalOccupations&lt; Tbase &gt; & orbital\_occupations) <br>_Initialize with precomputed orbitals and occupations._  |
|  int | [**maximum\_history\_length**](#function-maximum_history_length-12) () const<br>_Get maximum\_history\_length._  |
|  void | [**maximum\_history\_length**](#function-maximum_history_length-22) (int maximum\_history\_length) <br>_Set maximum\_history\_length._  |
|  size\_t | [**maximum\_iterations**](#function-maximum_iterations-12) () const<br>_Get the maximum number of iterations._  |
|  void | [**maximum\_iterations**](#function-maximum_iterations-22) (size\_t maxit) <br>_Set the maximum number of iterations._  |
|  Tbase | [**norm**](#function-norm) (const arma::Mat&lt; Tbase &gt; & mat, std::string norm="") const<br>_Evaluate the norm._  |
|  Tbase | [**optimal\_damping\_threshold**](#function-optimal_damping_threshold-12) () const<br>_Use optimal damping when max error bigger than this._  |
|  void | [**optimal\_damping\_threshold**](#function-optimal_damping_threshold-22) (Tbase eps) <br>_Use optimal damping when max error bigger than this._  |
|  arma::uword | [**particle\_block\_offset**](#function-particle_block_offset) (size\_t iparticle) const<br>_Determines the offset for the blocks of the iparticle:th particle._  |
|  void | [**print\_history**](#function-print_history) () const<br>_Print the DIIS history._  |
|  void | [**reset\_history**](#function-reset_history) () <br>_Reset the DIIS history._  |
|  void | [**run**](#function-run) () <br>_Run the SCF._  |
|  void | [**run\_optimal\_damping**](#function-run_optimal_damping) () <br>_Run optimal damping._  |
|  OrbitalOccupations&lt; Tbase &gt; | [**update\_occupations**](#function-update_occupations) (const OrbitalEnergies&lt; Tbase &gt; & orbital\_energies) const<br>_Determines occupations based on the current orbital energies._  |
|  int | [**verbosity**](#function-verbosity-12) () const<br>_Get verbosity._  |
|  void | [**verbosity**](#function-verbosity-22) (int verbosity) <br>_Set verbosity._  |




























## Public Functions Documentation




### function SCFSolver 

_Constructor._ 
```C++
inline OpenOrbitalOptimizer::SCFSolver::SCFSolver (
    const arma::uvec & number_of_blocks_per_particle_type,
    const arma::Col< Tbase > & maximum_occupation,
    const arma::Col< Tbase > & number_of_particles,
    const FockBuilder< Torb, Tbase > & fock_builder,
    const std::vector< std::string > & block_descriptions
) 
```




<hr>



### function add\_entry [1/2]

_Add entry to history, return value is True if energy was lowered._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::add_entry (
    const DensityMatrix< Torb, Tbase > & density
) 
```




<hr>



### function add\_entry [2/2]

_Add entry to history, return value is True if energy was lowered._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::add_entry (
    const DensityMatrix< Torb, Tbase > & density,
    const FockBuilderReturn< Torb, Tbase > & fock
) 
```




<hr>



### function brute\_force\_search\_for\_lowest\_configuration 

_Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::brute_force_search_for_lowest_configuration () 
```




<hr>



### function callback\_function 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::callback_function (
    std::function< void(const std::map< std::string, std::any > &)> callback_function=nullptr
) 
```




<hr>



### function compute\_orbitals 

_Computes orbitals and orbital energies by diagonalizing the Fock matrix._ 
```C++
inline DiagonalizedFockMatrix< Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::compute_orbitals (
    const FockMatrix< Torb > & fock
) const
```




<hr>



### function converged 

_Check if we are converged._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::converged () const
```




<hr>



### function convergence\_threshold [1/2]

_Get convergence threshold._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::convergence_threshold () const
```




<hr>



### function convergence\_threshold [2/2]

_Set verbosity._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::convergence_threshold (
    Tbase convergence_threshold
) 
```




<hr>



### function density\_matrix\_difference 

_Density matrix difference norm._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::density_matrix_difference (
    size_t ihist,
    size_t jhist
) 
```




<hr>



### function determine\_number\_of\_particles\_by\_aufbau 

_Determine number of particles in each block._ 
```C++
inline arma::Col< Tbase > OpenOrbitalOptimizer::SCFSolver::determine_number_of_particles_by_aufbau (
    const OrbitalEnergies< Tbase > & orbital_energies
) const
```




<hr>



### function diis\_diagonal\_damping [1/2]

_Damping factor for DIIS diagonal._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::diis_diagonal_damping () const
```




<hr>



### function diis\_diagonal\_damping [2/2]

_Damping factor for DIIS diagonal._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::diis_diagonal_damping (
    Tbase eps
) 
```




<hr>



### function diis\_epsilon [1/2]

_When to start mixing in DIIS._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::diis_epsilon () const
```




<hr>



### function diis\_epsilon [2/2]

_When to start mixing in DIIS._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::diis_epsilon (
    Tbase eps
) 
```




<hr>



### function diis\_restart\_factor [1/2]

_DIIS restart criterion._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::diis_restart_factor () const
```




<hr>



### function diis\_restart\_factor [2/2]

_DIIS restart criterion._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::diis_restart_factor (
    Tbase eps
) 
```




<hr>



### function diis\_threshold [1/2]

_When to switch over to DIIS._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::diis_threshold () const
```




<hr>



### function diis\_threshold [2/2]

_When to switch over to DIIS._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::diis_threshold (
    Tbase eps
) 
```




<hr>



### function error\_norm [1/2]

_Get the used error norm._ 
```C++
inline std::string OpenOrbitalOptimizer::SCFSolver::error_norm () const
```




<hr>



### function error\_norm [2/2]

_Set the used error norm._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::error_norm (
    const std::string & error_norm
) 
```




<hr>



### function fixed\_number\_of\_particles\_per\_block 

_Fix the number of occupied orbitals per block._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::fixed_number_of_particles_per_block (
    const arma::Col< Tbase > & number_of_particles_per_block
) 
```




<hr>



### function frozen\_occupations [1/2]

_Get frozen occupations._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::frozen_occupations () const
```




<hr>



### function frozen\_occupations [2/2]

_Set frozen occupations._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::frozen_occupations (
    bool frozen
) 
```




<hr>



### function get\_energy 

_Get the energy for the n:th entry._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::get_energy (
    size_t ihist=0
) const
```




<hr>



### function get\_fock\_build 

_Get the Fock matrix builder return._ 
```C++
inline FockBuilderReturn< Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::get_fock_build (
    size_t ihist=0
) const
```




<hr>



### function get\_fock\_matrix 

_Get the Fock matrix for the ihist:th entry._ 
```C++
inline FockMatrix< Torb > OpenOrbitalOptimizer::SCFSolver::get_fock_matrix (
    size_t ihist=0
) const
```




<hr>



### function get\_orbital\_occupations 

_Get the orbital occupations._ 
```C++
inline OrbitalOccupations< Tbase > OpenOrbitalOptimizer::SCFSolver::get_orbital_occupations (
    size_t ihist=0
) const
```




<hr>



### function get\_orbitals 

_Get the orbitals._ 
```C++
inline Orbitals< Torb > OpenOrbitalOptimizer::SCFSolver::get_orbitals (
    size_t ihist=0
) const
```




<hr>



### function get\_solution 

_Get the SCF solution._ 
```C++
inline DensityMatrix< Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::get_solution (
    size_t ihist=0
) const
```




<hr>



### function initialize\_with\_fock 

_Initialize the solver with a guess Fock matrix._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::initialize_with_fock (
    const FockMatrix< Torb > & fock_guess
) 
```




<hr>



### function initialize\_with\_orbitals 

_Initialize with precomputed orbitals and occupations._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::initialize_with_orbitals (
    const Orbitals< Torb > & orbitals,
    const OrbitalOccupations< Tbase > & orbital_occupations
) 
```




<hr>



### function maximum\_history\_length [1/2]

_Get maximum\_history\_length._ 
```C++
inline int OpenOrbitalOptimizer::SCFSolver::maximum_history_length () const
```




<hr>



### function maximum\_history\_length [2/2]

_Set maximum\_history\_length._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::maximum_history_length (
    int maximum_history_length
) 
```




<hr>



### function maximum\_iterations [1/2]

_Get the maximum number of iterations._ 
```C++
inline size_t OpenOrbitalOptimizer::SCFSolver::maximum_iterations () const
```




<hr>



### function maximum\_iterations [2/2]

_Set the maximum number of iterations._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::maximum_iterations (
    size_t maxit
) 
```




<hr>



### function norm 

_Evaluate the norm._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::norm (
    const arma::Mat< Tbase > & mat,
    std::string norm=""
) const
```




<hr>



### function optimal\_damping\_threshold [1/2]

_Use optimal damping when max error bigger than this._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::optimal_damping_threshold () const
```




<hr>



### function optimal\_damping\_threshold [2/2]

_Use optimal damping when max error bigger than this._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::optimal_damping_threshold (
    Tbase eps
) 
```




<hr>



### function particle\_block\_offset 

_Determines the offset for the blocks of the iparticle:th particle._ 
```C++
inline arma::uword OpenOrbitalOptimizer::SCFSolver::particle_block_offset (
    size_t iparticle
) const
```




<hr>



### function print\_history 

_Print the DIIS history._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::print_history () const
```




<hr>



### function reset\_history 

_Reset the DIIS history._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::reset_history () 
```




<hr>



### function run 

_Run the SCF._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::run () 
```




<hr>



### function run\_optimal\_damping 

_Run optimal damping._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::run_optimal_damping () 
```




<hr>



### function update\_occupations 

_Determines occupations based on the current orbital energies._ 
```C++
inline OrbitalOccupations< Tbase > OpenOrbitalOptimizer::SCFSolver::update_occupations (
    const OrbitalEnergies< Tbase > & orbital_energies
) const
```




<hr>



### function verbosity [1/2]

_Get verbosity._ 
```C++
inline int OpenOrbitalOptimizer::SCFSolver::verbosity () const
```




<hr>



### function verbosity [2/2]

_Set verbosity._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::verbosity (
    int verbosity
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

