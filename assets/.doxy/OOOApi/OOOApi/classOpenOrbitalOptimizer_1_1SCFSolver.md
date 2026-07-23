

# Class OpenOrbitalOptimizer::SCFSolver

**template &lt;typename Torb, typename Tbase&gt;**



[**ClassList**](annotated.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md)



_SCF solver class._ 

* `#include <scfsolver.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**OptionInfo**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1OptionInfo.md) <br>_Descriptor for a single option in the catalog._  |






















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**SCFSolver**](#function-scfsolver) (const [**IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) & number\_of\_blocks\_per\_particle\_type, const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; & maximum\_occupation, const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; & number\_of\_particles, const [**FockBuilder**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilder)&lt; Torb, Tbase &gt; & fock\_builder, const std::vector&lt; std::string &gt; & block\_descriptions) <br>_Constructor._  |
|  bool | [**add\_entry**](#function-add_entry-12) (const [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; & density) <br>_Add entry to history, return value is True if energy was lowered._  |
|  bool | [**add\_entry**](#function-add_entry-22) (const [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; & density, const [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt; & fock) <br>_Add entry to history, return value is True if energy was lowered._  |
|  void | [**brute\_force\_search\_for\_lowest\_configuration**](#function-brute_force_search_for_lowest_configuration) () <br>_Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search._  |
|  void | [**callback\_convergence\_function**](#function-callback_convergence_function) (std::function&lt; bool(const std::map&lt; std::string, std::any &gt; &)&gt; callback\_convergence\_function=nullptr) <br> |
|  void | [**callback\_function**](#function-callback_function) (std::function&lt; void(const std::map&lt; std::string, std::any &gt; &)&gt; callback\_function=nullptr) <br> |
|  [**DiagonalizedFockMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-diagonalizedfockmatrix)&lt; Torb, Tbase &gt; | [**compute\_orbitals**](#function-compute_orbitals) (const FockMatrix&lt; Torb &gt; & fock) const<br>_Computes orbitals and orbital energies by diagonalizing the Fock matrix._  |
|  bool | [**converged**](#function-converged) () const<br>_Check if we are converged._  |
|  Tbase | [**density\_matrix\_difference**](#function-density_matrix_difference) (size\_t ihist, size\_t jhist) <br>_Density matrix difference norm._  |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**determine\_number\_of\_particles\_by\_aufbau**](#function-determine_number_of_particles_by_aufbau) (const [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; & orbital\_energies) const<br>_Determine number of particles in each block._  |
|  void | [**fixed\_number\_of\_particles\_per\_block**](#function-fixed_number_of_particles_per_block) (const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; & number\_of\_particles\_per\_block) <br>_Fix the number of occupied orbitals per block._  |
|  Tbase | [**get\_energy**](#function-get_energy) (size\_t ihist=0) const<br>_Get the energy for the n:th entry._  |
|  [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt; | [**get\_fock\_build**](#function-get_fock_build) (size\_t ihist=0) const<br>_Get the Fock matrix builder return._  |
|  FockMatrix&lt; Torb &gt; | [**get\_fock\_matrix**](#function-get_fock_matrix) (size\_t ihist=0) const<br>_Get the Fock matrix for the ihist:th entry._  |
|  int | [**get\_int**](#function-get_int) (const std::string & key) const<br>_Get an integer-valued option or diagnostic._  |
|  OrbitalOccupations&lt; Tbase &gt; | [**get\_orbital\_occupations**](#function-get_orbital_occupations) (size\_t ihist=0) const<br>_Get the orbital occupations._  |
|  [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt; | [**get\_orbitals**](#function-get_orbitals) (size\_t ihist=0) const<br>_Get the orbitals._  |
|  Tbase | [**get\_real**](#function-get_real) (const std::string & key) const<br>_Get a real-valued option or diagnostic._  |
|  [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; | [**get\_solution**](#function-get_solution) (size\_t ihist=0) const<br>_Get the SCF solution._  |
|  std::string | [**get\_string**](#function-get_string) (const std::string & key) const<br>_Get a string-valued option._  |
|  bool | [**has\_batched\_fock\_builder**](#function-has_batched_fock_builder) () const<br>_Whether a batched Fock builder is registered._  |
|  void | [**initialize\_with\_fock**](#function-initialize_with_fock) (const FockMatrix&lt; Torb &gt; & fock\_guess) <br>_Initialize the solver with a guess Fock matrix._  |
|  void | [**initialize\_with\_orbitals**](#function-initialize_with_orbitals) (const [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt; & orbitals, const OrbitalOccupations&lt; Tbase &gt; & orbital\_occupations) <br>_Initialize with precomputed orbitals and occupations._  |
|  Tbase | [**norm**](#function-norm) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; Tbase &gt; & mat, std::string norm="") const<br>_Evaluate the norm._  |
|  std::vector&lt; std::tuple&lt; Tbase, size\_t, size\_t &gt; &gt; | [**order\_orbitals\_by\_energy**](#function-order_orbitals_by_energy) (const [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; & orbital\_energies, size\_t iparticle) const<br> |
|  [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) | [**particle\_block\_offset**](#function-particle_block_offset) (size\_t iparticle) const<br>_Determines the offset for the blocks of the iparticle:th particle._  |
|  void | [**print\_history**](#function-print_history) () const<br>_Print the DIIS history._  |
|  void | [**print\_settings**](#function-print_settings) (std::ostream & os=std::cout) const<br> |
|  void | [**reset\_history**](#function-reset_history) () <br>_Reset the DIIS history._  |
|  void | [**run**](#function-run) () <br> |
|  void | [**set**](#function-set-13) (const std::string & key, Tbase v) <br>_Set a real-valued option._  |
|  void | [**set**](#function-set-23) (const std::string & key, int v) <br>_Set an integer-valued option. Bool settings ride here as 0/1._  |
|  void | [**set**](#function-set-33) (const std::string & key, const std::string & v) <br>_Set a string-valued option._  |
|  void | [**set\_batched\_fock\_builder**](#function-set_batched_fock_builder) ([**BatchedFockBuilder**](namespaceOpenOrbitalOptimizer.md#typedef-batchedfockbuilder)&lt; Torb, Tbase &gt; builder) <br> |
|  OrbitalOccupations&lt; Tbase &gt; | [**update\_occupations**](#function-update_occupations) (const [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; & orbital\_energies) const<br>_Determines occupations based on the current orbital energies._  |


## Public Static Functions

| Type | Name |
| ---: | :--- |
|  std::string | [**citation**](#function-citation) () <br> |
|  const std::vector&lt; [**OptionInfo**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1OptionInfo.md) &gt; & | [**options**](#function-options) () <br>_Enumerate every option the solver understands._  |
|  void | [**print\_citation**](#function-print_citation) (std::ostream & os=std::cout) <br>_Print a two-line "please cite" block to_ `os` _._ |


























## Public Functions Documentation




### function SCFSolver 

_Constructor._ 
```C++
inline OpenOrbitalOptimizer::SCFSolver::SCFSolver (
    const IndexVector & number_of_blocks_per_particle_type,
    const Vector < Tbase > & maximum_occupation,
    const Vector < Tbase > & number_of_particles,
    const FockBuilder < Torb, Tbase > & fock_builder,
    const std::vector< std::string > & block_descriptions
) 
```




<hr>



### function add\_entry [1/2]

_Add entry to history, return value is True if energy was lowered._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::add_entry (
    const DensityMatrix < Torb, Tbase > & density
) 
```




<hr>



### function add\_entry [2/2]

_Add entry to history, return value is True if energy was lowered._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::add_entry (
    const DensityMatrix < Torb, Tbase > & density,
    const FockBuilderReturn < Torb, Tbase > & fock
) 
```




<hr>



### function brute\_force\_search\_for\_lowest\_configuration 

_Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::brute_force_search_for_lowest_configuration () 
```




<hr>



### function callback\_convergence\_function 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::callback_convergence_function (
    std::function< bool(const std::map< std::string, std::any > &)> callback_convergence_function=nullptr
) 
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
inline DiagonalizedFockMatrix < Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::compute_orbitals (
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
inline Vector < Tbase > OpenOrbitalOptimizer::SCFSolver::determine_number_of_particles_by_aufbau (
    const OrbitalEnergies < Tbase > & orbital_energies
) const
```




<hr>



### function fixed\_number\_of\_particles\_per\_block 

_Fix the number of occupied orbitals per block._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::fixed_number_of_particles_per_block (
    const Vector < Tbase > & number_of_particles_per_block
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
inline FockBuilderReturn < Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::get_fock_build (
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



### function get\_int 

_Get an integer-valued option or diagnostic._ 
```C++
inline int OpenOrbitalOptimizer::SCFSolver::get_int (
    const std::string & key
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
inline Orbitals < Torb > OpenOrbitalOptimizer::SCFSolver::get_orbitals (
    size_t ihist=0
) const
```




<hr>



### function get\_real 

_Get a real-valued option or diagnostic._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::get_real (
    const std::string & key
) const
```




<hr>



### function get\_solution 

_Get the SCF solution._ 
```C++
inline DensityMatrix < Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::get_solution (
    size_t ihist=0
) const
```




<hr>



### function get\_string 

_Get a string-valued option._ 
```C++
inline std::string OpenOrbitalOptimizer::SCFSolver::get_string (
    const std::string & key
) const
```




<hr>



### function has\_batched\_fock\_builder 

_Whether a batched Fock builder is registered._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::has_batched_fock_builder () const
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
    const Orbitals < Torb > & orbitals,
    const OrbitalOccupations< Tbase > & orbital_occupations
) 
```




<hr>



### function norm 

_Evaluate the norm._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::norm (
    const Matrix < Tbase > & mat,
    std::string norm=""
) const
```




<hr>



### function order\_orbitals\_by\_energy 

```C++
inline std::vector< std::tuple< Tbase, size_t, size_t > > OpenOrbitalOptimizer::SCFSolver::order_orbitals_by_energy (
    const OrbitalEnergies < Tbase > & orbital_energies,
    size_t iparticle
) const
```



Collect orbital energies for a given particle type, sorted in increasing energy. Each tuple holds (energy, iblock, iorb). 


        

<hr>



### function particle\_block\_offset 

_Determines the offset for the blocks of the iparticle:th particle._ 
```C++
inline Index OpenOrbitalOptimizer::SCFSolver::particle_block_offset (
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



### function print\_settings 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::print_settings (
    std::ostream & os=std::cout
) const
```



Print every catalog entry with its current value to `os`. Read-only diagnostics that require a populated orbital history (converged; anything derived from the current Fock) print as "n/a" before the first `initialize_with_*`. 


        

<hr>



### function reset\_history 

_Reset the DIIS history._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::reset_history () 
```




<hr>



### function run 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::run () 
```



Run the SCF


Consumes the `methods` string setting, a `+`-separated case-insensitive list drawn from `"DIIS"` (Pulay's A/EDIIS-bracketed direct inversion in the iterative subspace), `"ODA"` (optimal-damping polytope step on the skeleton density matrices), and `"CG"` (preconditioned PR+ scaled steepest descent on orbital rotations at fixed occupations). Configure via `set ("methods", ...)`; default is `"DIIS + ODA + CG"`. Examples:


`"DIIS"` pure A/EDIIS extrapolation `"ODA"` standalone polytope minimisation `"DIIS + ODA + CG"` full compound algorithm (default) `"ODA + CG"` DIIS-less compound `"DIIS + ODA + LBFGS"` L-BFGS in place of PR+ CG (when both `CG` and `LBFGS` are listed L-BFGS is preferred)


State-transition rules: from DIIS we leave to ODA (or to CG when ODA is not allowed) on stall or large error; from ODA we hand to DIIS on integer occupations or to CG on fractional / failed occupations; from CG we burst `orbital_rotation_steps_after_oda_` (or the polytope dimension when that is left at zero) steps and then hand back to DIIS. The state-machine collapses gracefully when only a subset of the methods is allowed: `"DIIS"` alone keeps retrying DIIS until `maximum_iterations_` runs out; other subsets terminate early when every allowed method has failed in succession. 


        

<hr>



### function set [1/3]

_Set a real-valued option._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    Tbase v
) 
```




<hr>



### function set [2/3]

_Set an integer-valued option. Bool settings ride here as 0/1._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    int v
) 
```




<hr>



### function set [3/3]

_Set a string-valued option._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    const std::string & v
) 
```




<hr>



### function set\_batched\_fock\_builder 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::set_batched_fock_builder (
    BatchedFockBuilder < Torb, Tbase > builder
) 
```



Register a batched Fock builder. When set, optimal\_damping\_step uses it for the axis-vertex sweep, sharing integral / grid setup across the N\_par builds. The single-density fock\_builder remains in use for mixed-density trials (model minimum, cubic edges, backoff scales). Passing a default-constructed std::function clears the override and restores the loop-over- fock\_builder default. 


        

<hr>



### function update\_occupations 

_Determines occupations based on the current orbital energies._ 
```C++
inline OrbitalOccupations< Tbase > OpenOrbitalOptimizer::SCFSolver::update_occupations (
    const OrbitalEnergies < Tbase > & orbital_energies
) const
```




<hr>
## Public Static Functions Documentation




### function citation 

```C++
static inline std::string OpenOrbitalOptimizer::SCFSolver::citation () 
```



Canonical citation for the library. Downstream drivers should forward this to their users; the string is deliberately kept as a single line so it wraps cleanly in log output. 


        

<hr>



### function options 

_Enumerate every option the solver understands._ 
```C++
static inline const std::vector< OptionInfo > & OpenOrbitalOptimizer::SCFSolver::options () 
```




<hr>



### function print\_citation 

_Print a two-line "please cite" block to_ `os` _._
```C++
static inline void OpenOrbitalOptimizer::SCFSolver::print_citation (
    std::ostream & os=std::cout
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

