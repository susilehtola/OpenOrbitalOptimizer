

# Class OpenOrbitalOptimizer::SCFSolver

**template &lt;typename Torb, typename Tbase&gt;**



[**ClassList**](annotated.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md)



_SCF solver class._ 

* `#include <scfsolver.hpp>`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**HistorySnapshot**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1HistorySnapshot.md) <br> |
| struct | [**OptionInfo**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1OptionInfo.md) <br>_Descriptor for a single option in the catalog._  |






## Public Attributes

| Type | Name |
| ---: | :--- |
|  bool | [**relocating\_iterate\_**](#variable-relocating_iterate_)   = `false`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**SCFSolver**](#function-scfsolver-12) () = default<br> |
|   | [**SCFSolver**](#function-scfsolver-22) (const [**IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) & number\_of\_blocks\_per\_particle\_type, const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; & maximum\_occupation, const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; & number\_of\_particles, const [**FockBuilder**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilder)&lt; Torb, Tbase &gt; & fock\_builder, const std::vector&lt; std::string &gt; & block\_descriptions) <br> |
|  bool | [**add\_entry**](#function-add_entry-12) (const [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; & density) <br>_Add entry to history, return value is True if energy was lowered._  |
|  bool | [**add\_entry**](#function-add_entry-22) (const [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; & density, const [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt; & fock) <br>_Add entry to history, return value is True if energy was lowered._  |
|  bool | [**aufbau\_cleanup\_step**](#function-aufbau_cleanup_step) (const AllowedMethods & allowed, bool must\_stay\_converged) <br> |
|  Tbase | [**aufbau\_error**](#function-aufbau_error) () const<br> |
|  void | [**brute\_force\_search\_for\_lowest\_configuration**](#function-brute_force_search_for_lowest_configuration) () <br>_Finds the lowest "Aufbau" configuration by moving particles between symmetries by brute force search._  |
|  void | [**callback\_convergence\_function**](#function-callback_convergence_function) (std::function&lt; bool(const std::map&lt; std::string, std::any &gt; &)&gt; callback\_convergence\_function=nullptr) <br> |
|  void | [**callback\_function**](#function-callback_function) (std::function&lt; void(const std::map&lt; std::string, std::any &gt; &)&gt; callback\_function=nullptr) <br> |
|  [**DiagonalizedFockMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-diagonalizedfockmatrix)&lt; Torb, Tbase &gt; | [**compute\_orbitals**](#function-compute_orbitals) (const FockMatrix&lt; Torb &gt; & fock) const<br>_Computes orbitals and orbital energies by diagonalizing the Fock matrix._  |
|  bool | [**converged**](#function-converged) () const<br>_Check if we are converged._  |
|  size\_t | [**degenerate\_cluster\_end\_**](#function-degenerate_cluster_end_) (size\_t start, size\_t n, EnergyAt && energy\_at) const<br> |
|  Tbase | [**density\_matrix\_difference**](#function-density_matrix_difference) (size\_t ihist, size\_t jhist) const<br>_Density matrix difference norm._  |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**determine\_number\_of\_particles\_by\_aufbau**](#function-determine_number_of_particles_by_aufbau) (const [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; & orbital\_energies) const<br>_Determine number of particles in each block._  |
|  Tbase | [**diis\_error\_norm**](#function-diis_error_norm) (size\_t ihist=0) const<br> |
|  Tbase | [**fermi\_level\_error**](#function-fermi_level_error) () const<br> |
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
|  bool | [**has\_logger**](#function-has_logger) () const<br>_True iff a caller-supplied log sink is currently installed._  |
|  void | [**initialize\_with\_fock**](#function-initialize_with_fock) (const FockMatrix&lt; Torb &gt; & fock\_guess) <br>_Initialize the solver with a guess Fock matrix._  |
|  void | [**initialize\_with\_orbitals**](#function-initialize_with_orbitals) (const [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt; & orbitals, const OrbitalOccupations&lt; Tbase &gt; & orbital\_occupations) <br>_Initialize with precomputed orbitals and occupations._  |
|  bool | [**kkt\_occupation\_refinement\_**](#function-kkt_occupation_refinement_) (const AllowedMethods & allowed) <br> |
|  void | [**log\_occupation\_iterate\_**](#function-log_occupation_iterate_) (const char \* tag, int step, Tbase energy\_change) <br> |
|  void | [**logger**](#function-logger) (std::function&lt; void(int, const std::string &)&gt; sink=nullptr) <br> |
|  [**OrbitalHistoryEntry**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalhistoryentry)&lt; Torb, Tbase &gt; | [**make\_history\_entry**](#function-make_history_entry) (const [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; & density\_matrix, const [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt; & fock) const<br> |
|  Tbase | [**norm**](#function-norm) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; Tbase &gt; & mat, std::string norm="") const<br>_Evaluate the norm._  |
|  bool | [**occupations\_converged**](#function-occupations_converged) () const<br> |
|  OrbitalOccupations&lt; Tbase &gt; | [**occupations\_from\_lambda\_**](#function-occupations_from_lambda_) (const SkeletonOccupations & skeletons, const std::vector&lt; std::pair&lt; size\_t, size\_t &gt; &gt; & axis, const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; & lambda) const<br> |
|  std::vector&lt; std::tuple&lt; Tbase, size\_t, size\_t &gt; &gt; | [**order\_orbitals\_by\_energy**](#function-order_orbitals_by_energy) (const [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; & orbital\_energies, size\_t iparticle) const<br> |
|  [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) | [**particle\_block\_offset**](#function-particle_block_offset) (size\_t iparticle) const<br>_Determines the offset for the blocks of the iparticle:th particle._  |
|  Tbase | [**particle\_number\_error**](#function-particle_number_error) () const<br> |
|  void | [**print\_history**](#function-print_history) () const<br>_Print the DIIS history._  |
|  void | [**print\_settings**](#function-print_settings) (std::ostream & os=std::cout) const<br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**project\_occupations\_onto\_polytope\_**](#function-project_occupations_onto_polytope_) (const SkeletonOccupations & skeletons, const std::vector&lt; std::pair&lt; size\_t, size\_t &gt; &gt; & axis, const std::vector&lt; size\_t &gt; & particle\_off, const std::vector&lt; size\_t &gt; & particle\_len) const<br> |
|  void | [**relax\_orbitals\_at\_fixed\_occupations\_**](#function-relax_orbitals_at_fixed_occupations_) (const AllowedMethods & allowed) <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**relaxed\_occupation\_gradient\_**](#function-relaxed_occupation_gradient_) (const SkeletonOccupations & skeletons, const std::vector&lt; std::pair&lt; size\_t, size\_t &gt; &gt; & axis) const<br> |
|  Tbase | [**relaxed\_occupation\_search\_**](#function-relaxed_occupation_search_) (const AllowedMethods & allowed, const SkeletonOccupations & skeletons, const [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt; & orbitals, int & fock\_evaluations) <br> |
|  void | [**reset\_history**](#function-reset_history) () <br>_Reset the DIIS history._  |
|  void | [**restore\_history\_**](#function-restore_history_) (const [**HistorySnapshot**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1HistorySnapshot.md) & snapshot) <br> |
|  void | [**run**](#function-run) () <br> |
|  [**HistorySnapshot**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1HistorySnapshot.md) | [**save\_history\_**](#function-save_history_) () const<br> |
|  void | [**set**](#function-set-14) (const std::string & key, T value) <br> |
|  void | [**set**](#function-set-14) (const std::string & key, T value) <br> |
|  void | [**set**](#function-set-34) (const std::string & key, const std::string & value) <br> |
|  void | [**set**](#function-set-44) (const std::string & key, const char \* value) <br> |
|  void | [**set\_batched\_fock\_builder**](#function-set_batched_fock_builder) ([**BatchedFockBuilder**](namespaceOpenOrbitalOptimizer.md#typedef-batchedfockbuilder)&lt; Torb, Tbase &gt; builder) <br> |
|  void | [**set\_int**](#function-set_int) (const std::string & key, int v) <br>_Set an integer-valued option. Bool-like settings ride here as 0/1._  |
|  void | [**set\_real**](#function-set_real) (const std::string & key, Tbase v) <br>_Set a real-valued option._  |
|  void | [**set\_string**](#function-set_string) (const std::string & key, const std::string & v) <br>_Set a string-valued option._  |
|  void | [**skeleton\_axis\_layout\_**](#function-skeleton_axis_layout_) (const SkeletonOccupations & skeletons, std::vector&lt; size\_t &gt; & particle\_off, std::vector&lt; size\_t &gt; & particle\_len, std::vector&lt; std::pair&lt; size\_t, size\_t &gt; &gt; & axis) const<br> |
|  OrbitalOccupations&lt; Tbase &gt; | [**update\_occupations**](#function-update_occupations) (const [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; & orbital\_energies) const<br>_Determines occupations based on the current orbital energies._  |


## Public Static Functions

| Type | Name |
| ---: | :--- |
|  std::string | [**citation**](#function-citation) () <br> |
|  const std::vector&lt; [**OptionInfo**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1OptionInfo.md) &gt; & | [**options**](#function-options) () <br> |
|  void | [**print\_citation**](#function-print_citation) (std::ostream & os=std::cout) <br>_Print a two-line "please cite" block to_ `os` _._ |


























## Public Attributes Documentation




### variable relocating\_iterate\_ 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::relocating_iterate_;
```



How far the occupations are from Aufbau: the largest occupation sitting above the Fermi level, or missing from below it, measured against the Aufbau filling of the current orbital energies.


[**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals) inside the degenerate cluster the Fermi level lands in are exempt, since that is where fractional occupation is legitimate; the window is the same `optimal_damping_degeneracy_threshold_` the ODA skeleton walk uses. The comparison is against the whole particle type's energy ordering rather than each block's, because the Fermi level is filled across blocks and its degeneracies routinely span them.


Unlike the particle-number error this _is_ a convergence measure: the iterate is a mixed density whose occupations are only Aufbau once the mixing has collapsed onto the minimiser, so this falls as the SCF converges. Returns 0 when the occupations are not the solver's to choose. Set while a routine is moving the iterate to trial points and back rather than starting a fresh calculation. Consulted by initialize\_with\_orbitals, which otherwise resets the Fock-evaluation counter. 


        

<hr>
## Public Functions Documentation




### function SCFSolver [1/2]

```C++
OpenOrbitalOptimizer::SCFSolver::SCFSolver () = default
```



Constructor Default constructor, private and used only by prototype\_(): every setting carries its own initialiser, so a default object describes the catalog correctly. 


        

<hr>



### function SCFSolver [2/2]

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



### function aufbau\_cleanup\_step 

```C++
inline bool OpenOrbitalOptimizer::SCFSolver::aufbau_cleanup_step (
    const AllowedMethods & allowed,
    bool must_stay_converged
) 
```



Replace the converged iterate's occupations with an Aufbau filling, relaxing the orbitals at those occupations before judging whether the swap was worth making.


What the SCF converges to is the natural occupation vector of a _mixed_ density. A mixture of densities carrying different orbitals is not idempotent shell by shell, so a nominally full shell comes out at max\_occ - epsilon and orbitals far above the Fermi level carry epsilon  even though the minimiser of a fractional-occupation functional is Aufbau.


Both halves are needed and neither suffices alone. Choosing occupations is an ODA step over the skeletons with the current density excluded, which keeps the result Aufbau: every skeleton is an Aufbau filling of one common set of orbitals, so any combination of them has those orbitals as its natural orbitals and the combined vector as its occupations, exactly. Judging that vector on orbitals relaxed for the _previous_ occupations costs 1e-4 to 2e-2 Eh on a transition metal, which is more than the swap is worth, so the orbitals are relaxed at the new occupations before any energy is compared.


Where the polytope is one-dimensional the two are coupled directly up to `maximum_modelled_dimension`, see `relaxed_occupation_search_`; beyond that they alternate, which makes this a small SCF restricted to Aufbau-occupied densities.


All of it runs on a scratch iterate seeded from the converged Fock matrix, so the history holds nothing but Aufbau states and its lowest-energy entry is the best of them rather than the mixed density  which is what makes the final comparison possible at all, the history being kept sorted by energy.


The result is adopted only if it does not cost energy. Losing after relaxation would say the mixed density sits below every Aufbau-occupied state, which a variational fractional-occupation functional should not permit, so a rejection is reported rather than passed over in silence. `must_stay_converged` says the caller arrived here converged, and so has a gradient criterion to lose: settling the occupations perturbs the orbitals, and the cleanup trades on energy and occupations while saying nothing about the commutator. It does not veto the swap  an occupation state orders better is not worth refusing over a gradient a few per cent short  but it does buy the orbitals a few relaxation passes at the refined occupations, to win back what can be won. The exits that arrive unconverged have nothing to protect and skip that work. 


        

<hr>



### function aufbau\_error 

```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::aufbau_error () const
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



### function degenerate\_cluster\_end\_ 

```C++
template<typename EnergyAt>
inline size_t OpenOrbitalOptimizer::SCFSolver::degenerate_cluster_end_ (
    size_t start,
    size_t n,
    EnergyAt && energy_at
) const
```



Find the end of the near-degenerate orbital cluster starting at index `start` in an energy-ascending list of `n` orbitals. The cluster is anchored on its first member: it extends while `energy(k) - energy(start) <= optimal_damping_degeneracy_threshold_`. The return value is one past the last member, so the cluster is the half-open range `[start, end)` and is never empty.


Anchoring on the first member rather than on the previous one is what keeps the cluster width bounded by the threshold; a pairwise-gap walk would chain arbitrarily far up a dense ladder of orbitals.


This is the single definition of "degenerate group" in the solver. The ODA skeleton enumeration uses it to decide which orbitals share a fractional filling, and the active-rotation count uses it to size the post-ODA CG burst  the latter has to size the burst _for the clusters the former created_, so the two must agree exactly, boundary included. 


        

<hr>



### function density\_matrix\_difference 

_Density matrix difference norm._ 
```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::density_matrix_difference (
    size_t ihist,
    size_t jhist
) const
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



### function diis\_error\_norm 

```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::diis_error_norm (
    size_t ihist=0
) const
```



Norm of the error vector of the ihist:th history entry  the quantity [**converged()**](classOpenOrbitalOptimizer_1_1SCFSolver.md#function-converged) compares against the threshold.


Defaults to the current iterate. The index is there because picking among stored iterates on their gradients is something the solver already does when a walk stalls, and something a caller may want for the same reason: whether the gradient of a given entry clears the threshold is not always a question the solver's own verdict answers, since the occupation cleanup can spend gradient on settling the occupations and the repair  worth about g^2/2H, far under what a Fock builder reproduces  cannot be verified by any line search. 


        

<hr>



### function fermi\_level\_error 

```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::fermi_level_error () const
```



Residual of the conditions the occupations have to satisfy at a minimum, in energy units.


Minimising the energy over the occupations subject to a fixed particle number and 0 &lt;= n &lt;= n\_max gives, with a chemical potential mu and Janak's dE/dn\_k = eps\_k, 


This returns the largest violation of any of the three over the particle types. `aufbau_error` tests the two inequalities in occupation units and exempts everything inside the Fermi-level cluster, fractional occupation being legitimate exactly there  so the equality, which is the condition that fixes _where_ inside the cluster the fractions sit, went unmeasured.


Measuring it in energy units rather than occupation units is the point. Near a stationary point the energy is quadratic in the occupation displacement while these residuals are linear, so a criterion built on energy decrease is insensitive to exactly the displacements this catches. Measured on a spin-restricted iron atom, two runs of the same command settled at 4s/3d fractions 0.005 apart, 2e-6 Eh apart, both reporting convergence by every other diagnostic the solver carries.


mu is taken as the mean orbital energy over the fractionally occupied orbitals. Where there are none it is not determined by the equality at all, and any value between the highest full and the lowest empty orbital satisfies the inequalities, so the residual is how far those two have crossed. 


        

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



### function has\_logger 

_True iff a caller-supplied log sink is currently installed._ 
```C++
inline bool OpenOrbitalOptimizer::SCFSolver::has_logger () const
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



### function kkt\_occupation\_refinement\_ 

```C++
inline bool OpenOrbitalOptimizer::SCFSolver::kkt_occupation_refinement_ (
    const AllowedMethods & allowed
) 
```



Drive the occupations onto the conditions fermi\_level\_error measures, by Newton on the residual rather than by minimising the energy.


The distinction is the whole point. Every occupation move the solver otherwise makes is accepted on an energy decrease, and near a stationary point the energy is quadratic in the displacement while the residual is linear  so an energy test cannot resolve displacements this one settles directly. On a spin-restricted iron atom the optimal-damping step stopped with a predicted gain of 9.9e-8, just under the 1e-7 below which it refuses to churn, leaving 2e-6 Eh and 0.005 electrons on the table with every other diagnostic reporting convergence.


The unknowns are the fractionally occupied occupations and the chemical potential. Transfers between them conserve particle number by construction, so the step is expressed in the differences d\_m = e\_{k\_m} - e\_{k\_0}: the gradient along one is eps\_{k\_m} - eps\_{k\_0}, free, and the curvature comes from second differences of the perturbatively relaxed energy, one Fock build per point rather than one orbital optimisation.


[**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals) pinned full or empty enter through an active set. A pinned orbital on the wrong side of the chemical potential has a descent direction into the fractional region, so it joins the set and the step is re-solved; one that stays on its own side is at a legitimate bound and is left alone.


The step is chosen by the residual and only _checked_ against the energy: a step that raises it is rejected, so the rule that nothing is adopted unless it lowers the energy survives. 


        

<hr>



### function log\_occupation\_iterate\_ 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::log_occupation_iterate_ (
    const char * tag,
    int step,
    Tbase energy_change
) 
```



Report the iterate the way [**run()**](classOpenOrbitalOptimizer_1_1SCFSolver.md#function-run) reports an SCF iteration.


The cleanup and the occupation refinement move the occupations around after the SCF loop has stopped printing, so without this the phase that decides how a fractionally filled shell is divided is the one phase with no iteration history to read. Same quantities, same order, so the two can be followed as one trace. 


        

<hr>



### function logger 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::logger (
    std::function< void(int, const std::string &)> sink=nullptr
) 
```



Register a log sink. The callback receives `(level, message)` where `level` is the minimum verbosity\_ at which the message would normally print and `message` is the finished, formatted text (newlines included). Pass a default-constructed std::function (or nullptr) to restore the stdout default. 


        

<hr>



### function make\_history\_entry 

```C++
inline OrbitalHistoryEntry < Torb, Tbase > OpenOrbitalOptimizer::SCFSolver::make_history_entry (
    const DensityMatrix < Torb, Tbase > & density_matrix,
    const FockBuilderReturn < Torb, Tbase > & fock
) const
```



Make an orbital history entry, stamping it with a monotonically increasing index.


The index is a per-solver member rather than a function-local static. It was originally a static, which was harmless while the index served only to order the history stack; but the DIIS caches key on it, so it is now correctness-critical that it be unique within a solver. A static is shared by every instance of a given instantiation and `index++` is a non-atomic read-modify-write, so two solvers driven from different threads could lose an update and hand one solver a repeated index  which would make a cache return another entry's commutator and silently corrupt the DIIS extrapolation. 


        

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



### function occupations\_converged 

```C++
inline bool OpenOrbitalOptimizer::SCFSolver::occupations_converged () const
```



Whether the occupations are Aufbau to within `aufbau_convergence_threshold_`. The other half of convergence: `converged()` asks whether the orbitals are at a stationary point, this asks whether the occupations are at one.


Kept separate rather than folded into `converged()` because the two are established differently. The gradient criterion is a pure observation, but occupation space is only put right by `aufbau_cleanup_step`, which needs the gradient to have converged first  so a single predicate that demanded both would be circular, blocking on an error that nothing had yet been allowed to fix. `run()` therefore orders them: gradient, then cleanup, then this. 


        

<hr>



### function occupations\_from\_lambda\_ 

```C++
inline OrbitalOccupations< Tbase > OpenOrbitalOptimizer::SCFSolver::occupations_from_lambda_ (
    const SkeletonOccupations & skeletons,
    const std::vector< std::pair< size_t, size_t > > & axis,
    const Vector < Tbase > & lambda
) const
```



Occupations at a point of that polytope: the promoted skeleton carries the slack 1 - sum(lambda), the axes carry the rest, and a particle offering only one skeleton contributes it whole. 


        

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



### function particle\_number\_error 

```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::particle_number_error () const
```



How far the occupations are from carrying the requested number of particles: the largest `|sum(n) - N|` over the particle types.


This is an invariant rather than a convergence measure. Every step forms densities out of ingredients that already carry the right particle number, so the only way to lose any is for something to discard it, and iterating will not bring it back. It is reported so that a run cannot quietly finish carrying a micro-electron of error, but deliberately does not gate `converged()`: a solver that cannot converge is worse than one that tells you its answer is off. 


        

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



### function project\_occupations\_onto\_polytope\_ 

```C++
inline Vector < Tbase > OpenOrbitalOptimizer::SCFSolver::project_occupations_onto_polytope_ (
    const SkeletonOccupations & skeletons,
    const std::vector< std::pair< size_t, size_t > > & axis,
    const std::vector< size_t > & particle_off,
    const std::vector< size_t > & particle_len
) const
```



The polytope point whose occupations best reproduce those the solver is standing on, in the least-squares sense.


occ(lambda) is affine in lambda  the promoted skeleton plus a combination of the differences to the others  so the fit is a quadratic on the same simplex the energy model is minimised over, and the same QP solves it. 


        

<hr>



### function relax\_orbitals\_at\_fixed\_occupations\_ 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::relax_orbitals_at_fixed_occupations_ (
    const AllowedMethods & allowed
) 
```



Replace the converged iterate's occupations with the Aufbau filling of the converged Fock matrix.


What the SCF reports at convergence is the natural occupation vector of a _mixed_ density. A mixture of densities carrying different orbitals is not idempotent shell by shell, so a nominally full shell comes out at max\_occ - epsilon and orbitals well above the Fermi level carry epsilon  even though the minimiser of a fractional-occupation energy functional is Aufbau: full below the Fermi level, zero above it, fractional only inside the degenerate cluster at it.


This is one ODA step with the current density left out of the polytope. That is all it takes, because the mixing is the whole problem: every skeleton is an Aufbau filling of one common set of orbitals, so any combination of skeletons alone has those same orbitals as its natural orbitals and the combined occupation vector as its occupations, exactly. The Aufbau structure is inherited rather than imposed, and the Fermi-level fractions come from minimising the energy over the skeleton simplex rather than from a filling rule  which matters, since that is the one place where the occupations are genuinely free.


Being an ODA step, it is adopted only if it lowers the energy; a cleanup that raised it would mean the converged iterate was not the Aufbau minimiser it is reported to be, which is worth leaving visible rather than papering over. Relax the orbitals at the occupations the iterate already carries, the way the state machine does after an ODA step. Runs the same burst, and stops early once a rotation step stops descending. 


        

<hr>



### function relaxed\_occupation\_gradient\_ 

```C++
inline Vector < Tbase > OpenOrbitalOptimizer::SCFSolver::relaxed_occupation_gradient_ (
    const SkeletonOccupations & skeletons,
    const std::vector< std::pair< size_t, size_t > > & axis
) const
```



Gradient of the relaxed energy with respect to those axes, at the current iterate. The one-dimensional case of this is Free of any orbital-response term: at an orbital-stationary point that term carries a factor dE/dkappa = 0, and Hellmann-Feynman is all that is left. 


        

<hr>



### function relaxed\_occupation\_search\_ 

```C++
inline Tbase OpenOrbitalOptimizer::SCFSolver::relaxed_occupation_search_ (
    const AllowedMethods & allowed,
    const SkeletonOccupations & skeletons,
    const Orbitals < Torb > & orbitals,
    int & fock_evaluations
) 
```



Minimise the relaxed energy over a skeleton polytope of any dimension, and leave the iterate at the best point found.


The relaxed Hessian 


is what governs where the minimum sits, and it is never formed here  the Schur complement would need the orbital response equations the solver does not have. It does not need to be. The object being minimised is the reduced surface E\_relaxed(lambda) = min\_kappa E(lambda, kappa), whose gradient is free at any relaxed point, so relaxing at the lambda = 0 vertex and at each axis vertex gives H\_relaxed column by column as a difference of gradients: 


That is the whole generalisation. The energies come along for the ride and are kept as candidates. Cost is one relaxation per vertex plus one at the predicted minimum, which is why the caller caps the dimension it will attempt this at. 


        

<hr>



### function reset\_history 

_Reset the DIIS history._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::reset_history () 
```




<hr>



### function restore\_history\_ 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::restore_history_ (
    const HistorySnapshot & snapshot
) 
```




<hr>



### function run 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::run () 
```



Run the SCF


Consumes the `methods` string setting, a `+`-separated case-insensitive list drawn from `"DIIS"` (Pulay's A/EDIIS-bracketed direct inversion in the iterative subspace), `"LCIIS"` (Li & Yaron's least-squares commutator variant of the same extrapolation step  it replaces the CDIIS coefficients and implies `"DIIS"`, so asking for both is an error rather than a silent preference), `"ODA"` (optimal-damping polytope step on the skeleton density matrices), and `"CG"` (preconditioned PR+ scaled steepest descent on orbital rotations at fixed occupations). The orbital-rotation slot also accepts `"LBFGS"` and `"ARH"` (augmented Roothaan-Hall: the Roothaan-Hall diagonal made exact on the directions the density history spans). Configure via `set ("methods", ...)`; default is `"DIIS + ODA + LBFGS"`. Examples:


`"DIIS"` pure A/EDIIS extrapolation `"LCIIS"` least-squares commutator extrapolation `"ODA"` standalone polytope minimisation `"DIIS + ODA + LBFGS"` full compound algorithm (default) `"DIIS + ODA + CG"` PR+ CG in place of L-BFGS `"DIIS + ODA + ARH"` augmented Roothaan-Hall rotations `"ODA + CG"` DIIS-less compound


State-transition rules: from DIIS we leave to ODA (or to CG when ODA is not allowed) on stall or large error; from ODA we hand to DIIS on integer occupations or to CG on fractional / failed occupations; from CG we burst `orbital_rotation_steps_after_oda_` (or the polytope dimension when that is left at zero) steps and then hand back to DIIS. The state-machine collapses gracefully when only a subset of the methods is allowed: `"DIIS"` alone keeps retrying DIIS until `maximum_iterations_` runs out; other subsets terminate early when every allowed method has failed in succession. 


        

<hr>



### function save\_history\_ 

```C++
inline HistorySnapshot OpenOrbitalOptimizer::SCFSolver::save_history_ () const
```




<hr>



### function set [1/4]

```C++
template<typename T, std::enable_if_t< std::is_integral_v< T >, int >>
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    T value
) 
```



Set an option, dispatching on the argument type: integral arguments go to `set_int`, floating-point (and `Tbase`) arguments to `set_real`, strings to `set_string`.


These are SFINAE-constrained templates rather than plain overloads on `(Tbase)` and `(int)`: with plain overloads a literal like `1e-9` converts to both `int` and a non-double `Tbase` at the same rank, so `set ("convergence_threshold", 1e-9)` was ambiguous — i.e. it did not compile at all — for the `float` and `_Float128` instantiations, and `set(key, 100u)` was ambiguous for every instantiation. Dispatching on `is_integral` removes the tie. 


        

<hr>



### function set [1/4]

```C++
template<typename T, std::enable_if_t<!std::is_integral_v< T > &&(std::is_floating_point_v< T >||std::is_same_v< T, Tbase >), int >>
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    T value
) 
```




<hr>



### function set [3/4]

```C++
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    const std::string & value
) 
```




<hr>



### function set [4/4]

```C++
inline void OpenOrbitalOptimizer::SCFSolver::set (
    const std::string & key,
    const char * value
) 
```



String-literal overload; without it a `const char *` argument would not match the `std::string` overload any better than the numeric templates reject it, and the diagnostic would be poor. 


        

<hr>



### function set\_batched\_fock\_builder 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::set_batched_fock_builder (
    BatchedFockBuilder < Torb, Tbase > builder
) 
```



Register a batched Fock builder. When set, optimal\_damping\_step uses it for the axis-vertex sweep, sharing integral / grid setup across the N\_par builds. The single-density fock\_builder remains in use for mixed-density trials (model minimum, cubic edges, backoff scales). Passing a default-constructed std::function clears the override and restores the loop-over- fock\_builder default. 


        

<hr>



### function set\_int 

_Set an integer-valued option. Bool-like settings ride here as 0/1._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::set_int (
    const std::string & key,
    int v
) 
```




<hr>



### function set\_real 

_Set a real-valued option._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::set_real (
    const std::string & key,
    Tbase v
) 
```




<hr>



### function set\_string 

_Set a string-valued option._ 
```C++
inline void OpenOrbitalOptimizer::SCFSolver::set_string (
    const std::string & key,
    const std::string & v
) 
```




<hr>



### function skeleton\_axis\_layout\_ 

```C++
inline void OpenOrbitalOptimizer::SCFSolver::skeleton_axis_layout_ (
    const SkeletonOccupations & skeletons,
    std::vector< size_t > & particle_off,
    std::vector< size_t > & particle_len,
    std::vector< std::pair< size_t, size_t > > & axis
) const
```



Axis layout of the skeleton polytope once the first skeleton of each particle is promoted to the lambda = 0 vertex: one axis per skeleton after that, and `axis[k]` names the (particle, skeleton) the k'th axis carries. Matches what `optimal_damping_step_` searches under `exclude_reference`, so the same QP can minimise over it. 


        

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

```C++
static inline const std::vector< OptionInfo > & OpenOrbitalOptimizer::SCFSolver::options () 
```



Enumerate every option the solver understands, in declaration order. Read straight off the settings themselves, so it cannot drift out of step with what set\_\* and get\_\* accept.


Static, so callers can inspect the catalog without building a solver  the Python layer does exactly that. The settings own their values, so describing them needs _an_ object; a private default-constructed prototype supplies one. Every setting carries its own default initialiser, so the prototype has the right metadata even though its other members are empty. 


        

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

