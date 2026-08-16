

# Struct OpenOrbitalOptimizer::SCFSolver::HistorySnapshot



[**ClassList**](annotated.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) **>** [**HistorySnapshot**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1HistorySnapshot.md)



[More...](#detailed-description)

* `#include <scfsolver.hpp>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::map&lt; size\_t, std::vector&lt; [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; Torb &gt; &gt; &gt; | [**commutators**](#variable-commutators)  <br> |
|  std::map&lt; std::pair&lt; size\_t, size\_t &gt;, Tbase &gt; | [**density\_diff**](#variable-density_diff)  <br> |
|  std::map&lt; std::pair&lt; size\_t, size\_t &gt;, Tbase &gt; | [**diis\_matrix**](#variable-diis_matrix)  <br> |
|  OrbitalHistory&lt; Torb, Tbase &gt; | [**history**](#variable-history)  <br> |
|  bool | [**last\_oda\_collapsed**](#variable-last_oda_collapsed)   = `false`<br> |
|  LBFGSState | [**lbfgs**](#variable-lbfgs)  <br> |
|  size\_t | [**next\_index**](#variable-next_index)   = `0`<br> |
|  Tbase | [**old\_energy**](#variable-old_energy)   = `0`<br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**previous\_direction**](#variable-previous_direction)  <br> |
|  std::vector&lt; [**OrbitalRotation**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalrotation) &gt; | [**previous\_dofs**](#variable-previous_dofs)  <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**previous\_gradient**](#variable-previous_gradient)  <br> |
|  std::map&lt; std::pair&lt; size\_t, size\_t &gt;, Tbase &gt; | [**trace\_DF**](#variable-trace_df)  <br> |












































## Detailed Description


Everything the history-based methods carry between steps.


initialize\_with\_orbitals clears the orbital history and the DIIS caches and starts again from the state it is handed, which is what a routine wanting to move the iterate has to use. At the end of a run that costs nothing. Called from inside the SCF it leaves the state machine running on extrapolation and curvature histories that no longer describe where the iterate has been  measured, that segfaulted on the next rotation step. Saving and restoring around such a routine keeps the relocation local to it. 


    
## Public Attributes Documentation




### variable commutators 

```C++
std::map<size_t, std::vector<Matrix<Torb> > > OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::commutators;
```




<hr>



### variable density\_diff 

```C++
std::map<std::pair<size_t, size_t>, Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::density_diff;
```




<hr>



### variable diis\_matrix 

```C++
std::map<std::pair<size_t, size_t>, Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::diis_matrix;
```




<hr>



### variable history 

```C++
OrbitalHistory<Torb, Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::history;
```




<hr>



### variable last\_oda\_collapsed 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::last_oda_collapsed;
```




<hr>



### variable lbfgs 

```C++
LBFGSState OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::lbfgs;
```




<hr>



### variable next\_index 

```C++
size_t OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::next_index;
```




<hr>



### variable old\_energy 

```C++
Tbase OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::old_energy;
```




<hr>



### variable previous\_direction 

```C++
Vector<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::previous_direction;
```




<hr>



### variable previous\_dofs 

```C++
std::vector<OrbitalRotation> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::previous_dofs;
```




<hr>



### variable previous\_gradient 

```C++
Vector<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::previous_gradient;
```




<hr>



### variable trace\_DF 

```C++
std::map<std::pair<size_t, size_t>, Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::HistorySnapshot::trace_DF;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

