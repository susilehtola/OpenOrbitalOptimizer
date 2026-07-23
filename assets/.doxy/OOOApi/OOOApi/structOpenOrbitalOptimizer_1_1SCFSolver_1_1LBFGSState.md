

# Struct OpenOrbitalOptimizer::SCFSolver::LBFGSState



[**ClassList**](annotated.md) **>** [**LBFGSState**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1LBFGSState.md)



[More...](#detailed-description)






















## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; [**OrbitalRotation**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalrotation) &gt; | [**history\_dofs**](#variable-history_dofs)  <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**pending\_g**](#variable-pending_g)  <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**pending\_s**](#variable-pending_s)  <br> |
|  std::deque&lt; Tbase &gt; | [**rho**](#variable-rho)  <br> |
|  std::deque&lt; [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; &gt; | [**s**](#variable-s)  <br> |
|  std::deque&lt; [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; &gt; | [**y**](#variable-y)  <br> |












































## Detailed Description


Limited-memory BFGS state retained between lbfgs\_step calls. `s`, `y`, `rho` store the last maximum\_history\_length\_ triples that drive the two-loop recursion (same cap as DIIS  the two share one history-depth knob). `pending_s`, `pending_g` hold the (s, g) recorded at the previous accepted step so the y = g\_new - g\_old pair can be formed on entry to the next call. All members are cleared whenever the orbital basis changes globally (ODA accept), the line search fails, or the DOF set changes. The owning solver only allocates the struct on the first lbfgs\_step() call, so when the user has not enabled L-BFGS the deque headers are not present at all. 


    
## Public Attributes Documentation




### variable history\_dofs 

```C++
std::vector<OrbitalRotation> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::LBFGSState::history_dofs;
```




<hr>



### variable pending\_g 

```C++
Vector<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::LBFGSState::pending_g;
```




<hr>



### variable pending\_s 

```C++
Vector<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::LBFGSState::pending_s;
```




<hr>



### variable rho 

```C++
std::deque<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::LBFGSState::rho;
```




<hr>



### variable s 

```C++
std::deque<Vector<Tbase> > OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::LBFGSState::s;
```




<hr>



### variable y 

```C++
std::deque<Vector<Tbase> > OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::LBFGSState::y;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

