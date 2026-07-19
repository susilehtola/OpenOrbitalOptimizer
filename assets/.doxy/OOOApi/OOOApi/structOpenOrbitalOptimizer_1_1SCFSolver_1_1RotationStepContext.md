

# Struct OpenOrbitalOptimizer::SCFSolver::RotationStepContext



[**ClassList**](annotated.md) **>** [**RotationStepContext**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1RotationStepContext.md)



[More...](#detailed-description)






















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt; | [**C\_pseudo**](#variable-c_pseudo)  <br> |
|  Tbase | [**E\_ref**](#variable-e_ref)   = `0`<br> |
|  std::vector&lt; [**OrbitalRotation**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalrotation) &gt; | [**dofs**](#variable-dofs)  <br> |
|  std::vector&lt; [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; &gt; | [**eps**](#variable-eps)  <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**g**](#variable-g)  <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; Tbase &gt; | [**h**](#variable-h)  <br> |
|  bool | [**is\_complex**](#variable-is_complex)   = `false`<br> |
|  size\_t | [**n\_dof**](#variable-n_dof)   = `0`<br> |
|  size\_t | [**n\_par**](#variable-n_par)   = `0`<br> |
|  OrbitalOccupations&lt; Tbase &gt; | [**n\_ref**](#variable-n_ref)  <br> |












































## Detailed Description


Take one preconditioned scaled-steepest-descent step on the orbital-rotation manifold. Pseudo-diagonalizes the reference Fock matrix within each equal-occupation sub-block to obtain canonical-orbital energy estimates; uses those for a Newton- like diagonal-Hessian preconditioner so that rotations whose natural energy scales span many orders of magnitude are still scaled appropriately. Line-searches along the unitary curve C(t) = C\_pseudo \* exp(t K) by parabolic-fit refinement. Returns true if a strictly lower-energy entry was added to the history, false on stall (no descent, no rotation degrees of freedom, or line search exhausted). 


    
## Public Attributes Documentation




### variable C\_pseudo 

```C++
Orbitals<Torb> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::C_pseudo;
```




<hr>



### variable E\_ref 

```C++
Tbase OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::E_ref;
```




<hr>



### variable dofs 

```C++
std::vector<OrbitalRotation> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::dofs;
```




<hr>



### variable eps 

```C++
std::vector<Vector<Tbase> > OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::eps;
```




<hr>



### variable g 

```C++
Vector<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::g;
```




<hr>



### variable h 

```C++
Vector<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::h;
```




<hr>



### variable is\_complex 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::is_complex;
```




<hr>



### variable n\_dof 

```C++
size_t OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::n_dof;
```




<hr>



### variable n\_par 

```C++
size_t OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::n_par;
```




<hr>



### variable n\_ref 

```C++
OrbitalOccupations<Tbase> OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::RotationStepContext::n_ref;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

