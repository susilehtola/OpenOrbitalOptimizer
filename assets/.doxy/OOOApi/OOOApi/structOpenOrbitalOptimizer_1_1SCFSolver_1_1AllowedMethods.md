

# Struct OpenOrbitalOptimizer::SCFSolver::AllowedMethods



[**ClassList**](annotated.md) **>** [**AllowedMethods**](structOpenOrbitalOptimizer_1_1SCFSolver_1_1AllowedMethods.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  bool | [**cg**](#variable-cg)   = `false`<br> |
|  bool | [**diis**](#variable-diis)   = `false`<br> |
|  bool | [**lbfgs**](#variable-lbfgs)   = `false`<br> |
|  bool | [**lciis**](#variable-lciis)   = `false`<br> |
|  bool | [**oda**](#variable-oda)   = `false`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**any**](#function-any) () const<br> |
|  bool | [**orbital\_rotation**](#function-orbital_rotation) () const<br> |




























## Public Attributes Documentation




### variable cg 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::AllowedMethods::cg;
```




<hr>



### variable diis 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::AllowedMethods::diis;
```




<hr>



### variable lbfgs 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::AllowedMethods::lbfgs;
```




<hr>



### variable lciis 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::AllowedMethods::lciis;
```



LCIIS is a variant of the extrapolation step, not a step of its own: the "LCIIS" token sets `diis` as well, so every existing `allowed.diis` gate keeps working and this flag only selects which coefficients the extrapolation uses. 


        

<hr>



### variable oda 

```C++
bool OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::AllowedMethods::oda;
```




<hr>
## Public Functions Documentation




### function any 

```C++
inline bool AllowedMethods::any () const
```




<hr>



### function orbital\_rotation 

```C++
inline bool AllowedMethods::orbital_rotation () const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

