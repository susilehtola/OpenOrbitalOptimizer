

# Namespace OpenOrbitalOptimizer



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md)


















## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**ConjugateGradients**](namespaceOpenOrbitalOptimizer_1_1ConjugateGradients.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| class | [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) &lt;typename Torb, typename Tbase&gt;<br>_SCF solver class._  |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::pair&lt; Orbitals&lt; Torb &gt;, OrbitalOccupations&lt; Tbase &gt; &gt; | [**DensityMatrix**](#typedef-densitymatrix)  <br>_The pair of orbitals and occupations defines the density matrix._  |
| typedef arma::Mat&lt; T &gt; | [**DiagonalOrbitalHessianBlock**](#typedef-diagonalorbitalhessianblock)  <br> |
| typedef std::vector&lt; DiagonalOrbitalHessianBlock&lt; T &gt; &gt; | [**DiagonalOrbitalHessians**](#typedef-diagonalorbitalhessians)  <br> |
| typedef std::pair&lt; Orbitals&lt; Torb &gt;, OrbitalEnergies&lt; Tbase &gt; &gt; | [**DiagonalizedFockMatrix**](#typedef-diagonalizedfockmatrix)  <br>_The return of Fock matrix diagonalization is._  |
| typedef std::function&lt; FockBuilderReturn&lt; Torb, Tbase &gt;(const DensityMatrix&lt; Torb, Tbase &gt; &)&gt; | [**FockBuilder**](#typedef-fockbuilder)  <br> |
| typedef std::pair&lt; Tbase, FockMatrix&lt; Torb &gt; &gt; | [**FockBuilderReturn**](#typedef-fockbuilderreturn)  <br> |
| typedef std::vector&lt; FockMatrixBlock&lt; T &gt; &gt; | [**FockMatrix**](#typedef-fockmatrix)  <br>_The whole set of Fock matrices is a vector of blocks._  |
| typedef arma::Mat&lt; T &gt; | [**FockMatrixBlock**](#typedef-fockmatrixblock)  <br> |
| typedef arma::Mat&lt; T &gt; | [**OrbitalBlock**](#typedef-orbitalblock)  <br> |
| typedef arma::Col&lt; T &gt; | [**OrbitalBlockOccupations**](#typedef-orbitalblockoccupations)  <br>_The occupations for each orbitals are floating point numbers._  |
| typedef std::vector&lt; arma::Col&lt; T &gt; &gt; | [**OrbitalEnergies**](#typedef-orbitalenergies)  <br>_Orbital energies are stored as a vector of vectors._  |
| typedef arma::Mat&lt; T &gt; | [**OrbitalGradientBlock**](#typedef-orbitalgradientblock)  <br> |
| typedef std::vector&lt; OrbitalGradientBlock&lt; T &gt; &gt; | [**OrbitalGradients**](#typedef-orbitalgradients)  <br> |
| typedef std::vector&lt; OrbitalHistoryEntry&lt; Torb, Tbase &gt; &gt; | [**OrbitalHistory**](#typedef-orbitalhistory)  <br>_The history is then a vector._  |
| typedef std::tuple&lt; DensityMatrix&lt; Torb, Tbase &gt;, FockBuilderReturn&lt; Torb, Tbase &gt;, size\_t &gt; | [**OrbitalHistoryEntry**](#typedef-orbitalhistoryentry)  <br> |
| typedef std::vector&lt; OrbitalBlockOccupations&lt; T &gt; &gt; | [**OrbitalOccupations**](#typedef-orbitaloccupations)  <br> |
| typedef std::tuple&lt; size\_t, arma::uword, arma::uword &gt; | [**OrbitalRotation**](#typedef-orbitalrotation)  <br>_List of orbital rotation angles: block index and orbital indices._  |
| typedef std::vector&lt; OrbitalBlock&lt; T &gt; &gt; | [**Orbitals**](#typedef-orbitals)  <br> |
















































## Public Types Documentation




### typedef DensityMatrix 

_The pair of orbitals and occupations defines the density matrix._ 
```C++
using OpenOrbitalOptimizer::DensityMatrix =  std::pair<Orbitals<Torb>,OrbitalOccupations<Tbase>>;
```




<hr>



### typedef DiagonalOrbitalHessianBlock 

```C++
using OpenOrbitalOptimizer::DiagonalOrbitalHessianBlock =  arma::Mat<T>;
```



A symmetry block of diagonal orbital Hessians is defined by the corresponding N x N matrix 


        

<hr>



### typedef DiagonalOrbitalHessians 

```C++
using OpenOrbitalOptimizer::DiagonalOrbitalHessians =  std::vector<DiagonalOrbitalHessianBlock<T>>;
```



The set of diagonal orbital Hessians is defined by a vector of orbital blocks, corresponding to each symmetry block of each particle type 


        

<hr>



### typedef DiagonalizedFockMatrix 

_The return of Fock matrix diagonalization is._ 
```C++
using OpenOrbitalOptimizer::DiagonalizedFockMatrix =  std::pair<Orbitals<Torb>,OrbitalEnergies<Tbase>>;
```




<hr>



### typedef FockBuilder 

```C++
using OpenOrbitalOptimizer::FockBuilder =  std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;
```



The Fock builder takes in the orbitals and orbital occupations, and returns the energy and Fock matrices 


        

<hr>



### typedef FockBuilderReturn 

```C++
using OpenOrbitalOptimizer::FockBuilderReturn =  std::pair<Tbase, FockMatrix<Torb>>;
```



The Fock matrix builder returns the energy and the Fock matrices for each orbital block 


        

<hr>



### typedef FockMatrix 

_The whole set of Fock matrices is a vector of blocks._ 
```C++
using OpenOrbitalOptimizer::FockMatrix =  std::vector<FockMatrixBlock<T>>;
```




<hr>



### typedef FockMatrixBlock 

```C++
using OpenOrbitalOptimizer::FockMatrixBlock =  arma::Mat<T>;
```



A symmetry block in a Fock matrix is likewise defined by a N x N matrix 


        

<hr>



### typedef OrbitalBlock 

```C++
using OpenOrbitalOptimizer::OrbitalBlock =  arma::Mat<T>;
```



A symmetry block of orbitals is defined by the corresponding N x N matrix of orbital coefficients 


        

<hr>



### typedef OrbitalBlockOccupations 

_The occupations for each orbitals are floating point numbers._ 
```C++
using OpenOrbitalOptimizer::OrbitalBlockOccupations =  arma::Col<T>;
```




<hr>



### typedef OrbitalEnergies 

_Orbital energies are stored as a vector of vectors._ 
```C++
using OpenOrbitalOptimizer::OrbitalEnergies =  std::vector<arma::Col<T>>;
```




<hr>



### typedef OrbitalGradientBlock 

```C++
using OpenOrbitalOptimizer::OrbitalGradientBlock =  arma::Mat<T>;
```



A symmetry block of orbital gradients is defined by the corresponding N x N matrix 


        

<hr>



### typedef OrbitalGradients 

```C++
using OpenOrbitalOptimizer::OrbitalGradients =  std::vector<OrbitalGradientBlock<T>>;
```



The set of orbital gradients is defined by a vector of orbital blocks, corresponding to each symmetry block of each particle type 


        

<hr>



### typedef OrbitalHistory 

_The history is then a vector._ 
```C++
using OpenOrbitalOptimizer::OrbitalHistory =  std::vector<OrbitalHistoryEntry<Torb, Tbase>>;
```




<hr>



### typedef OrbitalHistoryEntry 

```C++
using OpenOrbitalOptimizer::OrbitalHistoryEntry =  std::tuple<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>, size_t>;
```



The history of orbital optimization is defined by the orbitals and their occupations - together the density matrix - and the resulting energy and Fock matrix 


        

<hr>



### typedef OrbitalOccupations 

```C++
using OpenOrbitalOptimizer::OrbitalOccupations =  std::vector<OrbitalBlockOccupations<T>>;
```



The occupations for the whole set of orbitals are again a vector 


        

<hr>



### typedef OrbitalRotation 

_List of orbital rotation angles: block index and orbital indices._ 
```C++
using OpenOrbitalOptimizer::OrbitalRotation =  std::tuple<size_t, arma::uword, arma::uword>;
```




<hr>



### typedef Orbitals 

```C++
using OpenOrbitalOptimizer::Orbitals =  std::vector<OrbitalBlock<T>>;
```



The set of orbitals is defined by a vector of orbital blocks, corresponding to each symmetry block of each particle type 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/cg_optimizer.hpp`

