

# Namespace OpenOrbitalOptimizer



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md)



[More...](#detailed-description)














## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**Armadillo**](namespaceOpenOrbitalOptimizer_1_1Armadillo.md) <br> |
| namespace | [**HelperRoutines**](namespaceOpenOrbitalOptimizer_1_1HelperRoutines.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| class | [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) &lt;typename Torb, typename Tbase&gt;<br>_SCF solver class._  |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::function&lt; std::vector&lt; [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt; &gt;(const std::vector&lt; [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; &gt; &)&gt; | [**BatchedFockBuilder**](#typedef-batchedfockbuilder)  <br> |
| typedef std::pair&lt; [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt;, OrbitalOccupations&lt; Tbase &gt; &gt; | [**DensityMatrix**](#typedef-densitymatrix)  <br>_Density matrix bundle: orbitals + occupations._  |
| typedef [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**DiagonalOrbitalHessianBlock**](#typedef-diagonalorbitalhessianblock)  <br>_Diagonal orbital Hessian (one column per orbital)._  |
| typedef std::vector&lt; [**DiagonalOrbitalHessianBlock**](namespaceOpenOrbitalOptimizer.md#typedef-diagonalorbitalhessianblock)&lt; T &gt; &gt; | [**DiagonalOrbitalHessians**](#typedef-diagonalorbitalhessians)  <br> |
| typedef std::pair&lt; [**Orbitals**](namespaceOpenOrbitalOptimizer.md#typedef-orbitals)&lt; Torb &gt;, [**OrbitalEnergies**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalenergies)&lt; Tbase &gt; &gt; | [**DiagonalizedFockMatrix**](#typedef-diagonalizedfockmatrix)  <br>_Diagonalized Fock matrix: orbitals + energies._  |
| typedef std::function&lt; [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt;(const [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt; &)&gt; | [**FockBuilder**](#typedef-fockbuilder)  <br>_User-supplied Fock builder callback signature._  |
| typedef std::pair&lt; Tbase, FockMatrix&lt; Torb &gt; &gt; | [**FockBuilderReturn**](#typedef-fockbuilderreturn)  <br>_Fock builder return value: (energy, Fock)._  |
| typedef std::vector&lt; [**FockMatrixBlock**](namespaceOpenOrbitalOptimizer.md#typedef-fockmatrixblock)&lt; T &gt; &gt; | [**FockMatrix**](#typedef-fockmatrix)  <br> |
| typedef [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**FockMatrixBlock**](#typedef-fockmatrixblock)  <br>_Fock matrix in one symmetry block._  |
| typedef Eigen::Index | [**Index**](#typedef-index)  <br>[_**Index**_](namespaceOpenOrbitalOptimizer.md#typedef-index) _type._ |
| typedef Eigen::Matrix&lt; [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index), Eigen::Dynamic, 1 &gt; | [**IndexVector**](#typedef-indexvector)  <br>[_**Index**_](namespaceOpenOrbitalOptimizer.md#typedef-index) _column vector (replacement for arma::uvec)._ |
| typedef Eigen::Matrix&lt; T, Eigen::Dynamic, Eigen::Dynamic &gt; | [**Matrix**](#typedef-matrix)  <br>_Dense matrix alias._  |
| typedef [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**OrbitalBlock**](#typedef-orbitalblock)  <br>_Orbital coefficients in one symmetry block (rows = basis, cols = orbitals)._  |
| typedef [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**OrbitalBlockOccupations**](#typedef-orbitalblockoccupations)  <br>_Real-valued occupations in one symmetry block._  |
| typedef std::vector&lt; [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; | [**OrbitalEnergies**](#typedef-orbitalenergies)  <br>_Real-valued orbital energies in one symmetry block._  |
| typedef [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**OrbitalGradientBlock**](#typedef-orbitalgradientblock)  <br>_Block-diagonal orbital gradient._  |
| typedef std::vector&lt; [**OrbitalGradientBlock**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalgradientblock)&lt; T &gt; &gt; | [**OrbitalGradients**](#typedef-orbitalgradients)  <br> |
| typedef std::vector&lt; [**OrbitalHistoryEntry**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalhistoryentry)&lt; Torb, Tbase &gt; &gt; | [**OrbitalHistory**](#typedef-orbitalhistory)  <br> |
| typedef std::tuple&lt; [**DensityMatrix**](namespaceOpenOrbitalOptimizer.md#typedef-densitymatrix)&lt; Torb, Tbase &gt;, [**FockBuilderReturn**](namespaceOpenOrbitalOptimizer.md#typedef-fockbuilderreturn)&lt; Torb, Tbase &gt;, size\_t &gt; | [**OrbitalHistoryEntry**](#typedef-orbitalhistoryentry)  <br>_Single history entry: density, Fock-builder output, generation id._  |
| typedef std::vector&lt; [**OrbitalBlockOccupations**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalblockoccupations)&lt; T &gt; &gt; | [**OrbitalOccupations**](#typedef-orbitaloccupations)  <br> |
| typedef std::tuple&lt; size\_t, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index), [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) &gt; | [**OrbitalRotation**](#typedef-orbitalrotation)  <br>_(block index, orbital i, orbital j) describing a single orbital rotation._  |
| typedef std::conditional\_t&lt; IsComplex, std::complex&lt; Tbase &gt;, Tbase &gt; | [**OrbitalScalar**](#typedef-orbitalscalar)  <br> |
| typedef std::vector&lt; [**OrbitalBlock**](namespaceOpenOrbitalOptimizer.md#typedef-orbitalblock)&lt; T &gt; &gt; | [**Orbitals**](#typedef-orbitals)  <br>_One_ [_**OrbitalBlock**_](namespaceOpenOrbitalOptimizer.md#typedef-orbitalblock) _per symmetry block, per particle type._ |
| typedef typename Eigen::NumTraits&lt; T &gt;::Real | [**RealOf**](#typedef-realof)  <br>_Real component type of a (possibly complex) scalar._  |
| typedef Eigen::Matrix&lt; T, Eigen::Dynamic, 1 &gt; | [**Vector**](#typedef-vector)  <br>_Dense column-vector alias._  |




















## Public Functions

| Type | Name |
| ---: | :--- |
|  [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**expm\_antihermitian**](#function-expm_antihermitian) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & K) <br> |
|  [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**expm\_antihermitian\_by\_eigendecomposition**](#function-expm_antihermitian_by_eigendecomposition) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & K) <br> |
|  [**IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) | [**find\_indices\_where**](#function-find_indices_where) (const Vec & v, Pred pred) <br> |
|  bool | [**has\_inf**](#function-has_inf) (const Mat & M) <br>_True iff M contains an infinity._  |
|  bool | [**has\_nan**](#function-has_nan) (const Mat & M) <br>_True iff M contains a NaN. Eigen has allFinite() but not_ [_**has\_nan()**_](namespaceOpenOrbitalOptimizer.md#function-has_nan) _._ |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**join\_columns**](#function-join_columns) (const std::vector&lt; [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; & parts) <br> |
|  [**IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) | [**sort\_index\_ascending**](#function-sort_index_ascending) (const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & v) <br> |
|  std::enable\_if\_t&lt;!Eigen::NumTraits&lt; T &gt;::IsComplex, [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; | [**vectorise\_real\_imag**](#function-vectorise_real_imag) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & M) <br> |
|  std::enable\_if\_t&lt; Eigen::NumTraits&lt; T &gt;::IsComplex, [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; [**RealOf**](namespaceOpenOrbitalOptimizer.md#typedef-realof)&lt; T &gt; &gt; &gt; | [**vectorise\_real\_imag**](#function-vectorise_real_imag) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & M) <br> |




























## Detailed Description


Opt-in compatibility shim that exposes the pre-Eigen public API:


[**OpenOrbitalOptimizer::Armadillo::SCFSolver&lt;Torb, Tbase&gt;**](classOpenOrbitalOptimizer_1_1Armadillo_1_1SCFSolver.md)


with all containers Armadillo-typed (arma::Mat, arma::Col, arma::uvec, etc.). Internally wraps the new Eigen-based [**OpenOrbitalOptimizer::SCFSolver&lt;Tbase, IsComplex&gt;**](classOpenOrbitalOptimizer_1_1SCFSolver.md), with Armadillo&lt;-&gt; Eigen conversion at the [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) boundary. The conversions are memcpy-cost (column-major to column-major) and the Fock-builder callback is bridged transparently.


Including this header pulls in &lt;armadillo&gt;. The core library itself remains Armadillo-free; only consumers who include this header pay the Armadillo dependency. Only the four legacy scalar pairs are supported: (float,float), (double,double), (std::complex&lt;float&gt;,float), (std::complex&lt;double&gt;,double). 


    
## Public Types Documentation




### typedef BatchedFockBuilder 

```C++
using OpenOrbitalOptimizer::BatchedFockBuilder = std::function<std::vector<FockBuilderReturn<Torb, Tbase>>(const std::vector<DensityMatrix<Torb, Tbase>> &)>;
```



Optional batched Fock builder: given a list of densities, return the corresponding list of (energy, Fock) pairs. Enables per-vertex sweeps in the ODA polytope face-minimization step. 


        

<hr>



### typedef DensityMatrix 

_Density matrix bundle: orbitals + occupations._ 
```C++
using OpenOrbitalOptimizer::DensityMatrix = std::pair<Orbitals<Torb>, OrbitalOccupations<Tbase>>;
```




<hr>



### typedef DiagonalOrbitalHessianBlock 

_Diagonal orbital Hessian (one column per orbital)._ 
```C++
using OpenOrbitalOptimizer::DiagonalOrbitalHessianBlock = Matrix<T>;
```




<hr>



### typedef DiagonalOrbitalHessians 

```C++
using OpenOrbitalOptimizer::DiagonalOrbitalHessians = std::vector<DiagonalOrbitalHessianBlock<T>>;
```




<hr>



### typedef DiagonalizedFockMatrix 

_Diagonalized Fock matrix: orbitals + energies._ 
```C++
using OpenOrbitalOptimizer::DiagonalizedFockMatrix = std::pair<Orbitals<Torb>, OrbitalEnergies<Tbase>>;
```




<hr>



### typedef FockBuilder 

_User-supplied Fock builder callback signature._ 
```C++
using OpenOrbitalOptimizer::FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;
```




<hr>



### typedef FockBuilderReturn 

_Fock builder return value: (energy, Fock)._ 
```C++
using OpenOrbitalOptimizer::FockBuilderReturn = std::pair<Tbase, FockMatrix<Torb>>;
```




<hr>



### typedef FockMatrix 

```C++
using OpenOrbitalOptimizer::FockMatrix = std::vector<FockMatrixBlock<T>>;
```




<hr>



### typedef FockMatrixBlock 

_Fock matrix in one symmetry block._ 
```C++
using OpenOrbitalOptimizer::FockMatrixBlock = Matrix<T>;
```




<hr>



### typedef Index 

[_**Index**_](namespaceOpenOrbitalOptimizer.md#typedef-index) _type._
```C++
using OpenOrbitalOptimizer::Index = Eigen::Index;
```




<hr>



### typedef IndexVector 

[_**Index**_](namespaceOpenOrbitalOptimizer.md#typedef-index) _column vector (replacement for arma::uvec)._
```C++
using OpenOrbitalOptimizer::IndexVector = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
```




<hr>



### typedef Matrix 

_Dense matrix alias._ 
```C++
using OpenOrbitalOptimizer::Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
```




<hr>



### typedef OrbitalBlock 

_Orbital coefficients in one symmetry block (rows = basis, cols = orbitals)._ 
```C++
using OpenOrbitalOptimizer::OrbitalBlock = Matrix<T>;
```




<hr>



### typedef OrbitalBlockOccupations 

_Real-valued occupations in one symmetry block._ 
```C++
using OpenOrbitalOptimizer::OrbitalBlockOccupations = Vector<T>;
```




<hr>



### typedef OrbitalEnergies 

_Real-valued orbital energies in one symmetry block._ 
```C++
using OpenOrbitalOptimizer::OrbitalEnergies = std::vector<Vector<T>>;
```




<hr>



### typedef OrbitalGradientBlock 

_Block-diagonal orbital gradient._ 
```C++
using OpenOrbitalOptimizer::OrbitalGradientBlock = Matrix<T>;
```




<hr>



### typedef OrbitalGradients 

```C++
using OpenOrbitalOptimizer::OrbitalGradients = std::vector<OrbitalGradientBlock<T>>;
```




<hr>



### typedef OrbitalHistory 

```C++
using OpenOrbitalOptimizer::OrbitalHistory = std::vector<OrbitalHistoryEntry<Torb, Tbase>>;
```




<hr>



### typedef OrbitalHistoryEntry 

_Single history entry: density, Fock-builder output, generation id._ 
```C++
using OpenOrbitalOptimizer::OrbitalHistoryEntry = std::tuple<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>, size_t>;
```




<hr>



### typedef OrbitalOccupations 

```C++
using OpenOrbitalOptimizer::OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;
```




<hr>



### typedef OrbitalRotation 

_(block index, orbital i, orbital j) describing a single orbital rotation._ 
```C++
using OpenOrbitalOptimizer::OrbitalRotation = std::tuple<size_t, Index, Index>;
```




<hr>



### typedef OrbitalScalar 

```C++
using OpenOrbitalOptimizer::OrbitalScalar = std::conditional_t<IsComplex, std::complex<Tbase>, Tbase>;
```



Resolves the orbital scalar type from (Tbase, IsComplex): Tbase for IsComplex=false, std::complex&lt;Tbase&gt; for IsComplex=true. 


        

<hr>



### typedef Orbitals 

_One_ [_**OrbitalBlock**_](namespaceOpenOrbitalOptimizer.md#typedef-orbitalblock) _per symmetry block, per particle type._
```C++
using OpenOrbitalOptimizer::Orbitals = std::vector<OrbitalBlock<T>>;
```




<hr>



### typedef RealOf 

_Real component type of a (possibly complex) scalar._ 
```C++
using OpenOrbitalOptimizer::RealOf = typename Eigen::NumTraits<T>::Real;
```




<hr>



### typedef Vector 

_Dense column-vector alias._ 
```C++
using OpenOrbitalOptimizer::Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
```




<hr>
## Public Functions Documentation




### function expm\_antihermitian 

```C++
template<class T>
Matrix < T > OpenOrbitalOptimizer::expm_antihermitian (
    const Matrix < T > & K
) 
```



exp(K) for an anti-Hermitian K = -K^\dagger. Returns a matrix of the same scalar type as K, orthogonal for a real K and unitary for a complex one to round-off, which is what an orbital rotation has to be.


Eigen's matrix exponential  scaling and squaring with a diagonal Pade approximant  is several times faster than diagonalising, and a real anti-symmetric K stays in real arithmetic throughout it: the approximant is a handful of matrix multiplications and one LU solve, so nothing has to be widened to complex to make eigenvalues real.


It carries that Pade path only for the scalar types it names, though. Everything else it routes through a complex Schur decomposition, which for \_Float128 is both slower than diagonalising iK directly (measured 49 s against 8.5 s at n = 150) and reintroduces the very complex promotion the Pade path avoids. So the choice is made on Eigen's own dispatch condition rather than on a copy of it: a copy would keep taking the Schur route in silence if Eigen ever dropped a type from the list, where naming the trait fails to compile if it goes away, and picks up any type Eigen adds. 


        

<hr>



### function expm\_antihermitian\_by\_eigendecomposition 

```C++
template<class T>
Matrix < T > OpenOrbitalOptimizer::expm_antihermitian_by_eigendecomposition (
    const Matrix < T > & K
) 
```



exp(K) for an anti-Hermitian K = -K^\dagger, via the Hermitian eigendecomposition of iK: iK = U diag(w) U^\dagger with real w, so exp(K) = exp(-i \* iK) = U diag(exp(-i w)) U^\dagger.


This is the slower of the two routes for the scalar types Eigen exponentiates natively, and is reserved for those it does not  see expm\_antihermitian. 


        

<hr>



### function find\_indices\_where 

```C++
template<class Vec, class Pred>
IndexVector OpenOrbitalOptimizer::find_indices_where (
    const Vec & v,
    Pred pred
) 
```



Find every index i where pred(v[i]) is true. Stand-in for arma::find(some\_predicate). 


        

<hr>



### function has\_inf 

_True iff M contains an infinity._ 
```C++
template<class Mat>
bool OpenOrbitalOptimizer::has_inf (
    const Mat & M
) 
```




<hr>



### function has\_nan 

_True iff M contains a NaN. Eigen has allFinite() but not_ [_**has\_nan()**_](namespaceOpenOrbitalOptimizer.md#function-has_nan) _._
```C++
template<class Mat>
bool OpenOrbitalOptimizer::has_nan (
    const Mat & M
) 
```




<hr>



### function join\_columns 

```C++
template<class T>
Vector < T > OpenOrbitalOptimizer::join_columns (
    const std::vector< Vector < T > > & parts
) 
```



Stack a vector of column vectors into one long column vector. Replaces arma::join\_cols on Cols. 


        

<hr>



### function sort\_index\_ascending 

```C++
template<class T>
IndexVector OpenOrbitalOptimizer::sort_index_ascending (
    const Vector < T > & v
) 
```



Return the indices that sort v in ascending order (stable). Stand-in for arma::sort\_index. 


        

<hr>



### function vectorise\_real\_imag 

```C++
template<class T>
std::enable_if_t<!Eigen::NumTraits< T >::IsComplex, Vector < T > > OpenOrbitalOptimizer::vectorise_real_imag (
    const Matrix < T > & M
) 
```



Vectorise a real-valued matrix to a column vector (column-major), with no real/imag splitting. 


        

<hr>



### function vectorise\_real\_imag 

```C++
template<class T>
std::enable_if_t< Eigen::NumTraits< T >::IsComplex, Vector < RealOf < T > > > OpenOrbitalOptimizer::vectorise_real_imag (
    const Matrix < T > & M
) 
```



Vectorise a complex-valued matrix into a real column vector by stacking the real part on top of the imaginary part. Mirrors the layout the SCF solver relies on for real-valued optimisation over complex orbital rotations. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/armadillo_compat.hpp`

