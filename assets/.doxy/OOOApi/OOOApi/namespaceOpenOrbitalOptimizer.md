

# Namespace OpenOrbitalOptimizer



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md)



[More...](#detailed-description)














## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**Armadillo**](namespaceOpenOrbitalOptimizer_1_1Armadillo.md) <br> |
| namespace | [**ConjugateGradients**](namespaceOpenOrbitalOptimizer_1_1ConjugateGradients.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| class | [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) &lt;typename Torb, typename Tbase&gt;<br> |


## Public Types

| Type | Name |
| ---: | :--- |
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
|  auto | [**dot\_nonconj**](#function-dot_nonconj) (const V1 & a, const V2 & b) <br> |
|  [**IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) | [**find\_indices\_where**](#function-find_indices_where) (const Vec & v, Pred pred) <br> |
|  bool | [**has\_inf**](#function-has_inf) (const Mat & M) <br>_True iff M contains an infinity._  |
|  bool | [**has\_nan**](#function-has_nan) (const Mat & M) <br>_True iff M contains a NaN. Eigen has allFinite() but not_ [_**has\_nan()**_](namespaceOpenOrbitalOptimizer.md#function-has_nan) _._ |
|  [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) | [**index\_max\_abs**](#function-index_max_abs) (const Vec & v) <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**join\_columns**](#function-join_columns) (const std::vector&lt; [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; & parts) <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**linspace**](#function-linspace) (T a, T b, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) n) <br> |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**logspace**](#function-logspace) (T a, T b, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) n) <br> |
|  void | [**save\_raw\_ascii**](#function-save_raw_ascii) (const Mat & M, const std::string & filename) <br> |
|  [**IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) | [**sort\_index\_ascending**](#function-sort_index_ascending) (const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & v) <br> |
|  std::enable\_if\_t&lt;!Eigen::NumTraits&lt; T &gt;::IsComplex, [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; &gt; | [**unvectorise\_real\_imag**](#function-unvectorise_real_imag) (const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & v, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) rows, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) cols) <br>_Inverse of vectorise\_real\_imag for the real case._  |
|  [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; std::complex&lt; T &gt; &gt; | [**unvectorise\_real\_imag\_complex**](#function-unvectorise_real_imag_complex) (const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & v, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) rows, [**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) cols) <br>_Inverse of vectorise\_real\_imag for the complex case._  |
|  std::enable\_if\_t&lt;!Eigen::NumTraits&lt; T &gt;::IsComplex, [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; | [**vectorise\_real\_imag**](#function-vectorise_real_imag) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & M) <br> |
|  std::enable\_if\_t&lt; Eigen::NumTraits&lt; T &gt;::IsComplex, [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; [**RealOf**](namespaceOpenOrbitalOptimizer.md#typedef-realof)&lt; T &gt; &gt; &gt; | [**vectorise\_real\_imag**](#function-vectorise_real_imag) (const [**Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & M) <br> |




























## Detailed Description


Opt-in compatibility shim that exposes the pre-Eigen public API:


[**OpenOrbitalOptimizer::Armadillo::SCFSolver&lt;Torb, Tbase&gt;**](classOpenOrbitalOptimizer_1_1Armadillo_1_1SCFSolver.md)


with all containers Armadillo-typed (arma::Mat, arma::Col, arma::uvec, etc.). Internally wraps the new Eigen-based [**OpenOrbitalOptimizer::SCFSolver&lt;Tbase, IsComplex&gt;**](classOpenOrbitalOptimizer_1_1SCFSolver.md), with Armadillo&lt;-&gt; Eigen conversion at the [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) boundary. The conversions are memcpy-cost (column-major to column-major) and the Fock-builder callback is bridged transparently.


Including this header pulls in &lt;armadillo&gt;. The core library itself remains Armadillo-free; only consumers who include this header pay the Armadillo dependency. Only the four legacy scalar pairs are supported: (float,float), (double,double), (std::complex&lt;float&gt;,float), (std::complex&lt;double&gt;,double). 


    
## Public Types Documentation




### typedef DensityMatrix 

_Density matrix bundle: orbitals + occupations._ 
```C++
using OpenOrbitalOptimizer::DensityMatrix =  std::pair<Orbitals<Torb>, OrbitalOccupations<Tbase>>;
```




<hr>



### typedef DiagonalOrbitalHessianBlock 

_Diagonal orbital Hessian (one column per orbital)._ 
```C++
using OpenOrbitalOptimizer::DiagonalOrbitalHessianBlock =  Matrix<T>;
```




<hr>



### typedef DiagonalOrbitalHessians 

```C++
using OpenOrbitalOptimizer::DiagonalOrbitalHessians =  std::vector<DiagonalOrbitalHessianBlock<T>>;
```




<hr>



### typedef DiagonalizedFockMatrix 

_Diagonalized Fock matrix: orbitals + energies._ 
```C++
using OpenOrbitalOptimizer::DiagonalizedFockMatrix =  std::pair<Orbitals<Torb>, OrbitalEnergies<Tbase>>;
```




<hr>



### typedef FockBuilder 

_User-supplied Fock builder callback signature._ 
```C++
using OpenOrbitalOptimizer::FockBuilder =  std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;
```




<hr>



### typedef FockBuilderReturn 

_Fock builder return value: (energy, Fock)._ 
```C++
using OpenOrbitalOptimizer::FockBuilderReturn =  std::pair<Tbase, FockMatrix<Torb>>;
```




<hr>



### typedef FockMatrix 

```C++
using OpenOrbitalOptimizer::FockMatrix =  std::vector<FockMatrixBlock<T>>;
```




<hr>



### typedef FockMatrixBlock 

_Fock matrix in one symmetry block._ 
```C++
using OpenOrbitalOptimizer::FockMatrixBlock =  Matrix<T>;
```




<hr>



### typedef Index 

[_**Index**_](namespaceOpenOrbitalOptimizer.md#typedef-index) _type._
```C++
using OpenOrbitalOptimizer::Index =  Eigen::Index;
```




<hr>



### typedef IndexVector 

[_**Index**_](namespaceOpenOrbitalOptimizer.md#typedef-index) _column vector (replacement for arma::uvec)._
```C++
using OpenOrbitalOptimizer::IndexVector =  Eigen::Matrix<Index, Eigen::Dynamic, 1>;
```




<hr>



### typedef Matrix 

_Dense matrix alias._ 
```C++
using OpenOrbitalOptimizer::Matrix =  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
```




<hr>



### typedef OrbitalBlock 

_Orbital coefficients in one symmetry block (rows = basis, cols = orbitals)._ 
```C++
using OpenOrbitalOptimizer::OrbitalBlock =  Matrix<T>;
```




<hr>



### typedef OrbitalBlockOccupations 

_Real-valued occupations in one symmetry block._ 
```C++
using OpenOrbitalOptimizer::OrbitalBlockOccupations =  Vector<T>;
```




<hr>



### typedef OrbitalEnergies 

_Real-valued orbital energies in one symmetry block._ 
```C++
using OpenOrbitalOptimizer::OrbitalEnergies =  std::vector<Vector<T>>;
```




<hr>



### typedef OrbitalGradientBlock 

_Block-diagonal orbital gradient._ 
```C++
using OpenOrbitalOptimizer::OrbitalGradientBlock =  Matrix<T>;
```




<hr>



### typedef OrbitalGradients 

```C++
using OpenOrbitalOptimizer::OrbitalGradients =  std::vector<OrbitalGradientBlock<T>>;
```




<hr>



### typedef OrbitalHistory 

```C++
using OpenOrbitalOptimizer::OrbitalHistory =  std::vector<OrbitalHistoryEntry<Torb, Tbase>>;
```




<hr>



### typedef OrbitalHistoryEntry 

_Single history entry: density, Fock-builder output, generation id._ 
```C++
using OpenOrbitalOptimizer::OrbitalHistoryEntry =  std::tuple<DensityMatrix<Torb, Tbase>, FockBuilderReturn<Torb, Tbase>, size_t>;
```




<hr>



### typedef OrbitalOccupations 

```C++
using OpenOrbitalOptimizer::OrbitalOccupations =  std::vector<OrbitalBlockOccupations<T>>;
```




<hr>



### typedef OrbitalRotation 

_(block index, orbital i, orbital j) describing a single orbital rotation._ 
```C++
using OpenOrbitalOptimizer::OrbitalRotation =  std::tuple<size_t, Index, Index>;
```




<hr>



### typedef OrbitalScalar 

```C++
using OpenOrbitalOptimizer::OrbitalScalar =  std::conditional_t<IsComplex, std::complex<Tbase>, Tbase>;
```



Resolves the orbital scalar type from (Tbase, IsComplex): Tbase for IsComplex=false, std::complex&lt;Tbase&gt; for IsComplex=true. 


        

<hr>



### typedef Orbitals 

_One_ [_**OrbitalBlock**_](namespaceOpenOrbitalOptimizer.md#typedef-orbitalblock) _per symmetry block, per particle type._
```C++
using OpenOrbitalOptimizer::Orbitals =  std::vector<OrbitalBlock<T>>;
```




<hr>



### typedef RealOf 

_Real component type of a (possibly complex) scalar._ 
```C++
using OpenOrbitalOptimizer::RealOf =  typename Eigen::NumTraits<T>::Real;
```




<hr>



### typedef Vector 

_Dense column-vector alias._ 
```C++
using OpenOrbitalOptimizer::Vector =  Eigen::Matrix<T, Eigen::Dynamic, 1>;
```




<hr>
## Public Functions Documentation




### function dot\_nonconj 

```C++
template<class V1, class V2>
auto OpenOrbitalOptimizer::dot_nonconj (
    const V1 & a,
    const V2 & b
) 
```



arma::dot for complex vectors is non-conjugating; Eigen's a.dot(b) is conjugating. Provide a non-conjugating dot for parity. 


        

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



### function index\_max\_abs 

```C++
template<class Vec>
Index OpenOrbitalOptimizer::index_max_abs (
    const Vec & v
) 
```



[**Index**](namespaceOpenOrbitalOptimizer.md#typedef-index) of the largest absolute value in v (matches arma's index\_max for real and arma's index\_max(abs(v)) for complex). 


        

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



### function linspace 

```C++
template<class T>
Vector < T > OpenOrbitalOptimizer::linspace (
    T a,
    T b,
    Index n
) 
```



arma::linspace(a, b, n) replacement returning n equally-spaced points [a, b]. 


        

<hr>



### function logspace 

```C++
template<class T>
Vector < T > OpenOrbitalOptimizer::logspace (
    T a,
    T b,
    Index n
) 
```



Logarithmically-spaced points 10^a ... 10^b (n points). Mirrors arma::logspace. 


        

<hr>



### function save\_raw\_ascii 

```C++
template<class Mat>
void OpenOrbitalOptimizer::save_raw_ascii (
    const Mat & M,
    const std::string & filename
) 
```



Dump a dense matrix as ASCII (one row per line, space-separated). Stand-in for arma::Mat::save(name, arma::raw\_ascii). 


        

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



### function unvectorise\_real\_imag 

_Inverse of vectorise\_real\_imag for the real case._ 
```C++
template<class T>
std::enable_if_t<!Eigen::NumTraits< T >::IsComplex, Matrix < T > > OpenOrbitalOptimizer::unvectorise_real_imag (
    const Vector < T > & v,
    Index rows,
    Index cols
) 
```




<hr>



### function unvectorise\_real\_imag\_complex 

_Inverse of vectorise\_real\_imag for the complex case._ 
```C++
template<class T>
Matrix < std::complex< T > > OpenOrbitalOptimizer::unvectorise_real_imag_complex (
    const Vector < T > & v,
    Index rows,
    Index cols
) 
```




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

