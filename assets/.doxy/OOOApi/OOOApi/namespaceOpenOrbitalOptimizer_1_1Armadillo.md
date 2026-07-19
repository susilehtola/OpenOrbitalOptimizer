

# Namespace OpenOrbitalOptimizer::Armadillo



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**Armadillo**](namespaceOpenOrbitalOptimizer_1_1Armadillo.md)




















## Classes

| Type | Name |
| ---: | :--- |
| class | [**SCFSolver**](classOpenOrbitalOptimizer_1_1Armadillo_1_1SCFSolver.md) &lt;class Torb, class Tbase&gt;<br> |


## Public Types

| Type | Name |
| ---: | :--- |
| typedef std::pair&lt; Orbitals&lt; Torb &gt;, OrbitalOccupations&lt; Tbase &gt; &gt; | [**DensityMatrix**](#typedef-densitymatrix)  <br> |
| typedef arma::Mat&lt; T &gt; | [**DiagonalOrbitalHessianBlock**](#typedef-diagonalorbitalhessianblock)  <br> |
| typedef std::vector&lt; DiagonalOrbitalHessianBlock&lt; T &gt; &gt; | [**DiagonalOrbitalHessians**](#typedef-diagonalorbitalhessians)  <br> |
| typedef std::pair&lt; Orbitals&lt; Torb &gt;, OrbitalEnergies&lt; Tbase &gt; &gt; | [**DiagonalizedFockMatrix**](#typedef-diagonalizedfockmatrix)  <br> |
| typedef std::function&lt; FockBuilderReturn&lt; Torb, Tbase &gt;(const DensityMatrix&lt; Torb, Tbase &gt; &)&gt; | [**FockBuilder**](#typedef-fockbuilder)  <br> |
| typedef std::pair&lt; Tbase, FockMatrix&lt; Torb &gt; &gt; | [**FockBuilderReturn**](#typedef-fockbuilderreturn)  <br> |
| typedef std::vector&lt; FockMatrixBlock&lt; T &gt; &gt; | [**FockMatrix**](#typedef-fockmatrix)  <br> |
| typedef arma::Mat&lt; T &gt; | [**FockMatrixBlock**](#typedef-fockmatrixblock)  <br> |
| typedef arma::Mat&lt; T &gt; | [**OrbitalBlock**](#typedef-orbitalblock)  <br> |
| typedef arma::Col&lt; T &gt; | [**OrbitalBlockOccupations**](#typedef-orbitalblockoccupations)  <br> |
| typedef std::vector&lt; arma::Col&lt; T &gt; &gt; | [**OrbitalEnergies**](#typedef-orbitalenergies)  <br> |
| typedef arma::Mat&lt; T &gt; | [**OrbitalGradientBlock**](#typedef-orbitalgradientblock)  <br> |
| typedef std::vector&lt; OrbitalGradientBlock&lt; T &gt; &gt; | [**OrbitalGradients**](#typedef-orbitalgradients)  <br> |
| typedef std::vector&lt; OrbitalBlockOccupations&lt; T &gt; &gt; | [**OrbitalOccupations**](#typedef-orbitaloccupations)  <br> |
| typedef std::tuple&lt; size\_t, arma::uword, arma::uword &gt; | [**OrbitalRotation**](#typedef-orbitalrotation)  <br> |
| typedef std::vector&lt; OrbitalBlock&lt; T &gt; &gt; | [**Orbitals**](#typedef-orbitals)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|  arma::Mat&lt; T &gt; | [**to\_arma**](#function-to_arma) (const [**OpenOrbitalOptimizer::Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; & E) <br> |
|  arma::Col&lt; T &gt; | [**to\_arma**](#function-to_arma) (const [**OpenOrbitalOptimizer::Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & v) <br> |
|  std::vector&lt; arma::Mat&lt; T &gt; &gt; | [**to\_arma**](#function-to_arma) (const std::vector&lt; [**OpenOrbitalOptimizer::Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; &gt; & v) <br> |
|  std::vector&lt; arma::Col&lt; T &gt; &gt; | [**to\_arma**](#function-to_arma) (const std::vector&lt; [**OpenOrbitalOptimizer::Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; & v) <br> |
|  [**OpenOrbitalOptimizer::Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; | [**to\_eigen**](#function-to_eigen) (const arma::Mat&lt; T &gt; & A) <br> |
|  [**OpenOrbitalOptimizer::Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**to\_eigen**](#function-to_eigen) (const arma::Col&lt; T &gt; & v) <br> |
|  [**OpenOrbitalOptimizer::IndexVector**](namespaceOpenOrbitalOptimizer.md#typedef-indexvector) | [**to\_eigen**](#function-to_eigen) (const arma::uvec & v) <br> |
|  std::vector&lt; [**OpenOrbitalOptimizer::Matrix**](namespaceOpenOrbitalOptimizer.md#typedef-matrix)&lt; T &gt; &gt; | [**to\_eigen**](#function-to_eigen) (const std::vector&lt; arma::Mat&lt; T &gt; &gt; & v) <br> |
|  std::vector&lt; [**OpenOrbitalOptimizer::Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt; | [**to\_eigen**](#function-to_eigen) (const std::vector&lt; arma::Col&lt; T &gt; &gt; & v) <br> |




























## Public Types Documentation




### typedef DensityMatrix 

```C++
using OpenOrbitalOptimizer::Armadillo::DensityMatrix = std::pair<Orbitals<Torb>, OrbitalOccupations<Tbase>>;
```




<hr>



### typedef DiagonalOrbitalHessianBlock 

```C++
using OpenOrbitalOptimizer::Armadillo::DiagonalOrbitalHessianBlock = arma::Mat<T>;
```




<hr>



### typedef DiagonalOrbitalHessians 

```C++
using OpenOrbitalOptimizer::Armadillo::DiagonalOrbitalHessians = std::vector<DiagonalOrbitalHessianBlock<T>>;
```




<hr>



### typedef DiagonalizedFockMatrix 

```C++
using OpenOrbitalOptimizer::Armadillo::DiagonalizedFockMatrix = std::pair<Orbitals<Torb>, OrbitalEnergies<Tbase>>;
```




<hr>



### typedef FockBuilder 

```C++
using OpenOrbitalOptimizer::Armadillo::FockBuilder = std::function<FockBuilderReturn<Torb, Tbase>(const DensityMatrix<Torb, Tbase> &)>;
```




<hr>



### typedef FockBuilderReturn 

```C++
using OpenOrbitalOptimizer::Armadillo::FockBuilderReturn = std::pair<Tbase, FockMatrix<Torb>>;
```




<hr>



### typedef FockMatrix 

```C++
using OpenOrbitalOptimizer::Armadillo::FockMatrix = std::vector<FockMatrixBlock<T>>;
```




<hr>



### typedef FockMatrixBlock 

```C++
using OpenOrbitalOptimizer::Armadillo::FockMatrixBlock = arma::Mat<T>;
```




<hr>



### typedef OrbitalBlock 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalBlock = arma::Mat<T>;
```




<hr>



### typedef OrbitalBlockOccupations 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalBlockOccupations = arma::Col<T>;
```




<hr>



### typedef OrbitalEnergies 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalEnergies = std::vector<arma::Col<T>>;
```




<hr>



### typedef OrbitalGradientBlock 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalGradientBlock = arma::Mat<T>;
```




<hr>



### typedef OrbitalGradients 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalGradients = std::vector<OrbitalGradientBlock<T>>;
```




<hr>



### typedef OrbitalOccupations 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalOccupations = std::vector<OrbitalBlockOccupations<T>>;
```




<hr>



### typedef OrbitalRotation 

```C++
using OpenOrbitalOptimizer::Armadillo::OrbitalRotation = std::tuple<size_t, arma::uword, arma::uword>;
```




<hr>



### typedef Orbitals 

```C++
using OpenOrbitalOptimizer::Armadillo::Orbitals = std::vector<OrbitalBlock<T>>;
```




<hr>
## Public Functions Documentation




### function to\_arma 

```C++
template<class T>
inline arma::Mat< T > OpenOrbitalOptimizer::Armadillo::to_arma (
    const OpenOrbitalOptimizer::Matrix < T > & E
) 
```




<hr>



### function to\_arma 

```C++
template<class T>
inline arma::Col< T > OpenOrbitalOptimizer::Armadillo::to_arma (
    const OpenOrbitalOptimizer::Vector < T > & v
) 
```




<hr>



### function to\_arma 

```C++
template<class T>
inline std::vector< arma::Mat< T > > OpenOrbitalOptimizer::Armadillo::to_arma (
    const std::vector< OpenOrbitalOptimizer::Matrix < T > > & v
) 
```




<hr>



### function to\_arma 

```C++
template<class T>
inline std::vector< arma::Col< T > > OpenOrbitalOptimizer::Armadillo::to_arma (
    const std::vector< OpenOrbitalOptimizer::Vector < T > > & v
) 
```




<hr>



### function to\_eigen 

```C++
template<class T>
inline OpenOrbitalOptimizer::Matrix < T > OpenOrbitalOptimizer::Armadillo::to_eigen (
    const arma::Mat< T > & A
) 
```




<hr>



### function to\_eigen 

```C++
template<class T>
inline OpenOrbitalOptimizer::Vector < T > OpenOrbitalOptimizer::Armadillo::to_eigen (
    const arma::Col< T > & v
) 
```




<hr>



### function to\_eigen 

```C++
inline OpenOrbitalOptimizer::IndexVector OpenOrbitalOptimizer::Armadillo::to_eigen (
    const arma::uvec & v
) 
```




<hr>



### function to\_eigen 

```C++
template<class T>
inline std::vector< OpenOrbitalOptimizer::Matrix < T > > OpenOrbitalOptimizer::Armadillo::to_eigen (
    const std::vector< arma::Mat< T > > & v
) 
```




<hr>



### function to\_eigen 

```C++
template<class T>
inline std::vector< OpenOrbitalOptimizer::Vector < T > > OpenOrbitalOptimizer::Armadillo::to_eigen (
    const std::vector< arma::Col< T > > & v
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/armadillo_compat.hpp`

