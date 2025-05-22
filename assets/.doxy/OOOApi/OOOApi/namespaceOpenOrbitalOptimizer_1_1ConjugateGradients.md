

# Namespace OpenOrbitalOptimizer::ConjugateGradients



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**ConjugateGradients**](namespaceOpenOrbitalOptimizer_1_1ConjugateGradients.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  arma::Col&lt; T &gt; | [**cg\_optimize**](#function-cg_optimize) (const arma::Col&lt; T &gt; & x0, const std::function&lt; std::pair&lt; T, arma::Col&lt; T &gt; &gt;(arma::Col&lt; T &gt;)&gt; & fx, T f\_tol=1e-8, T df\_tol=1e-6, T x\_tol=100 \*std::numeric\_limits&lt; T &gt;::epsilon(), size\_t max\_iter=1000) <br>_Conjugate gradient optimization._  |




























## Public Functions Documentation




### function cg\_optimize 

_Conjugate gradient optimization._ 
```C++
template<typename T>
arma::Col< T > OpenOrbitalOptimizer::ConjugateGradients::cg_optimize (
    const arma::Col< T > & x0,
    const std::function< std::pair< T, arma::Col< T > >(arma::Col< T >)> & fx,
    T f_tol=1e-8,
    T df_tol=1e-6,
    T x_tol=100 *std::numeric_limits< T >::epsilon(),
    size_t max_iter=1000
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/cg_optimizer.hpp`

