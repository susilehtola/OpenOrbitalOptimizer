

# Namespace OpenOrbitalOptimizer::ConjugateGradients



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**ConjugateGradients**](namespaceOpenOrbitalOptimizer_1_1ConjugateGradients.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; | [**cg\_optimize**](#function-cg_optimize) (const [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & x0, const std::function&lt; std::pair&lt; T, [**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; &gt;([**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt;)&gt; & fx, T f\_tol=1e-8, T df\_tol=1e-6, T x\_tol=100 \*std::numeric\_limits&lt; T &gt;::epsilon(), size\_t max\_iter=1000) <br>_Conjugate gradient optimization._  |




























## Public Functions Documentation




### function cg\_optimize 

_Conjugate gradient optimization._ 
```C++
template<typename T>
Vector < T > OpenOrbitalOptimizer::ConjugateGradients::cg_optimize (
    const Vector < T > & x0,
    const std::function< std::pair< T, Vector < T > >( Vector < T >)> & fx,
    T f_tol=1e-8,
    T df_tol=1e-6,
    T x_tol=100 *std::numeric_limits< T >::epsilon(),
    size_t max_iter=1000
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/cg_optimizer.hpp`

