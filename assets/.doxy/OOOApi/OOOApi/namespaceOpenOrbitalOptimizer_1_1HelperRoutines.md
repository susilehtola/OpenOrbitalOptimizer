

# Namespace OpenOrbitalOptimizer::HelperRoutines



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**HelperRoutines**](namespaceOpenOrbitalOptimizer_1_1HelperRoutines.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::pair&lt; T, T &gt; | [**cubic\_polynomial\_zeros**](#function-cubic_polynomial_zeros) (T a0, T a1, T a2, T a3) <br> |
|  std::tuple&lt; T, T, T, T &gt; | [**fit\_cubic\_polynomial\_with\_derivatives**](#function-fit_cubic_polynomial_with_derivatives) (T E0, T dE0, T x1, T E1, T dE1) <br> |




























## Public Functions Documentation




### function cubic\_polynomial\_zeros 

```C++
template<typename T>
std::pair< T, T > OpenOrbitalOptimizer::HelperRoutines::cubic_polynomial_zeros (
    T a0,
    T a1,
    T a2,
    T a3
) 
```



Return the (real) zeros of the derivative f'(x) = a1 + 2\*a2\*x + 3\*a3\*x^2 of the cubic polynomial f(x) = a0 + a1\*x + a2\*x^2 + a3\*x^3, i.e. the candidate extrema of f. Throws if no real roots exist. 


        

<hr>



### function fit\_cubic\_polynomial\_with\_derivatives 

```C++
template<typename T>
std::tuple< T, T, T, T > OpenOrbitalOptimizer::HelperRoutines::fit_cubic_polynomial_with_derivatives (
    T E0,
    T dE0,
    T x1,
    T E1,
    T dE1
) 
```



Fit cubic polynomial f(x) = a0 + a1\*x + a2\*x^2 + a3\*x^3 to the data {f(0)=E0, f'(0)=dE0, f(x1)=E1, f'(x1)=dE1}. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

