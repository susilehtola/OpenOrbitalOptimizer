

# Namespace OpenOrbitalOptimizer::HelperRoutines



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**HelperRoutines**](namespaceOpenOrbitalOptimizer_1_1HelperRoutines.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::pair&lt; T, T &gt; | [**cubic\_polynomial\_zeros**](#function-cubic_polynomial_zeros) (T a0, T a1, T a2, T a3) <br> |
|  T | [**evaluate\_polynomial**](#function-evaluate_polynomial) (const std::array&lt; T, N &gt; & coeffs, T x) <br> |
|  std::tuple&lt; T, T, T, T &gt; | [**fit\_cubic\_polynomial\_with\_derivatives**](#function-fit_cubic_polynomial_with_derivatives) (T E0, T dE0, T x1, T E1, T dE1) <br> |
|  std::tuple&lt; T, T, T, T, T &gt; | [**fit\_quartic\_polynomial\_with\_derivatives**](#function-fit_quartic_polynomial_with_derivatives) (T E0, T dE0, T d2E0, T x1, T E1, T dE1) <br> |
|  std::vector&lt; T &gt; | [**real\_roots\_in\_interval**](#function-real_roots_in_interval) (Fn && f, T x\_lo, T x\_hi, int n\_samples=33) <br> |




























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



### function evaluate\_polynomial 

```C++
template<typename T, size_t N>
T OpenOrbitalOptimizer::HelperRoutines::evaluate_polynomial (
    const std::array< T, N > & coeffs,
    T x
) 
```



Evaluate a polynomial with the given coefficients (index i = coefficient of x^i) at `x` via Horner's scheme. 


        

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



### function fit\_quartic\_polynomial\_with\_derivatives 

```C++
template<typename T>
std::tuple< T, T, T, T, T > OpenOrbitalOptimizer::HelperRoutines::fit_quartic_polynomial_with_derivatives (
    T E0,
    T dE0,
    T d2E0,
    T x1,
    T E1,
    T dE1
) 
```



Fit the quartic polynomial f(x) = a0 + a1\*x + a2\*x^2 + a3\*x^3 + a4\*x^4 to the Hermite data {f(0)=E0, f'(0)=dE0, f''(0)=d2E0, f(x1)=E1, f'(x1)=dE1}  five constraints for five coefficients. Used along ODA polytope axes and pair-diagonal edges where the diagonal (or projected) Hessian element gives a free quartic data point without an additional Fock build. 


        

<hr>



### function real\_roots\_in\_interval 

```C++
template<typename T, class Fn>
std::vector< T > OpenOrbitalOptimizer::HelperRoutines::real_roots_in_interval (
    Fn && f,
    T x_lo,
    T x_hi,
    int n_samples=33
) 
```



Bisect for real roots of `f` in `[x_lo, x_hi]` by sampling on a coarse grid and refining sign changes. Robust and free of external polynomial-root dependencies. Used to locate the stationary points of the quartic axis / edge fits, where the derivative is a cubic. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

