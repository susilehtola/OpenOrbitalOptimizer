

# Namespace OpenOrbitalOptimizer::HelperRoutines



[**Namespace List**](namespaces.md) **>** [**OpenOrbitalOptimizer**](namespaceOpenOrbitalOptimizer.md) **>** [**HelperRoutines**](namespaceOpenOrbitalOptimizer_1_1HelperRoutines.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::pair&lt; T, T &gt; | [**cubic\_polynomial\_zeros**](#function-cubic_polynomial_zeros) (T a0, T a1, T a2, T a3) <br> |
|  T | [**evaluate\_polynomial**](#function-evaluate_polynomial) (const std::array&lt; T, N &gt; & coeffs, T x) <br> |
|  std::tuple&lt; T, T, T, T &gt; | [**fit\_cubic\_polynomial\_with\_derivatives**](#function-fit_cubic_polynomial_with_derivatives) (T E0, T dE0, T x1, T E1, T dE1) <br> |
|  void | [**project\_onto\_unit\_simplex**](#function-project_onto_unit_simplex) ([**Vector**](namespaceOpenOrbitalOptimizer.md#typedef-vector)&lt; T &gt; & v) <br> |




























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



### function project\_onto\_unit\_simplex 

```C++
template<typename T>
void OpenOrbitalOptimizer::HelperRoutines::project_onto_unit_simplex (
    Vector < T > & v
) 
```



Project `v` onto the simplex {v\_i &gt;= 0, sum(v) &lt;= 1}: negative entries are clamped to zero, and if the sum then still exceeds one the whole vector is rescaled by 1/sum.


Rescaling rather than merely capping the sum matters wherever the individual entries are used as weights in their own right: capping the sum alone would leave the entries summing to more than one while the complementary weight (1 - sum) went to zero, which shifts the total.


Used on the ODA polytope parameters, where the QP solver enforces the simplex only to the accuracy of its constrained linear solve. An ill-conditioned reduced Hessian  what a density projected between two different basis sets produces  can leave the sum above one by of order cond \* eps, which hands the reference density a negative weight and yields a non-positive-semidefinite mixed density. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

