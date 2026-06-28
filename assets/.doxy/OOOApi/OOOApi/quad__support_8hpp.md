

# File quad\_support.hpp



[**FileList**](files.md) **>** [**openorbitaloptimizer**](dir_3072c93c56dfbbd2cb4eee0809487533.md) **>** [**quad\_support.hpp**](quad__support_8hpp.md)

[Go to the source code of this file](quad__support_8hpp_source.md)



* `#include <limits>`
* `#include <ostream>`
* `#include <Eigen/Core>`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**Eigen**](namespaceEigen.md) <br> |


## Classes

| Type | Name |
| ---: | :--- |
| struct | [**NumTraits&lt; \_Float128 &gt;**](structEigen_1_1NumTraits_3_01__Float128_01_4.md) &lt;&gt;<br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  std::ostream & | [**operator&lt;&lt;**](#function-operator) (std::ostream & os, \_Float128 x) <br> |




























## Public Functions Documentation




### function operator&lt;&lt; 

```C++
inline std::ostream & operator<< (
    std::ostream & os,
    _Float128 x
) 
```



Opt-in glue making `_Float128` (the ISO C23 quad-precision type, available in GCC 13+/Clang 18+ with libstdc++ &gt;= 14) usable as a Tbase for SCFSolver. libstdc++ already specialises std::numeric\_limits and provides std::abs/sqrt/exp/log/pow/isnan/... for \_Float128, but only when compiled as C++23 or newer (the overloads are gated by \_\_cpp\_lib\_extended\_float). The missing pieces are an Eigen::NumTraits specialisation and an ostream insertion operator that resolves the otherwise-ambiguous lookup chain.


Requirements for consumers of this header:
* compile with -std=c++23 (or newer)
* link against libquadmath (system library for some helpers; Eigen itself only needs the language type) Include this header BEFORE the first Eigen header — the operator&lt;&lt;(ostream, \_Float128) declaration needs to be visible during Eigen's two-phase template lookup, and that means it must precede Eigen's IO.h definition. libstdc++ does not provide a stream output operator for \_Float128. Without one, any ostream insertion on x86-64 is ambiguous because the implicit conversions to float / double / long double all rank equally. Provide an explicit overload routed through long double — precision past ~19 digits is lost in print, but this is a development-time diagnostic path, not a serialisation format.




On targets where long double already is binary128 (aarch64, PowerPC IEEE long double, ...) the existing operator&lt;&lt;(long double) is an exact match for \_Float128 too, and adding our overload would itself create an ambiguity. Skip it there. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/quad_support.hpp`

