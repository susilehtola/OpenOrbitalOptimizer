

# Class OpenOrbitalOptimizer::SCFSolver::Setting

**template &lt;typename T&gt;**



[**ClassList**](annotated.md) **>** [**Setting**](classOpenOrbitalOptimizer_1_1SCFSolver_1_1Setting.md)



[More...](#detailed-description)




Inherits the following classes: OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::SettingBase














## Public Types

| Type | Name |
| ---: | :--- |
| typedef T(SCFSolver::\*)(const T &) const | [**Hook**](#typedef-hook)  <br> |
| typedef T(SCFSolver::\*)() const | [**Source**](#typedef-source)  <br> |




















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Setting**](#function-setting) (SettingRegistry & registry, const char \* key, const char \* doc, T value, bool writable=true, Hook hook=nullptr, Source source=nullptr) <br> |
|  const T & | [**get**](#function-get) () const<br> |
|  Hook | [**hook**](#function-hook) () const<br> |
|   | [**operator const T &**](#function-operator-const-t-&) () const<br> |
|  Setting & | [**operator+=**](#function-operator) (const T & v) <br> |
|  Setting & | [**operator=**](#function-operator_1) (T v) <br> |
|  void | [**print\_value**](#function-print_value) (std::ostream & os) override const<br> |
|  Source | [**source**](#function-source) () const<br> |
|  const char \* | [**type**](#function-type) () override const<br> |




























## Detailed Description


A setting holding a value of type T.


The implicit conversion means the solver body can keep using a setting exactly like the plain member it replaced  `if(verbosity_ >= 5)`, `x * diis_epsilon_`  and operator= keeps internal writes (which are not subject to the writable flag) equally plain. 


    
## Public Types Documentation




### typedef Hook 

```C++
using OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::Setting< T >::Hook = T (SCFSolver::*)(const T &) const;
```



Optional validator / canonicaliser, applied before a write that comes in through the string facade. It is a member function of the solver, so it can consult solver state without the setting holding a back-pointer  which would dangle the first time the solver was moved. Null means "store the value as given". 


        

<hr>



### typedef Source 

```C++
using OpenOrbitalOptimizer::SCFSolver< Torb, Tbase >::Setting< T >::Source = T (SCFSolver::*)() const;
```



Optional source. When set, the setting has no meaningful stored value: it is recomputed on every read. Used by the diagnostics that must reflect the solver's state right now rather than whatever it was when they were last written. 


        

<hr>
## Public Functions Documentation




### function Setting 

```C++
inline Setting::Setting (
    SettingRegistry & registry,
    const char * key,
    const char * doc,
    T value,
    bool writable=true,
    Hook hook=nullptr,
    Source source=nullptr
) 
```



Registers itself, so that a setting is published to the string facade by the act of declaring it. 


        

<hr>



### function get 

```C++
inline const T & Setting::get () const
```




<hr>



### function hook 

```C++
inline Hook Setting::hook () const
```




<hr>



### function operator const T & 

```C++
inline Setting::operator const T & () const
```




<hr>



### function operator+= 

```C++
inline Setting & Setting::operator+= (
    const T & v
) 
```




<hr>



### function operator= 

```C++
inline Setting & Setting::operator= (
    T v
) 
```



Internal writes. Not gated on writable(): that flag governs only what the string facade will accept from a caller. 


        

<hr>



### function print\_value 

```C++
inline void Setting::print_value (
    std::ostream & os
) override const
```




<hr>



### function source 

```C++
inline Source Setting::source () const
```




<hr>



### function type 

```C++
inline const char * Setting::type () override const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

