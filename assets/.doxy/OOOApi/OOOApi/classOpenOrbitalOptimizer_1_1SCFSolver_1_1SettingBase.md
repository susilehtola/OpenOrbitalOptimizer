

# Class OpenOrbitalOptimizer::SCFSolver::SettingBase



[**ClassList**](annotated.md) **>** [**SettingBase**](classOpenOrbitalOptimizer_1_1SCFSolver_1_1SettingBase.md)



_Everything about a setting that does not depend on its type._ 






Inherited by the following classes: [OpenOrbitalOptimizer::SCFSolver::Setting](classOpenOrbitalOptimizer_1_1SCFSolver_1_1Setting.md),  [OpenOrbitalOptimizer::SCFSolver::Setting](classOpenOrbitalOptimizer_1_1SCFSolver_1_1Setting.md),  [OpenOrbitalOptimizer::SCFSolver::Setting](classOpenOrbitalOptimizer_1_1SCFSolver_1_1Setting.md)
































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**SettingBase**](#function-settingbase) (const char \* key, const char \* doc, bool writable) <br> |
|  const char \* | [**doc**](#function-doc) () const<br> |
|  const char \* | [**key**](#function-key) () const<br> |
| virtual void | [**print\_value**](#function-print_value) (std::ostream & os) const = 0<br>_Used by print\_settings, which does not know the value type._  |
| virtual const char \* | [**type**](#function-type) () const = 0<br>_"real", "int" or "string"_  _which typed facade reaches this._ |
|  bool | [**writable**](#function-writable) () const<br>_False for read-only diagnostics, which the facade refuses to set._  |
| virtual  | [**~SettingBase**](#function-settingbase) () = default<br> |




























## Public Functions Documentation




### function SettingBase 

```C++
inline SettingBase::SettingBase (
    const char * key,
    const char * doc,
    bool writable
) 
```




<hr>



### function doc 

```C++
inline const char * SettingBase::doc () const
```




<hr>



### function key 

```C++
inline const char * SettingBase::key () const
```




<hr>



### function print\_value 

_Used by print\_settings, which does not know the value type._ 
```C++
virtual void SettingBase::print_value (
    std::ostream & os
) const = 0
```




<hr>



### function type 

_"real", "int" or "string"_  _which typed facade reaches this._
```C++
virtual const char * SettingBase::type () const = 0
```




<hr>



### function writable 

_False for read-only diagnostics, which the facade refuses to set._ 
```C++
inline bool SettingBase::writable () const
```




<hr>



### function ~SettingBase 

```C++
virtual SettingBase::~SettingBase () = default
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

