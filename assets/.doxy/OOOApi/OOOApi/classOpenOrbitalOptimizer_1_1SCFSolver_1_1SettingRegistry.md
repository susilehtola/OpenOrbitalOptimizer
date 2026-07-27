

# Class OpenOrbitalOptimizer::SCFSolver::SettingRegistry



[**ClassList**](annotated.md) **>** [**SettingRegistry**](classOpenOrbitalOptimizer_1_1SCFSolver_1_1SettingRegistry.md)



[More...](#detailed-description)






































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**add**](#function-add) (const SettingBase \* setting) <br> |
|  std::vector&lt; SettingBase \* &gt; | [**settings**](#function-settings-12) () <br> |
|  std::vector&lt; const SettingBase \* &gt; | [**settings**](#function-settings-22) () const<br> |




























## Detailed Description


Collects the solver's settings as they are constructed, so that declaring a setting is all it takes to publish it through the string facade  there is no second list to keep in step.


It records each setting's offset from the registry rather than its address. An address would belong to the object the setting was constructed in, and [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md) is movable, so a registry of addresses would point into the moved-from object. Offsets describe the class layout instead, which is by definition the same in the moved-to object, so the implicitly generated move stays correct and needs no fixup. 


    
## Public Functions Documentation




### function add 

```C++
inline void SettingRegistry::add (
    const SettingBase * setting
) 
```



Called by each Setting's constructor. The conversion to SettingBase \* happens in the caller, so what is recorded is the offset of the base subobject  exactly what settings() hands back. 


        

<hr>



### function settings [1/2]

```C++
inline std::vector< SettingBase * > SettingRegistry::settings () 
```




<hr>



### function settings [2/2]

```C++
inline std::vector< const SettingBase * > SettingRegistry::settings () const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

