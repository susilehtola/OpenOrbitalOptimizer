

# Class OpenOrbitalOptimizer::SCFSolver::LogStream



[**ClassList**](annotated.md) **>** [**LogStream**](classOpenOrbitalOptimizer_1_1SCFSolver_1_1LogStream.md)



[More...](#detailed-description)






































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**LogStream**](#function-logstream-13) (const [**SCFSolver**](classOpenOrbitalOptimizer_1_1SCFSolver.md#function-scfsolver-12) \* s, int level) <br> |
|   | [**LogStream**](#function-logstream-23) (const LogStream &) = delete<br> |
|   | [**LogStream**](#function-logstream-33) (LogStream && o) noexcept<br> |
|  bool | [**enabled**](#function-enabled) () const<br> |
|  LogStream & | [**operator&lt;&lt;**](#function-operator) (const T & v) <br> |
|  LogStream & | [**operator&lt;&lt;**](#function-operator_1) (std::ostream &(\*)(std::ostream &) manip) <br> |
|  LogStream & | [**operator=**](#function-operator_2) (const LogStream &) = delete<br> |
|  std::ostream & | [**stream**](#function-stream) () <br> |
|   | [**~LogStream**](#function-logstream) () <br> |




























## Detailed Description


Ostream-style companion to log\_(). Constructed via `log_stream_(level)`; accumulates output through `operator<<` and flushes to the logger / stdout on destruction. If `verbosity_ < level` the object is inert (its `operator<<` is a no-op), so the caller writes `log_stream_(N) << ...` in place of `if(verbosity_>=N) std::cout << ...` without a manual gate. 


    
## Public Functions Documentation




### function LogStream [1/3]

```C++
inline LogStream::LogStream (
    const SCFSolver * s,
    int level
) 
```




<hr>



### function LogStream [2/3]

```C++
LogStream::LogStream (
    const LogStream &
) = delete
```




<hr>



### function LogStream [3/3]

```C++
inline LogStream::LogStream (
    LogStream && o
) noexcept
```




<hr>



### function enabled 

```C++
inline bool LogStream::enabled () const
```



True iff the log gate opened and writes will be flushed. Use to short-circuit expensive formatting when the sink would discard it anyway. 


        

<hr>



### function operator&lt;&lt; 

```C++
template<class T>
inline LogStream & LogStream::operator<< (
    const T & v
) 
```




<hr>



### function operator&lt;&lt; 

```C++
inline LogStream & LogStream::operator<< (
    std::ostream &(*)(std::ostream &) manip
) 
```




<hr>



### function operator= 

```C++
LogStream & LogStream::operator= (
    const LogStream &
) = delete
```




<hr>



### function stream 

```C++
inline std::ostream & LogStream::stream () 
```



Direct handle on the underlying std::ostream buffer, for helpers that write via a std::ostream& argument (e.g. print\_settings). Writes when the gate is closed still land in oss\_ but are dropped when the LogStream destructs. 


        

<hr>



### function ~LogStream 

```C++
inline LogStream::~LogStream () 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `openorbitaloptimizer/scfsolver.hpp`

