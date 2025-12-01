# Fortran interface

This section describes how to integrate `OpenOrbitalOptimizer` into a
Fortran program.

## Preparing the interface

First, create a file named `ooo_module.F90`. The uppercase file extension is required so that the C preprocessor is invoked. The contents of this file should be:

```c
#include <OpenOrbitalOptimizer.F90>
```

Include this file in your project build system (for example, in the
`Makefile`) so that it is compiled together with your application.

## Defining the Fock Builder Callback

Your code must provide a subroutine that constructs the Fock matrix. A minimal implementation is shown below:

In your code, create a function that creates a Fock matrix as follows:

```fortran
subroutine c_fock_builder(Norb, C, n, F, Etot)
  use iso_c_binding
  implicit none

  ! Number of molecular orbitals
  integer(c_int64_t), intent(in), value :: Norb

  ! Partial occupation numbers
  real(c_double), intent(in)            :: n(Norb)

  ! Molecular orbital coefficients in a fixed orthonormal basis
  real(c_double), intent(in)            :: C(Norb,Norb)

  ! Fock matrix in the same orthonormal basis
  real(c_double), intent(out)           :: F(Norb,Norb)

  ! Total energy
  real(c_double), intent(out)           :: Etot

  

  ! Transform the C matrix from the orthonormal basis to the AO basis
  ! C_{ao} = X * C
  
  ! Build the AO basis Fock matrix F_{ao} and compute the total energy

  ! Transform the Fock matrix from the AO basis to the orthonormal basis
  ! F = X^T * F_{ao} * X
  
  
end subroutine c_fock_builder
```

This routine must compute the Fock matrix and total energy for the current
orbitals and occupation numbers and return them through `F` and `Etot`.

## Optional Printing Callback

You may also supply a callback function that the library will invoke at each
SCF iteration, allowing custom output formatting. An example is:

```fortran
subroutine c_print_callback(handle)
  use OpenOrbitalOptimizer
  use iso_c_binding 
  implicit none

  ! Handle used internally by the C++ library
  type(c_ptr), intent(in), value :: handle

  ! Variables to print
  integer(c_int64_t) :: iteration_SCF, dim_DIIS
  real(c_double)     :: energy_SCF

  ! Retrieve values from the optimizer
  iteration_SCF = ooo_get_int64(handle, 'iter')
  energy_SCF = ooo_get_double(handle, 'E')

  ! Display them in any desired format
  write(*,'(I4, 1X, F16.10)')  iteration_SCF, energy_SCF
end subroutine c_print_callback

```

Once the callbacks are defined, the SCF driver can be invoked as follows:

```fortran
subroutine my_scf(Norb, Nelec, C, n, energy)
  use iso_c_binding
  use OpenOrbitalOptimizer
  implicit none

  ! Number of orbitals and electrons
  integer, intent(in) :: Norb, Nelec

  ! Initial guess for the orbitals
  double precision, intent(in) :: C(Norb,Norb)

  ! Occupation numbers
  double precision, allocatable :: n(Norb)

  ! Hartree-Fock energy (result)
  double precision, intent(out) :: energy
  
  ! Function pointers to the user callbacks
  type(C_FUNPTR) :: c_fock, c_print

  c_fock  = c_funloc(c_fock_builder)
  c_print = c_funloc(c_print_callback) 

  ! SCF call; the final argument defines the convergence threshold.
  ! A value of 0.d0 uses the library default.
  energy = rhf_solve_nosym(c_fock, c_print, Norb*1_8, Nelec*1_8, C, n, 0.d0)

end subroutine my_scf

```
