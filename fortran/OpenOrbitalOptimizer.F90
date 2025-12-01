module OpenOrbitalOptimizer

  use, intrinsic :: iso_c_binding

  interface
    function rhf_solve_nosym(c_fock_builder, c_print_callback, Norb, Nelec, Cp, np, convergence) bind(C)
      import
      type(C_FUNPTR), value :: c_fock_builder
      type(C_FUNPTR), value :: c_print_callback
      integer(C_INT64_T), value :: Norb
      integer(C_INT64_T), value :: Nelec
      real(C_DOUBLE), intent(inout) :: Cp(*) ! orbital matrix (Norb x Norb flattened in C order)
      real(C_DOUBLE), intent(inout) :: np(*) ! occupations or extra data (length depends on caller)
      real(C_DOUBLE),  value :: convergence ! Convergence threshold. If zero, then use the library's default
      real(C_DOUBLE) :: rhf_solve_nosym
    end function
  end interface

  interface
    real(c_double) function ooo_get_double(handle, key) bind(C)
      import
      type(C_PTR), value :: handle
      character(c_char) :: key(*)
    end function
  end interface

  interface
    integer(c_int64_t) function ooo_get_int64(handle, key) bind(C)
      import
      type(C_PTR), value :: handle
      character(c_char) :: key(*)
    end function
  end interface

end module

