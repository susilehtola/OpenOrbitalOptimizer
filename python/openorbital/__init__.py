"""OpenOrbitalOptimizer Python bindings.

OpenOrbitalOptimizer is a header-only C++17 library for self-consistent
field (SCF) orbital optimization in quantum chemistry. The library
expects the caller to supply a Fock-builder callback that maps a
density matrix to an energy and Fock matrix in an orthonormal basis;
the library then performs DIIS / A-DIIS / E-DIIS extrapolation,
optimal damping with skeleton density matrices for degenerate shells,
and preconditioned Polak-Ribiere conjugate-gradient orbital rotations.

The single class exposed by this binding is ``SCFSolver`` (for real
orbital coefficients and real bases at double precision); the four
template instantiations of the underlying C++ library can be added if
the use case requires complex orbital coefficients or single precision.

Solver options are configured through a string-keyed façade:
``solver.set(key, value)``, ``solver.get_real / get_int / get_string(key)``,
and the class-level catalog ``SCFSolver.options()``. For interactive use
this module also exposes each solver as ``solver.settings`` -- an
attribute-style proxy so ``solver.settings.convergence_threshold =
1e-9`` is equivalent to ``solver.set("convergence_threshold", 1e-9)``,
with the same catalog-backed name check.
"""

from ._ext import SCFSolver, OptionInfo


class Settings:
    """Attribute-style proxy for :class:`SCFSolver` options.

    Backed by the same catalog as ``SCFSolver.options()``:

    * ``solver.settings.<key>`` dispatches to
      ``solver.get_real/int/string(<key>)`` based on the catalog entry.
    * ``solver.settings.<key> = value`` dispatches to
      ``solver.set(<key>, value)``.
    * Unknown names and writes to read-only diagnostics raise
      ``AttributeError``.
    * ``dir(solver.settings)`` lists every catalog key -- friendly to
      IPython/Jupyter tab-completion.
    """

    __slots__ = ("_solver", "_meta")

    def __init__(self, solver):
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(
            self,
            "_meta",
            {o.key: (o.type, o.writable) for o in SCFSolver.options()},
        )

    def _entry(self, name):
        entry = self._meta.get(name)
        if entry is None:
            raise AttributeError(
                "Unknown solver setting %r. Use SCFSolver.options() to "
                "list the catalog." % name
            )
        return entry

    def __getattr__(self, name):
        # __slots__ members are looked up before __getattr__ fires, so
        # only real catalog names reach here.
        t, _ = self._entry(name)
        s = self._solver
        if t == "real":
            return s.get_real(name)
        if t == "int":
            return s.get_int(name)
        if t == "string":
            return s.get_string(name)
        raise TypeError("Unknown catalog type %r for %r" % (t, name))

    def __setattr__(self, name, value):
        t, writable = self._entry(name)
        if not writable:
            raise AttributeError(
                "Setting %r is a read-only diagnostic." % name
            )
        self._solver.set(name, value)

    def __dir__(self):
        return sorted(self._meta)

    def __repr__(self):
        return "<Settings on %r>" % (self._solver,)


# Expose settings on every SCFSolver instance via a property. The
# proxy is cheap to build (its catalog dict is small and rebuilt lazily
# per access), so no caching is needed.
SCFSolver.settings = property(lambda self: Settings(self))


__all__ = ["SCFSolver", "OptionInfo", "Settings"]
