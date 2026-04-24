"""
tests/test_boys.py — Unit tests for the Boys function implementation.

Tests cover:
  1. Exact values at x=0: F_n(0) = 1/(2n+1)
  2. Known analytical value F_0(x) = sqrt(pi)*erf(sqrt(x))/(2*sqrt(x))
  3. Accuracy vs. the original hyp1f1-based reference across n=0..4 and a
     wide range of x values — must agree to within 1e-10 relatively.
  4. Recurrence consistency: downward recurrence must reproduce F_n values
     computed independently.
  5. Monotonicity: F_n(x) must decrease as x increases (for fixed n).
  6. Ordering: F_n(x) > F_{n+1}(x) for all x > 0 (higher-order Boys is smaller).

Reference implementation uses scipy.special.hyp1f1 and is kept in this test
file only — it is NOT used in the production path.
"""

import math
import pytest
import numpy as np
from scipy.special import hyp1f1

from molekul.integrals import _boys


# ---------------------------------------------------------------------------
# Reference: original hyp1f1-based Boys function
# ---------------------------------------------------------------------------

def _boys_ref(n: int, x: float) -> float:
    """Original implementation — used only as reference in these tests."""
    if x < 1e-8:
        return 1.0 / (2 * n + 1)
    return hyp1f1(n + 0.5, n + 1.5, -x) / (2 * n + 1)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

N_RANGE = range(5)   # n = 0, 1, 2, 3, 4  (full range for STO-3G)

X_VALUES = [
    0.0, 1e-12, 1e-8, 1e-7, 1e-5, 1e-3,
    0.01, 0.1, 0.3, 0.499, 0.5, 0.501,
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0,
]

# Very tight absolute tolerance for near-zero values, relative for others
ABS_TOL  = 1e-12
REL_TOL  = 1e-10   # well below the ERI pipeline's 1e-8 requirement


# ---------------------------------------------------------------------------
# 1. Exact boundary: F_n(0) = 1/(2n+1)
# ---------------------------------------------------------------------------

class TestBoysAtZero:
    @pytest.mark.parametrize("n", N_RANGE)
    def test_exact_at_zero(self, n):
        expected = 1.0 / (2 * n + 1)
        assert _boys(n, 0.0)   == pytest.approx(expected, rel=1e-14)
        assert _boys(n, 1e-12) == pytest.approx(expected, rel=1e-10)

    @pytest.mark.parametrize("n", N_RANGE)
    def test_x_below_cutoff(self, n):
        """x < 1e-8 must return the constant limit exactly."""
        assert _boys(n, 0.0) == 1.0 / (2 * n + 1)
        assert _boys(n, 5e-9) == 1.0 / (2 * n + 1)


# ---------------------------------------------------------------------------
# 2. Analytical value for F_0
# ---------------------------------------------------------------------------

class TestBoysF0Analytical:
    @pytest.mark.parametrize("x", [v for v in X_VALUES if v > 1e-8])
    def test_f0_vs_erf(self, x):
        """F_0(x) = sqrt(pi) * erf(sqrt(x)) / (2*sqrt(x)) — exact identity."""
        expected = 0.5 * math.sqrt(math.pi / x) * math.erf(math.sqrt(x))
        result   = _boys(0, x)
        assert result == pytest.approx(expected, rel=REL_TOL, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# 3. Accuracy vs. hyp1f1 reference
# ---------------------------------------------------------------------------

class TestBoysVsReference:
    @pytest.mark.parametrize("n", N_RANGE)
    @pytest.mark.parametrize("x", X_VALUES)
    def test_agrees_with_hyp1f1(self, n, x):
        ref = _boys_ref(n, x)
        new = _boys(n, x)
        if abs(ref) > 1e-20:
            assert abs(new - ref) / abs(ref) < REL_TOL, (
                f"n={n}, x={x}: ref={ref:.10e}, new={new:.10e}, "
                f"relerr={abs(new-ref)/abs(ref):.2e}"
            )
        else:
            assert abs(new - ref) < ABS_TOL, (
                f"n={n}, x={x}: ref={ref:.10e}, new={new:.10e}"
            )

    def test_max_error_over_grid(self):
        """Grid scan: max relative error must be < 1e-10."""
        xs = np.concatenate([
            np.logspace(-8, -1, 30),   # small x
            np.linspace(0.5, 100, 60), # intermediate / large x
        ])
        max_err = 0.0
        for n in N_RANGE:
            for x in xs:
                ref = _boys_ref(n, float(x))
                new = _boys(n,   float(x))
                if abs(ref) > 1e-20:
                    err = abs(new - ref) / abs(ref)
                    if err > max_err:
                        max_err = err
        assert max_err < REL_TOL, f"Max relative error {max_err:.2e} exceeds {REL_TOL:.0e}"


# ---------------------------------------------------------------------------
# 4. Recurrence consistency check
# ---------------------------------------------------------------------------

class TestBoysRecurrence:
    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 5.0, 20.0])
    def test_downward_recurrence(self, x):
        """
        Downward recurrence: F_{n-1}(x) = [(2n-1)*F_{n-1}... check self-consistency.

        Identity: F_{n-1}(x) = (2x*F_n(x) + exp(-x)) / (2n-1)
        => 2x*F_n + exp(-x) = (2n-1)*F_{n-1}
        """
        emx = math.exp(-x)
        for n in range(1, 5):
            Fn   = _boys(n,   x)
            Fn_1 = _boys(n-1, x)
            # Test: (2n-1)*F_{n-1} = 2x*F_n + exp(-x)
            lhs = (2 * n - 1) * Fn_1
            rhs = 2 * x * Fn + emx
            assert abs(lhs - rhs) < 1e-9, (
                f"Recurrence failed at n={n}, x={x}: "
                f"lhs={lhs:.10e} rhs={rhs:.10e} delta={abs(lhs-rhs):.2e}"
            )


# ---------------------------------------------------------------------------
# 5. Monotonicity in x (for fixed n)
# ---------------------------------------------------------------------------

class TestBoysMonotonicity:
    @pytest.mark.parametrize("n", N_RANGE)
    def test_decreasing_in_x(self, n):
        """F_n(x) is strictly decreasing in x > 0."""
        xs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        vals = [_boys(n, x) for x in xs]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i+1], (
                f"n={n}: F_n({xs[i]})={vals[i]:.8f} not > F_n({xs[i+1]})={vals[i+1]:.8f}"
            )


# ---------------------------------------------------------------------------
# 6. Ordering in n (for fixed x)
# ---------------------------------------------------------------------------

class TestBoysOrdering:
    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 20.0])
    def test_decreasing_in_n(self, x):
        """F_n(x) > F_{n+1}(x) for all n >= 0, x > 0."""
        vals = [_boys(n, x) for n in range(5)]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i+1], (
                f"x={x}: F_{i}={vals[i]:.8f} not > F_{i+1}={vals[i+1]:.8f}"
            )


# ---------------------------------------------------------------------------
# 7. ERI consistency: H2O ERI tensor must be unchanged (within 1e-8)
# ---------------------------------------------------------------------------

class TestERIConsistency:
    def test_h2o_eri_unchanged(self):
        """
        Full ERI tensor for H2O must agree with the hyp1f1-based reference
        to within 1e-8 (absolute), confirming the Boys change did not affect
        ERI values.

        The reference is computed by temporarily patching _boys with hyp1f1.
        """
        import math as _math
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

        from molekul.atoms import Atom
        from molekul.molecule import Molecule
        from molekul.basis_sto3g import STO3G
        from molekul.eri import build_eri
        import molekul.integrals as _intmod

        # Build H2O
        R, th = 1.870, _math.radians(50.0)
        mol = Molecule([
            Atom("O",  [0.0, 0.0, 0.0]),
            Atom("H",  [0.0,  R*_math.sin(th), -R*_math.cos(th)]),
            Atom("H",  [0.0, -R*_math.sin(th), -R*_math.cos(th)]),
        ], charge=0, multiplicity=1)

        # ERI with new boys
        eri_new = build_eri(STO3G, mol)

        # ERI with reference boys (patch temporarily)
        orig_boys = _intmod._boys
        _intmod._boys = _boys_ref
        try:
            _intmod._E.cache_clear()   # clear cache to force re-evaluation
            eri_ref = build_eri(STO3G, mol)
        finally:
            _intmod._boys = orig_boys
            _intmod._E.cache_clear()

        max_diff = float(np.max(np.abs(eri_new - eri_ref)))
        assert max_diff < 1e-8, (
            f"Max ERI difference = {max_diff:.2e} exceeds 1e-8"
        )
