"""Zero-dependency test runner.

`pytest` is the intended way to run these tests:

    pip install pytest && python -m pytest tests/ -v

But the offline build environment has no network and therefore no pytest, and
shipping tests that have never been executed is worse than shipping none. This
runner installs a minimal `pytest` shim (just `approx`), imports the test
module, and executes every `test_*` function, so the suite can always be run:

    python tests/run_tests.py
"""

import importlib.util
import os
import sys
import traceback
import types


def _install_pytest_shim():
    if "pytest" in sys.modules:
        return
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass

    shim = types.ModuleType("pytest")

    class _Approx:
        def __init__(self, expected, rel=1e-6, abs=None):
            self.expected, self.rel, self.abs = expected, rel, abs

        def __eq__(self, actual):
            if self.abs is not None:
                return abs(actual - self.expected) <= self.abs
            tol = self.rel * max(abs(self.expected), abs(actual), 1e-12)
            return abs(actual - self.expected) <= tol

        def __repr__(self):
            return f"approx({self.expected})"

    shim.approx = _Approx
    sys.modules["pytest"] = shim


def main():
    _install_pytest_shim()

    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "test_equity_factors.py")
    spec = importlib.util.spec_from_file_location("test_equity_factors", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tests = sorted(
        (name, fn) for name, fn in vars(module).items()
        if name.startswith("test_") and callable(fn)
    )

    passed, failures = 0, []
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures.append((name, traceback.format_exc()))
            print(f"FAIL  {name}")
        else:
            passed += 1
            print(f"ok    {name}")

    print(f"\n{passed} passed, {len(failures)} failed, {len(tests)} total")
    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
