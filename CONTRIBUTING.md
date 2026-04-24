# Contributing to MOLEKUL

Thank you for your interest in MOLEKUL.

## Reporting Issues

Open an issue on GitHub describing:
- What you expected to happen
- What actually happened
- A minimal reproducible example (XYZ file + basis + method)

## Development Setup

```bash
git clone https://github.com/yazicienis/molekul
cd molekul
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

All tests must pass before submitting a pull request. The test suite validates
results against PySCF at sub-μHartree tolerances.

## Code Style

- Pure Python and NumPy only (no compiled extensions in the core)
- Functions map 1-to-1 to textbook equations; preserve that correspondence
- New integral or SCF code must include a reference to the source equation

## Scope

MOLEKUL is intentionally a small, readable, educational codebase. Contributions
that add production-level complexity (sparse storage, MPI, compiled back-ends)
belong in a separate project. Contributions that make the code more transparent
or extend the validated feature set modestly are welcome.

## Pull Request Checklist

- [ ] Tests added for new functionality
- [ ] Existing tests still pass
- [ ] New basis data validated against EMSL Basis Set Exchange and PySCF
- [ ] Documentation updated if public API changed
