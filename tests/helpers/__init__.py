"""
Support code for the test suite: synthetic data, stub models, fake externals.

Importable as `helpers.*` because `pythonpath = ["tests"]` is set in
pyproject.toml, and never collected as tests because nothing here is named
`test_*.py`.

The smoke scripts under `scripts/simple_tests/` import from here too. That is
the point of the package existing rather than each side drawing its own
specimens: the scripts show a person what an operation did, the suite asserts
what it computed, and both must be looking at the same thing for either to mean
anything.
"""
