"""
What `import critterframe` must be true of.

Two rules that CLAUDE.md states and nothing has enforced. Both are the kind that
break by accident -- someone moves an import to the top of a module for
readability and the core quietly grows a torch dependency -- and neither shows
up as a failing test anywhere else, because a developer with torch installed
never notices.
"""

import os
import subprocess
import sys

import pytest

import critterframe as cf


def _import_reports(module_names):
    """
    Import critterframe in a FRESH interpreter and report which of the named
    modules ended up in sys.modules.

    A subprocess because this test is worthless in-process: the suite imports
    torch elsewhere (the gpu tests), pytest itself imports half the world, and
    `sys.modules` here reflects the session rather than the package.
    """
    code = (
        "import sys; import critterframe; "
        f"print([name for name in {module_names!r} if name in sys.modules])"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def test_import_pulls_in_no_heavy_dependency():
    """
    The core has no deep-learning dependency, deliberately, and a new core
    module must not add one. Torch is the one that matters -- the whole
    `[torch]` extra exists so a project segmenting with its own model is
    not made to install SAM2 -- and sklearn is the same bargain one size down.

    If this fails, look for an `import torch` that moved from inside a function
    to the top of a module.
    """
    assert _import_reports(["torch", "transformers", "sklearn"]) == "[]"


def test_import_does_not_read_a_dotenv():
    """
    Importing the package must not reach out and mutate the environment.

    `extensions/antenna_lighttraps/api.py` used to call `load_dotenv()` at
    import, so anything importing it inherited whatever .env sat beside the
    working directory. It is lazy now; this is what keeps it that way.
    """
    code = (
        "import os; "
        "import critterframe.extensions.antenna_lighttraps.api as api; "
        "print(sorted(n for n in os.environ if n.startswith('ANTENNA_')))"
    )
    # A scrubbed copy of the real environment, and the repo root as the working
    # directory -- so if the module did read .env, this would see it.
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("ANTENNA_")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, check=True, env=environment,
                            cwd=os.path.dirname(os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__)))))
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", cf.__all__)
def test_every_exported_name_resolves(name):
    """
    `__all__` is what `from critterframe import *` gives and what the package
    docstring advertises. A name listed there but not imported is a typo nobody
    finds until a user's script fails.
    """
    assert hasattr(cf, name), f"critterframe.__all__ names {name}, which isn't there"


def test_exported_names_are_sorted_within_their_groups():
    """
    `__all__` lists classes first, then lowercase callables alphabetically. Not
    a correctness property -- a readability one, and the sort is what keeps a
    hand-maintained 75-line list mergeable.
    """
    lowercase = [name for name in cf.__all__ if name[0].islower()]
    assert lowercase == sorted(lowercase)
