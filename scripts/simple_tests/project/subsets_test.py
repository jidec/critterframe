"""
Define subsets and see what each one actually selects.

Worth doing before running a per-subset recipe: a subset selecting zero
occurrences fails quietly later, because a run that processes nothing looks a
lot like a run whose work was already done. The counts here tell them apart.

Needs a project with occurrences ingested.

Run from the repo root:
    python scripts/simple_tests/project/subsets_test.py
"""

import logging

from critterframe.project import paths, subsets
from critterframe.records.occurrences import load_occurrences

logging.basicConfig(level=logging.INFO)

PROJECT_PATH = "projects/my_project"
GROUP_COLUMN = "collection"

occurrences = load_occurrences(PROJECT_PATH)
print(f"{len(occurrences)} occurrences, columns: {sorted(occurrences.columns)}")

if GROUP_COLUMN in occurrences.columns:
    values = occurrences[GROUP_COLUMN].dropna().unique()
    print(f"\n{GROUP_COLUMN} values: {list(values)}")

    # Name each subset after its value, tidied. definitions/subsets.toml is
    # hand-editable afterward, which is the point -- knowing which collection
    # is which is human knowledge about the data.
    subsets.define_subsets(
        PROJECT_PATH,
        column=GROUP_COLUMN,
        mapping={value: str(value).lower().replace(" ", "_") for value in values},
    )
else:
    print(f"\nno '{GROUP_COLUMN}' column here -- edit GROUP_COLUMN above")

print(f"\ndefinitions at {paths.subsets_path(PROJECT_PATH)}:")
for name, definition in sorted(subsets.load_subsets(PROJECT_PATH).items()):
    selected = subsets.select_ids(PROJECT_PATH, subset=name)
    print(f"  {name:<20} {definition}  -> {len(selected)} occurrence(s)")
