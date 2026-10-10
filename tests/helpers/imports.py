"""What an ingest left in a project's raw-import archive."""

from critterframe.project import paths


def imports_of(project_path):
    """Return the archived raw CSVs, sorted by name."""
    return sorted(paths.raw_imports_dir(project_path).glob("*.csv"))
