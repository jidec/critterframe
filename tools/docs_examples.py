"""mkdocs hook: build the Examples tab from the pipeline scripts, so the pages cannot drift from the code."""

import ast
import html
import logging
import re
from pathlib import Path

from mkdocs.structure.files import File

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "scripts" / "reusable_workflows"
FULL_PROJECT = REPO / "scripts" / "author_pipelines" / "odonata_inat_obsorg"

# The nav section the pages go under, and the entry it is placed after.
SECTION = "Examples"
AFTER = "Guide"

# mkdocs counts a warning from a logger under "mkdocs", so a strict build fails on one.
logger = logging.getLogger("mkdocs.hooks.docs_examples")


def _split(path):
    """Return a script's `(docstring, code after it)`; the docstring is None when it has none.

    Raises:
        SyntaxError: If the script does not parse.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree)
    lines = source.splitlines()
    if docstring is not None:
        lines = lines[tree.body[0].end_lineno :]
    return docstring, "\n".join(lines).strip("\n")


def _order(path, docstring):
    """Return a script's sort key: its "Step N of M", else a number leading its name, else its name."""
    step = re.search(r"\bStep (\d+) of \d+", docstring or "")
    leading = re.match(r"(\d+)_", path.name)
    number = int(step.group(1)) if step else int(leading.group(1)) if leading else float("inf")
    return (number, path.name)


_LIST_ITEM = re.compile(r"\s*(\d+\.|[-*])\s")


def _as_prose(docstring):
    """Return a plain-text docstring as Markdown that renders the way it reads.

    Angle brackets are escaped, so text like `<occurrence_id>__<part>.png` is not swallowed as
    an HTML tag. A list that follows its lead-in line directly gets the blank line Markdown
    needs before it, or it would run on as one paragraph.
    """
    lines = []
    for line in html.escape(docstring, quote=False).splitlines():
        previous = lines[-1] if lines else ""
        starts_list = (
            _LIST_ITEM.match(line) and not _LIST_ITEM.match(previous) and not previous.startswith(" ")
        )
        if starts_list and previous.strip():
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


def _section(path, repo_url):
    """Return one script as Markdown: a heading, its docstring as prose, its code in a block."""
    docstring, code = _split(path)
    relative = path.relative_to(REPO).as_posix()
    parts = [f"## `{path.name}`", f"[`{relative}`]({repo_url.rstrip('/')}/blob/main/{relative})"]
    if docstring:
        parts.append(_as_prose(docstring))
    parts.append(f"````python\n{code}\n````")
    return "\n\n".join(parts)


def _page(title, intro, scripts, repo_url):
    """Return one Examples page as Markdown, its scripts in run order; unparseable ones are skipped."""
    readable = []
    for path in scripts:
        try:
            readable.append((_order(path, _split(path)[0]), path))
        except SyntaxError as error:
            logger.warning("examples: skipping %s, which does not parse: %s", path.relative_to(REPO), error)
    sections = [_section(path, repo_url) for _key, path in sorted(readable)]
    return "\n\n".join([f"# {title}", intro, *sections]) + "\n"


def _title(folder_name):
    """Return a folder name as a page title: `custom_segmentation_model` -> `Custom segmentation model`."""
    return folder_name.replace("_", " ").capitalize()


def _pages(repo_url):
    """Return `[(nav title, page path, Markdown)]` for every Examples page, in nav order."""
    pages = []

    simplest = WORKFLOWS / "simplest_full_pipeline.py"
    if simplest.exists():
        pages.append(
            (
                "The simplest pipeline",
                "examples/simplest_full_pipeline.md",
                _page(
                    "The simplest pipeline",
                    "The place to start: every step once, end to end.",
                    [simplest],
                    repo_url,
                ),
            )
        )

    for folder in sorted(
        path for path in WORKFLOWS.iterdir() if path.is_dir() and path.name != "__pycache__"
    ):
        scripts = sorted(folder.glob("*.py"))
        if not scripts:
            continue
        intro = f"The scripts in `scripts/reusable_workflows/{folder.name}/`, in the order to run them."
        pages.append(
            (
                _title(folder.name),
                f"examples/{folder.name}.md",
                _page(_title(folder.name), intro, scripts, repo_url),
            )
        )

    steps = sorted(path for path in FULL_PROJECT.glob("*.py") if re.match(r"\d+_", path.name))
    if steps:
        intro = (
            f"A whole project as it was run, one numbered script per step: `scripts/author_pipelines/{FULL_PROJECT.name}/`. "
            "Its paths and occurrence ids are that project's own."
        )
        pages.append(
            ("A full project", "examples/full_project.md", _page("A full project", intro, steps, repo_url))
        )

    return pages


def on_config(config):
    """Add the Examples section to the nav, after the Guide."""
    entries = [{title: path} for title, path, _content in _pages(config["repo_url"])]
    nav = config["nav"]
    position = next((index + 1 for index, item in enumerate(nav) if AFTER in item), len(nav))
    nav.insert(position, {SECTION: entries})
    return config


def on_files(files, config):
    """Add each Examples page to the build, generated in memory."""
    for _title_, path, content in _pages(config["repo_url"]):
        files.append(File.generated(config, path, content=content))
    return files
