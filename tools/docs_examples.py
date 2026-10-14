"""mkdocs hook: build the Examples tab from the pipeline scripts, so the pages cannot drift from the code."""

import ast
import html
import logging
import re
from pathlib import Path

from mkdocs.structure.files import File

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# The folders under scripts/ whose pipelines are shown, with the nav group each gets.
GROUPS = (("reusable_workflows", "Reusable workflows"), ("author_pipelines", "Author pipelines"))

# The page linked first, outside its group: the pipeline to start from.
START_HERE = "reusable_workflows/simplest_full_pipeline"

# A folder named with this suffix is a pipeline that has not been run end to end.
UNTESTED = "_untested"

# The nav section the pages go under, and the entry it is placed after.
SECTION = "Examples"
AFTER = "Concepts"

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


def _is_scratch(path):
    """Return whether a file or folder is working scratch, by its name, and so left off the site."""
    return path.name.startswith(("_", ".", "test", "temp"))


def _title(name):
    """Return a file or folder name as a page title: `custom_segmentation_model` -> `Custom segmentation model`."""
    return name.removesuffix(UNTESTED).replace("_", " ").capitalize()


def _readable(scripts):
    """Return `[(path, docstring)]` for the scripts that parse, in run order; the rest are skipped."""
    readable = []
    for path in scripts:
        try:
            docstring = _split(path)[0]
        except SyntaxError as error:
            logger.warning("examples: skipping %s, which does not parse: %s", path.relative_to(REPO), error)
            continue
        readable.append((_order(path, docstring), path, docstring))
    return [(path, docstring) for _key, path, docstring in sorted(readable)]


def _discover():
    """Return every Examples page as a dict of `key`, `group`, `title`, `path`, `intro` and `scripts`.

    A page is a folder directly under one of `GROUPS`, holding the `.py` files directly inside it, or a
    `.py` file directly under one. Subfolders of a page's folder are not read.
    """
    pages = []
    for group, _label in GROUPS:
        root = SCRIPTS / group
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if _is_scratch(entry):
                continue
            if entry.is_dir():
                scripts = [path for path in entry.glob("*.py") if not _is_scratch(path)]
                name = entry.name
                intro = f"The scripts in `scripts/{group}/{name}/`, in the order to run them."
                if name.endswith(UNTESTED):
                    intro += (
                        '\n\n!!! warning "Not yet run end to end"\n    This pipeline is written but untested.'
                    )
            elif entry.suffix == ".py":
                scripts, name, intro = [entry], entry.stem, f"`scripts/{group}/{entry.name}`"
            else:
                continue
            scripts = _readable(scripts)
            if scripts:
                pages.append(
                    {
                        "key": f"{group}/{name}",
                        "group": group,
                        "title": _title(name),
                        "path": f"examples/{group}/{name}.md",
                        "intro": intro,
                        "scripts": scripts,
                    }
                )
    return pages


def _page(page, repo_url):
    """Return one Examples page as Markdown, its scripts in run order."""
    sections = [_section(path, repo_url) for path, _docstring in page["scripts"]]
    return "\n\n".join([f"# {page['title']}", page["intro"], *sections]) + "\n"


def _nav(pages):
    """Return the Examples nav entries: the starting pipeline, then one section per group."""
    entries = [{page["title"]: page["path"]} for page in pages if page["key"] == START_HERE]
    for group, label in GROUPS:
        members = [
            {page["title"]: page["path"]}
            for page in pages
            if page["group"] == group and page["key"] != START_HERE
        ]
        if members:
            entries.append({label: members})
    return entries


def on_config(config):
    """Add the Examples section to the nav."""
    nav = config["nav"]
    position = next((index + 1 for index, item in enumerate(nav) if AFTER in item), len(nav))
    nav.insert(position, {SECTION: _nav(_discover())})
    return config


def on_files(files, config):
    """Add the Examples pages to the build, generated in memory."""
    for page in _discover():
        files.append(File.generated(config, page["path"], content=_page(page, config["repo_url"])))
    return files
