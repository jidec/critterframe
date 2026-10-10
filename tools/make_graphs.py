"""Draw the package's graphs into docs/graphs/: imports between folders, calls within each folder, calls around each entry point.

In a call graph, hovering a function shows the first line of its docstring.
"""

import argparse
import ast
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "critterframe"
# Inside docs/, so the site publishes them; docs/internals.md links to each.
OUTPUT = REPO / "docs" / "graphs"

# The functions a pipeline is made of, as `{graph name: code2flow target}`. Each is drawn from the whole
# package, so the calls a per-folder graph cannot show (a driver into core/, records/, selection/) are in it.
# A name defined in more than one file is given as `<file>::<function>`.
ENTRY_POINTS = {
    "ingest_occurrences": "occurrences::ingest_occurrences",
    "run_segments": "run_segments",
    "run_metrics": "run_metrics",
    "export_metrics": "export_metrics",
}

# where the Windows Graphviz installer puts dot when it leaves PATH alone
WINDOWS_GRAPHVIZ = Path(r"C:\Program Files\Graphviz\bin")


def _require_dot():
    """Put Graphviz's dot on this process's PATH, or exit if it is not installed."""
    if shutil.which("dot") is None and (WINDOWS_GRAPHVIZ / "dot.exe").exists():
        os.environ["PATH"] = f"{WINDOWS_GRAPHVIZ}{os.pathsep}{os.environ['PATH']}"
    if shutil.which("dot") is None:
        sys.exit("Graphviz's `dot` is not on PATH. Install Graphviz (winget install Graphviz.Graphviz).")


def _external_imports():
    """Return the top-level name of every module the package imports from outside itself."""
    names = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    names.discard(PACKAGE.name)
    return sorted(names)


def _pydeps(output, *options):
    """Write one module import graph of the package."""
    # pydeps follows every import to the bottom before --only filters the picture, and the walk
    # through torch and transformers overflows the stack. Excluding what is not ours stops it at
    # the package's edge.
    excluded = [pattern for name in _external_imports() for pattern in (name, f"{name}.*")]
    command = [sys.executable, "-W", "ignore", "-m", "pydeps", PACKAGE.name, "--only", PACKAGE.name]
    command += ["--rmprefix", f"{PACKAGE.name}.", "--no-config", "--noshow", "-o", output, *options]
    command += ["-x", *excluded]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {output.name} (exit {result.returncode})\n{result.stderr.strip()}")
    else:
        print(f"  {output.relative_to(REPO)}")


def _dotted_path(filename):
    """Return a source file's module path inside the package, e.g. `records.masks`."""
    path = Path(filename).resolve()
    try:
        return ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
    except ValueError:
        return path.stem


def _ends_with(dotted, suffix):
    """Return whether a dotted module path is `suffix` or ends with it at a dot."""
    return dotted == suffix or dotted.endswith(f".{suffix}")


def _resolve_import(variable, file_groups, unknown):
    """Return the file, class or function an imported name refers to, matched by module path.

    code2flow knows a file only by its base name, so `from ..records import masks as mask_records`
    and `from ..core.recipes import hash_spec` both go unresolved and the calls made through them
    are dropped. Here the import is matched against each file's path inside the package; a name
    that fits more than one file is left unresolved, since a wrong edge is worse than a missing one.
    """
    imported = variable.points_to
    paths = {group: group.import_tokens[-1] for group in file_groups}

    modules = [group for group, dotted in paths.items() if _ends_with(dotted, imported)]
    if len(modules) == 1:
        return modules[0]

    if "." in imported and not modules:
        module, name = imported.rsplit(".", 1)
        owners = [group for group, dotted in paths.items() if _ends_with(dotted, module)]
        if len(owners) == 1:
            for member in [*owners[0].nodes, *owners[0].subgroups]:
                if member.token == name:
                    return member
    return unknown


def _summary(tree):
    """Return the first line of a function's docstring, or None if it has none."""
    docstring = ast.get_docstring(tree)
    return docstring.strip().splitlines()[0] if docstring and docstring.strip() else None


def _with_tooltip(dot, summary):
    """Return a Graphviz node line with the summary added as its hover tooltip."""
    if not summary:
        return dot
    escaped = summary.replace("\\", "\\\\").replace('"', '\\"')
    return f'{dot[:-1]}tooltip="{escaped}" ]'


def _code2flow(sources, output, target=None, upstream=0, downstream=0):
    """Write one function call graph of the given files or directories."""
    from code2flow import engine
    from code2flow import model as code2flow_model
    from code2flow import python as code2flow_python

    # code2flow asserts on a call whose callee is an expression, e.g. `(log or logger.info)(...)`.
    # Such a call has no name to resolve, so it is skipped like the subscript calls already are.
    original = code2flow_python.get_call_from_func_element
    original_tokens = code2flow_python.Python.file_import_tokens
    original_resolve = code2flow_model._resolve_str_variable
    original_make_nodes = code2flow_python.Python.make_nodes
    original_to_dot = code2flow_model.Node.to_dot

    def make_nodes_with_summary(tree, parent):
        nodes = original_make_nodes(tree, parent)
        for node in nodes:
            node.summary = _summary(tree)
        return nodes

    def tolerant(func):
        try:
            return original(func)
        except AssertionError:
            return None

    code2flow_python.get_call_from_func_element = tolerant
    # Each file also carries its module path, last, for _resolve_import to match on.
    code2flow_python.Python.file_import_tokens = staticmethod(
        lambda filename: [*original_tokens(filename), _dotted_path(filename)]
    )
    code2flow_model._resolve_str_variable = lambda variable, file_groups: _resolve_import(
        variable, file_groups, code2flow_model.OWNER_CONST.UNKNOWN_MODULE
    )
    # Each node carries its docstring's summary line, shown on hover in the SVG.
    code2flow_python.Python.make_nodes = staticmethod(make_nodes_with_summary)
    code2flow_model.Node.to_dot = lambda node: _with_tooltip(
        original_to_dot(node), getattr(node, "summary", None)
    )
    try:
        engine.code2flow(
            raw_source_paths=[str(source) for source in sources],
            output_file=str(output),
            language="py",
            hide_legend=False,
            subset_params=engine.SubsetParams.generate(target, upstream, downstream),
            level=logging.WARNING,
        )
        print(f"  {output.relative_to(REPO)}")
    except (Exception, SystemExit) as error:
        print(f"  FAILED: {output.name}: {error!r}")
    finally:
        code2flow_python.get_call_from_func_element = original
        code2flow_python.Python.file_import_tokens = staticmethod(original_tokens)
        code2flow_model._resolve_str_variable = original_resolve
        code2flow_python.Python.make_nodes = staticmethod(original_make_nodes)
        code2flow_model.Node.to_dot = original_to_dot


def _subpackages(directory):
    """Return the subpackage directories directly under a package directory."""
    return sorted(p for p in directory.iterdir() if (p / "__init__.py").exists())


def main():
    """Write the module graphs and the call graphs, or one call graph centered on --target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="Draw only the call graph around this function, e.g. run_metrics.")
    parser.add_argument("--upstream", type=int, default=1, help="Levels of callers to include.")
    parser.add_argument("--downstream", type=int, default=2, help="Levels of callees to include.")
    args = parser.parse_args()

    _require_dot()
    OUTPUT.mkdir(exist_ok=True)

    if args.target:
        output = OUTPUT / f"calls_around_{args.target.replace('::', '_').replace('.', '_')}.svg"
        _code2flow([PACKAGE], output, args.target, args.upstream, args.downstream)
        return

    # One node per folder or top-level module. A graph of every module is not drawn: at about
    # 100 nodes and 600 edges no arrow in it can be followed.
    print("imports between folders (pydeps):")
    _pydeps(OUTPUT / "subpackages.svg", "--max-module-depth", "2")

    print("function calls within each folder (code2flow):")
    # Not calls_core.svg: that name is the core/ subpackage's.
    _code2flow(sorted(PACKAGE.glob("*.py")), OUTPUT / "calls_top_level.svg")
    for subpackage in _subpackages(PACKAGE):
        if subpackage.name == "extensions":
            for extension in _subpackages(subpackage):
                _code2flow([extension], OUTPUT / f"calls_extensions_{extension.name}.svg")
        else:
            _code2flow([subpackage], OUTPUT / f"calls_{subpackage.name}.svg")

    print("function calls around each entry point, across the whole package (code2flow):")
    for name, target in ENTRY_POINTS.items():
        _code2flow([PACKAGE], OUTPUT / f"calls_around_{name}.svg", target, args.upstream, args.downstream)


if __name__ == "__main__":
    main()
