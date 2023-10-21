"""Generate the code reference pages and navigation."""

from pathlib import Path

import os

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "transformers_embedder"
DOC_DIR = ROOT_DIR / "references"

for path in sorted(Path("transformers_embedder").glob("**/*.py")):
    module_path = path.with_suffix("")
    doc_path = path.with_suffix(".md").name
    full_doc_path = DOC_DIR / doc_path
    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        # doc_path = doc_path.with_name("index.md")
        # full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open(DOC_DIR / "main.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
