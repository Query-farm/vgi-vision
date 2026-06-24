"""Shared per-object discovery/description metadata for the ``vgi-lint`` strict profile.

The strict profile (vgi-lint-check >= 0.26.0) expects, on **every** function and
table object, a consistent set of discovery/description tags:

* ``vgi.title`` (VGI124) -- a human-friendly display name (must not normalize-equal
  the machine name, so it always carries an extra descriptive word).
* ``vgi.doc_llm`` (VGI112) -- a Markdown narrative aimed at an LLM/agent audience:
  what the object does, when to reach for it, its inputs/outputs and edge cases.
* ``vgi.doc_md`` (VGI113) -- a Markdown narrative for human docs: overview, usage,
  and notes. Deliberately *distinct* prose from ``vgi.doc_llm`` (identical values
  are flagged as duplication).
* ``vgi.keywords`` (VGI126) -- comma-separated search terms / synonyms.
* ``vgi.source_url`` (VGI128) -- a link to the file that implements the object.

:func:`source_url` builds the canonical GitHub blob URL for a source file so every
object points at exactly where it is implemented; :func:`object_tags` assembles the
five standard tags into the ``dict`` shape the VGI ``Meta.tags`` API expects.
"""

from __future__ import annotations

# Base GitHub blob URL for source files in this repo (pinned to ``main``).
_SOURCE_BASE = "https://github.com/Query-farm/vgi-vision/blob/main"


def source_url(relative_path: str) -> str:
    """Build the implementation ``vgi.source_url`` for a file in the repo.

    ``relative_path`` is the file's path relative to the repo root, e.g.
    ``source_url("vgi_vision/scalars.py")``.
    """
    return f"{_SOURCE_BASE}/{relative_path}"


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: str,
    relative_path: str,
) -> dict[str, str]:
    """Build the five standard per-object discovery/description tags.

    ``relative_path`` is the implementing file relative to the repo root.
    Returns the ``dict`` that a function's ``Meta.tags`` expects; callers merge in
    any object-specific tags (``vgi.result_columns_md``, ``vgi.executable_examples``).
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords,
        "vgi.source_url": source_url(relative_path),
    }
