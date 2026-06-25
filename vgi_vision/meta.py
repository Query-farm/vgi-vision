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
* ``vgi.keywords`` (VGI126/VGI138) -- a JSON array of search-term / synonym strings,
  e.g. ``["classify", "predict"]`` (a bare comma-separated string is rejected).

:func:`object_tags` assembles the standard discovery/description tags into the
``dict`` shape the VGI ``Meta.tags`` API expects. Per-object ``vgi.source_url`` is
intentionally *not* emitted: the implementation link is carried once on the catalog
object (VGI139 flags redundant per-object ``source_url``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence


def keywords_json(keywords: Sequence[str]) -> str:
    """Serialize search keywords to the ``vgi.keywords`` JSON-array string.

    VGI138 requires ``vgi.keywords`` to be a JSON array of strings (e.g.
    ``["classify","predict"]``), not a comma-separated string. ``keywords`` is the
    ordered list of search terms / synonyms for the object.
    """
    return json.dumps(list(keywords))


def object_tags(
    *,
    title: str,
    doc_llm: str,
    doc_md: str,
    keywords: Sequence[str],
) -> dict[str, str]:
    """Build the standard per-object discovery/description tags.

    ``keywords`` is the ordered list of search terms / synonyms for the object; it
    is serialized to the ``vgi.keywords`` JSON-array string the linter expects.
    Returns the ``dict`` that a function's ``Meta.tags`` expects; callers merge in
    any object-specific tags (``vgi.result_columns_md``, ``vgi.executable_examples``).
    """
    return {
        "vgi.title": title,
        "vgi.doc_llm": doc_llm,
        "vgi.doc_md": doc_md,
        "vgi.keywords": keywords_json(keywords),
    }
