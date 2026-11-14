r"""Self-contained, executable example fixtures for the vision worker.

The ``vgi-lint`` strict profile (>= 0.26.0) *executes* every example query -- both
each function's per-function ``examples`` and the ``vgi.executable_examples`` JSON
-- against the attached worker. Examples therefore must be **self-contained** and
**re-runnable**: they cannot reference a ``photos`` table that the linter doesn't
have, nor pass a table function a ``(SELECT ...)`` subquery (DuckDB rejects that).

To stay self-contained we embed a tiny committed PNG (``test/sql/data/sample.png``,
the same fixture the SQL E2E suite uses) as a DuckDB ``BLOB`` literal. The classifier
labels it ``envelope`` -- a stable, decodable image -- so a BLOB-literal example
returns real rows without any filesystem or network dependency.

``SAMPLE_IMAGE_BLOB`` is the fixture's bytes as DuckDB hex escapes (``\\xNN`` is the
only escape form DuckDB accepts in a ``'...'::BLOB`` literal). It is a **raw** string
so the backslashes reach DuckDB verbatim instead of being decoded by Python.
"""

from __future__ import annotations

# The committed ``test/sql/data/sample.png`` fixture as a DuckDB hex BLOB literal.
# Decodable + deterministic, so classification examples return rows offline.
SAMPLE_IMAGE_BLOB = (
    r"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A\x00\x00\x00\x0D\x49\x48\x44\x52\x00\x00\x00\x40"
    r"\x00\x00\x00\x40\x08\x02\x00\x00\x00\x25\x0B\xE6\x89\x00\x00\x00\xE1\x49\x44\x41"
    r"\x54\x78\xDA\xED\xDA\xB1\x0D\x83\x30\x10\x85\x61\xB0\x18\x21\x43\x30\x4C\x26\xA0"
    r"\x66\x04\xC6\xC8\x08\x29\x52\x65\x82\x0C\xC3\x10\x19\x22\x05\x4D\xA4\x20\x41\x7C"
    r"\xCF\x1C\x27\xFF\x57\x02\xB2\xDE\x87\x8D\x31\x16\x6D\x3F\x3C\x9A\xC8\x95\x9A\xE0"
    r"\x05\x00\x00\x00\x00\x00\x00\x00\xF0\xAC\x6E\xCF\x45\xAF\xE1\xEE\x95\xEF\xFA\x1C"
    r"\x19\x42\x00\x00\x00\x00\xE0\xFD\x1E\xC8\xA8\x79\x7A\xFF\x1E\xEC\x6F\x97\x00\x80"
    r"\xD5\xE8\xDF\xA7\xB4\x8C\xEE\x98\xE8\xE5\x18\xE9\xE0\xF4\xD9\xD7\x97\x05\xE4\xA5"
    r"\x91\x18\x92\x57\x7A\x95\xA1\xFA\xF7\x80\xFD\x16\x1A\x5B\xA8\xBB\x07\x54\x33\x89"
    r"\xA5\x1D\xD6\x42\x00\x00\x00\x08\x0C\x50\xAD\x28\x2D\xED\x54\x3F\x84\xEC\x9D\x60"
    r"\x6C\x81\x87\xD8\x76\x0B\xED\x1D\x98\x1C\x9F\x42\xC9\x1C\x20\x1B\x42\xFF\xA6\x51"
    r"\xCD\x60\xCA\x8F\xFA\x25\xD3\xE6\xD2\xF2\xBC\xBB\x12\x9B\x8C\x18\xFB\x42\xE5\xB2"
    r"\x32\x8D\x02\x00\x00\x00\x80\xBE\x5A\x7E\xFA\x03\x00\x00\x00\x00\x00\x00\x00\xE4"
    r"\xD7\x07\x2B\x8B\x39\x99\xCE\x48\xA8\x86\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42"
    r"\x60\x82"
)

# The committed image fixture, as a path, for the path-overload examples. The
# linter launches the worker with the repo root as its working directory, so this
# relative path resolves to the same deterministic PNG the BLOB literal embeds.
SAMPLE_IMAGE_PATH = "test/sql/data/sample.png"

# ---------------------------------------------------------------------------
# Composed examples: queries that combine two of the worker's objects. Each SQL is
# defined once here and referenced from the schema-level list and from the example
# list of the object that drives the query (within its first five examples, which is
# all the vgi-lint agent simulator shows). They use the committed fixture path, so
# they run offline and deterministically.
#
# ImageNet's class index is ordered by synset: idx 0-397 are animals and
# idx 151-268 are the 118 dog breeds. ``classify`` returns labels only, so joining
# ``image_classes`` on ``label`` recovers each prediction's idx. Two labels are
# duplicated in the ImageNet list ("crane" at 134/517, "maillot" at 638/639); a
# label join fans those out, but neither falls inside the dog-breed range.
# ---------------------------------------------------------------------------

COMPOSED_CLASSIFY_WITH_INDEX = {
    "description": (
        "Join the top-10 predictions for an image file to the image_classes lookup table to "
        "recover each predicted label's ImageNet class index (idx), most-confident first."
    ),
    "sql": (
        "SELECT c.idx, p.label, round(p.confidence, 4) AS confidence "
        f"FROM vision.main.classify('{SAMPLE_IMAGE_PATH}', 10) AS p "
        "JOIN vision.main.image_classes AS c USING (label) "
        "ORDER BY p.confidence DESC"
    ),
}

COMPOSED_DOG_PROBABILITY = {
    "description": (
        "Roll the full 1000-class distribution for an image file up to one coarse category: "
        "total probability on the 118 dog-breed classes (ImageNet idx 151-268, found via "
        "image_classes) and the most likely breed among them."
    ),
    "sql": (
        "SELECT count(*) AS n_breeds, round(sum(p.confidence), 4) AS dog_probability, "
        "arg_max(p.label, p.confidence) AS likeliest_breed "
        f"FROM vision.main.classify('{SAMPLE_IMAGE_PATH}', 1000) AS p "
        "JOIN vision.main.image_classes AS c USING (label) "
        "WHERE c.idx BETWEEN 151 AND 268"
    ),
}

COMPOSED_TOP_LABEL_WITH_INDEX = {
    "description": (
        "Tag a list of image files with top_label and look each label up in image_classes for "
        "its class index; an unreadable file gets a NULL label, so the LEFT JOIN keeps it with "
        "NULL idx."
    ),
    "sql": (
        "SELECT f.path, c.idx, c.label "
        f"FROM (VALUES ('{SAMPLE_IMAGE_PATH}'), ('test/sql/data/missing.png')) AS f(path) "
        "LEFT JOIN vision.main.image_classes AS c ON c.label = vision.main.top_label(f.path) "
        "ORDER BY f.path"
    ),
}
