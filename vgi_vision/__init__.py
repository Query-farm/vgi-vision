"""vgi-vision: ImageNet image classification on image blobs as DuckDB SQL functions.

The package splits into:

* :mod:`vgi_vision.model` -- pure inference logic: model download/cache, warm-up,
  preprocessing, and a small ``classify_image(bytes, top_k)`` API over raw image
  bytes. No VGI/Arrow types leak in here, so it is unit-testable in isolation.
* :mod:`vgi_vision.scalars` -- per-row scalar functions (``top_label``).
* :mod:`vgi_vision.tables` -- set-returning table functions (``classify``,
  ``image_classes``).

The worker itself -- the ``vision`` catalog, :class:`~vgi_vision.worker.VisionWorker`,
and ``main()`` -- lives in :mod:`vgi_vision.worker` (wheel-importable, and the
``vgi-vision-worker`` console script); ``vision_worker.py`` at the repo root is a thin
PEP 723 shim re-exporting it.
"""
