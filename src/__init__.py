# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/__init__.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Top of the package tree. After `pip install -e .` (File 7 setup.py), the
# on-disk layout `src/<sub>/<file>.py` is exposed as `pafa.<sub>.<file>`, so
# `from pafa.models.snn_detector import SNNDetector` works.
#
# ───────────────────────────────────────────────────────────────────────────
# Lazy subpackage imports — READ THIS BEFORE ADDING AN IMPORT
# ───────────────────────────────────────────────────────────────────────────
# None of the subpackages (`utils`, `models`, `baselines`, `attacks`,
# `cli_tools`, `hardware_simulation`) is imported at the top of this
# module. Each subpackage pulls in heavy third-party dependencies:
#
#   utils      → numpy, scipy, pandas, ripser (via math_utils)
#   models     → torch, snntorch, gudhi, py_ecc, pycryptodome
#   baselines  → torch, scikit-learn
#   attacks    → numpy, scipy, pandas
#   cli_tools  → argparse + the models/attacks modules
#   hardware_simulation → numpy only
#
# A reviewer who only wants to inspect `pafa.__version__` should not
# have to install every one of those to do so. Keeping `import pafa`
# dependency-free also lets `data/*.py` and `tests/test_math_utils.py`
# import the artifact in a minimal CPU-only environment.
#
# If a future version needs an eager import, add it behind a try/except
# ImportError and a clear warning message rather than as a bare import
# at module top.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

# ─── Version and metadata ─────────────────────────────────────────────────
# `__version__` must match:
#   • setup.py::ARTIFACT_VERSION       (File 7)
#   • Dockerfile::LABEL org.opencontainers.image.version   (File 6)
#   • configs/default_pafa_config.yaml::version            (File 9)
#   • README.md status banner                              (File 1)
#   • REPRODUCIBILITY_CHECKLIST.md §12 sign-off            (File 2)
# A mismatch is a release-blocking bug. A future
# `scripts/check_version_consistency.py` (optional; not in the 74-file
# plan) can enforce this automatically.
__version__ = "1.0.0"

__thesis_chapter__ = "Chapter 4 (PAFA-DoH)"
__citation_doi__ = "10.1016/j.neunet.2025.108343"

# ─── Public subpackage names ──────────────────────────────────────────────
# Listed so that `from pafa import *` (and IDE auto-completion) works
# without importing them. The strings are what `import pafa.utils`
# resolves to after `pip install -e .`; the `__all__` is a declaration,
# not a trigger.
__all__ = [
    "utils",
    "models",
    "baselines",
    "attacks",
    "cli_tools",
    "hardware_simulation",
    "__version__",
    "__thesis_chapter__",
    "__citation_doi__",
]


def __getattr__(name: str):
    """Lazy subpackage loader.

    Allows `import pafa; pafa.utils` and `from pafa import utils` to
    work without importing the subpackage at module-import time. The
    first attribute access triggers the import; subsequent accesses
    are cached in the module `__dict__` by Python's normal attribute
    lookup (so the second access is a dict lookup, not a re-import).

    Raises:
        AttributeError: for any name that is not one of the six
            declared subpackages. This preserves the normal Python
            contract (a missing attribute is an error, not a silent
            empty module).
    """
    if name in __all__:
        import importlib
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache in the module dict
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
