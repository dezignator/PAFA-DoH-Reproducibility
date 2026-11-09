# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# src/utils/logger.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343
# License:        MIT
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# Three things that every entry point needs before doing any real work:
#
#   1. A `get_logger(name)` that writes both a human-readable `.log`
#      file and a machine-readable `.json` sidecar to `outputs/logs/`,
#      matching the `logging.format: "human+json"` setting in
#      `configs/default_pafa_config.yaml` (File 9).
#
#   2. A `ForbiddenObservablesGuard` that reads the four forbidden
#      observables from the master config
#      (`threat_model.forbidden_observables`) and refuses to let any
#      DataFrame that contains one of them pass. This is the code
#      enforcement of §1.7.2 and §4.13 — the privacy claims of the
#      chapter are conditional on this guard being active.
#
#   3. A `resolve_artifact_root()` that implements the three-step
#      resolution contract documented in File 7's footer and File 9's
#      header: $PAFA_ROOT → repo tree → <sys.prefix>/share/pafa-doh/.
#
# `data/data_utils.py` (File 10a) has its own `resolve_artifact_root`
# (identical contract). The two are kept in sync by convention; a
# future refactor could have `data_utils` import from here, but the
# data layer is deliberately dependency-free of `src/` so that the
# preprocessors can run in a minimal environment.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        f"PyYAML is required for src.utils.logger ({exc.name} missing). "
        "Install with `pip install -r requirements.txt`."
    ) from exc


# ─── Artifact-root resolver (same contract as File 10a) ──────────────────

def resolve_artifact_root(env_var: str = "PAFA_ROOT") -> Path:
    """Return the artifact root directory.

    Resolution order (see File 7 footer and File 9 header):

    1. ``$PAFA_ROOT`` if set and a directory.
    2. The nearest ancestor of this file that contains both
       ``requirements.txt`` and ``configs/default_pafa_config.yaml``.
    3. ``<sys.prefix>/share/pafa-doh`` if it exists (wheel installs).

    Raises:
        FileNotFoundError: if none resolves. Message names every path
            tried so a reviewer can fix the environment.
    """
    tried: list[Path] = []

    env_val = os.environ.get(env_var)
    if env_val:
        candidate = Path(env_val).expanduser().resolve()
        tried.append(candidate)
        if candidate.is_dir():
            return candidate

    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        tried.append(parent)
        if (
            (parent / "requirements.txt").is_file()
            and (parent / "configs" / "default_pafa_config.yaml").is_file()
        ):
            return parent

    wheel_candidate = Path(sys.prefix) / "share" / "pafa-doh"
    tried.append(wheel_candidate)
    if wheel_candidate.is_dir():
        return wheel_candidate

    raise FileNotFoundError(
        "Could not resolve the PAFA-DoH artifact root. Tried: "
        + ", ".join(str(p) for p in tried)
        + ". Set $PAFA_ROOT to the repository root, or run the artifact "
          "from a checkout that contains requirements.txt and "
          "configs/default_pafa_config.yaml."
    )


# ─── Forbidden-observables guard ─────────────────────────────────────────

class ForbiddenObservablesError(RuntimeError):
    """Raised when a component tries to read a forbidden observable."""


class ForbiddenObservablesGuard:
    """Enforce the four forbidden observables at every DataFrame boundary.

    The four observables are defined in `configs/default_pafa_config.yaml`
    under `threat_model.forbidden_observables`:

        query_payload_content
        response_payload_content
        application_layer_semantics
        database_information

    A component that reads one of these is violating the privacy claims
    of Chapter 4 (§1.7.2, §4.13, §4.5). The guard is fail-closed: any
    DataFrame or column list that contains one of these — in any casing,
    with any underscore / hyphen / space combination — raises
    `ForbiddenObservablesError`.

    Usage:

        guard = ForbiddenObservablesGuard.from_config(cfg_path)
        guard.check_dataframe(df, source="preprocess_zenodo")
        guard.check_columns(["flow_id", "query_payload"], source="custom loader")
    """

    def __init__(self, forbidden: Iterable[str]):
        # Normalize the forbidden names once, at construction, so the
        # per-call check is a set lookup rather than a re-normalization.
        normalized = {"".join(ch for ch in str(f).lower() if ch.isalnum())
                      for f in forbidden}
        if len(normalized) < 1:
            raise ValueError(
                "ForbiddenObservablesGuard requires at least one forbidden "
                "observable. An empty forbidden list would make the guard a "
                "no-op, which is worse than not having it."
            )
        self._forbidden_raw: tuple[str, ...] = tuple(forbidden)
        self._forbidden_norm: frozenset[str] = frozenset(normalized)

    @classmethod
    def from_config(cls, config_path: Path | str) -> "ForbiddenObservablesGuard":
        """Build a guard from the master config's threat_model block.

        Raises:
            FileNotFoundError: if the config is missing.
            yaml.YAMLError: if the config is malformed.
            KeyError: if `threat_model.forbidden_observables` is missing
                from the config. The guard is fail-closed: a config
                that does not declare the forbidden list is treated as
                a configuration error, not as "allow everything".
        """
        cfg_path = Path(config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Master config not found at {cfg_path}. The "
                f"ForbiddenObservablesGuard needs "
                f"threat_model.forbidden_observables from this file."
            )
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        try:
            forbidden = cfg["threat_model"]["forbidden_observables"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "Master config is missing "
                "threat_model.forbidden_observables. The guard is "
                "fail-closed; add the four forbidden observables to the "
                "config before running any component."
            ) from exc
        return cls(forbidden)

    @property
    def forbidden(self) -> tuple[str, ...]:
        """The raw forbidden-observable names, as declared in the config."""
        return self._forbidden_raw

    # ── Checks ──────────────────────────────────────────────────────────

    def _normalize(self, name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    def check_columns(self, columns: Iterable[str], *, source: str = "<unknown>") -> None:
        """Raise if any column name normalizes to a forbidden observable."""
        hits: list[tuple[str, str]] = []
        for col in columns:
            norm = self._normalize(col)
            if norm in self._forbidden_norm:
                hits.append((str(col), norm))
        if hits:
            raise ForbiddenObservablesError(
                f"Forbidden observable(s) in columns from {source!r}: "
                f"{sorted(hits)}. The four forbidden observables are: "
                f"{list(self._forbidden_raw)}. See §1.7.2 / §4.13 of the "
                f"thesis and "
                f"configs/default_pafa_config.yaml::threat_model."
                f"forbidden_observables."
            )

    def check_dataframe(self, df: Any, *, source: str = "<unknown>") -> None:
        """Raise if a pandas DataFrame has a forbidden column.

        Accepts any object with a `.columns` attribute (pandas
        DataFrame, Dask DataFrame, or a duck-typed equivalent) so the
        same guard works on the Dask partitions in
        `src/utils/math_utils.py`'s pipelines.
        """
        try:
            cols = list(df.columns)
        except AttributeError as exc:
            raise TypeError(
                f"ForbiddenObservablesGuard.check_dataframe requires an "
                f"object with a `.columns` attribute; got "
                f"{type(df).__name__}."
            ) from exc
        self.check_columns(cols, source=source)

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return (
            f"ForbiddenObservablesGuard(n={len(self._forbidden_raw)}, "
            f"raw={list(self._forbidden_raw)!r})"
        )


# ─── Logger ──────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """One-line JSON formatter for the .json sidecar log.

    Emits exactly the fields that `verification_script.py` (File 20i)
    reads: `ts`, `level`, `logger`, `msg`, plus any `extra` fields
    the caller passed via `logger.info(msg, extra={"k": v})`.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Pull caller-supplied extras off the record. Standard logging
        # attributes are not "extra"; anything else is.
        standard = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "asctime",
        }
        for k, v in record.__dict__.items():
            if k not in standard and not k.startswith("_"):
                payload[k] = v
        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(
    name: str,
    *,
    log_dir: Path | str | None = None,
    level: int | str = logging.INFO,
    json_sidecar: bool = True,
    console: bool = True,
) -> logging.Logger:
    """Return a logger that writes to `log_dir`.

    Creates two handlers:

      • `<log_dir>/<name>.<timestamp>.log`  — human-readable (FileHandler)
      • `<log_dir>/<name>.<timestamp>.json` — one-line JSON per record
        (FileHandler + _JsonFormatter), so downstream tools can grep
        structured fields without parsing the human log.

    If `console` is True (default), also attaches a StreamHandler to
    stderr with a compact prefix so a reviewer who runs the pipeline
    sees progress on their terminal.

    Args:
        name: Logger name, usually `__name__` of the calling module.
        log_dir: Where to write the two files. If None, resolves to
            `<artifact_root>/outputs/logs/` via `resolve_artifact_root`.
            A reviewer who runs a component from outside a checkout can
            set `$PAFA_ROOT` and get the correct path.
        level: Logging level. Accepts an int (logging.INFO) or a string
            ("INFO", "DEBUG"). Defaults to INFO.
        json_sidecar: If True (default), write the .json file. Set False
            for short-lived scripts that do not need structured logs.
        console: If True (default), also log to stderr.

    Returns:
        A configured `logging.Logger`. It is safe to call twice; the
        handlers are tagged with a sentinel attribute so the second call
        does not duplicate them.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if log_dir is None:
        log_dir = resolve_artifact_root() / "outputs" / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # we manage our own handlers

    # If we've already attached our handlers, do not attach again.
    if getattr(logger, "_pafa_handlers_attached", False):
        return logger
    logger._pafa_handlers_attached = True  # type: ignore[attr-defined]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Human-readable file handler
    human_path = log_dir / f"{name.replace('.', '_')}.{ts}.log"
    fh = logging.FileHandler(human_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(fh)

    # JSON sidecar
    if json_sidecar:
        json_path = log_dir / f"{name.replace('.', '_')}.{ts}.json"
        jh = logging.FileHandler(json_path, encoding="utf-8")
        jh.setLevel(level)
        jh.setFormatter(_JsonFormatter())
        logger.addHandler(jh)

    # Console
    if console:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter(
                "%(levelname)-8s  %(name)s  %(message)s",
            )
        )
        logger.addHandler(ch)

    return logger


# ─── Run-metadata helper ─────────────────────────────────────────────────

def log_run_metadata(logger: logging.Logger, config_path: Path | str) -> dict[str, Any]:
    """Read the master config, compute a SHA-256 of it, and log the pair.

    Used at the top of every entry point so that a run log is
    self-contained: a reviewer can look at a result and know exactly
    which config produced it, without a separate paper trail.

    Returns the dict that was logged, for callers that want to embed
    it in their own structured output.
    """
    import hashlib
    cfg_path = Path(config_path)
    cfg_bytes = cfg_path.read_bytes()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
    try:
        cfg = yaml.safe_load(cfg_bytes.decode("utf-8"))
    except yaml.YAMLError as exc:
        logger.warning("Could not parse config for metadata: %s", exc)
        cfg = {}
    metadata = {
        "config_path": str(cfg_path),
        "config_sha256": cfg_hash,
        "config_version": cfg.get("version"),
        "artifact_version": cfg.get("version"),
        "seed": cfg.get("seed"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_monotonic": time.monotonic(),
    }
    logger.info("Run metadata", extra=metadata)
    return metadata
