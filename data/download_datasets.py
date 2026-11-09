#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════
# download_datasets.py — PAFA-DoH Reproducibility Artifact
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to:
#   README.md, REPRODUCIBILITY_CHECKLIST.md, LICENSE, requirements.txt,
#   environment.yml, Dockerfile, setup.py, .gitignore   (Files 1–8)
#   configs/default_pafa_config.yaml                    (File 9, master config)
#   data/data_utils.py                                  (File 10a, shared
#                                                        schema + checksum +
#                                                        root resolution)
#
# Thesis chapter: Chapter 4 (PAFA-DoH)
# Citation DOI:   10.1016/j.neunet.2025.108343   (Neural Networks, April 2026)
# License:        MIT (this file, as part of the artifact's source code)
# Artifact version: 1.0.0
#
# ───────────────────────────────────────────────────────────────────────────
# Purpose
# ───────────────────────────────────────────────────────────────────────────
# Downloads the two public datasets used in Chapter 4, generates the toy
# dataset used by the README §3 toy demo, and verifies the SHA-256 of
# whatever lands on disk against `data/CHECKSUMS.txt` (File 10j).
#
# The two datasets are:
#
#   Zenodo DoH corpus   — DOI 10.5281/zenodo.5957677, ≈ 32 GB
#                         (§4.24.1, §4.25.1, Tables 14–17)
#                         Redistributed under CC-BY 4.0 upstream; NOT
#                         redistributed by this repository (LICENSE §2).
#
#   CIC-DoH2023         — https://www.unb.ca/cic/datasets/dohbrw-2023.html,
#                         ≈ 2.5 GB (§4.24.1). Used only for the 47-column
#                         schema-parity check on the preprocessor, not
#                         as a training source. CC-BY 4.0 upstream; not
#                         redistributed.
#
# The toy dataset is a 1,000 + 1,000 row synthetic derivative generated
# by `data/make_toy_dataset.py` (File 10f). It is CC-BY 4.0, header-only
# in git, and regenerated in-place. Its accuracy (~85–90 %) does NOT
# reproduce the Chapter 4 headline numbers (see README §4).
#
# ───────────────────────────────────────────────────────────────────────────
# Design notes
# ───────────────────────────────────────────────────────────────────────────
# • DOIs and URLs are read from `configs/default_pafa_config.yaml`
#   (`datasets.zenodo.doi`, `datasets.zenodo.url`,
#   `datasets.cic_doh2023.url`). No dataset identifier is hard-coded in
#   this file; the master config is the single source of truth.
#
# • The Zenodo download uses the Zenodo REST API
#   (`https://zenodo.org/api/records/{id}`) to list the files in the
#   record and to obtain their per-file size and direct download URL.
#   This avoids scraping the HTML landing page and is robust to
#   Zenodo's UI changes. The record ID is parsed from the DOI
#   (`10.5281/zenodo.5957677` → `5957677`); the DOI itself is read
#   from the config.
#
# • CIC-DoH2023 is distributed as individual CSV files behind a landing
#   page that requires a browser to navigate in the general case.
#   This script downloads the landing page and prints a clear manual-
#   extraction instruction. The checksums are the source of truth; a
#   user who has obtained the CSVs another way still gets full
#   integrity verification by placing them in the target directory.
#
# • Resume is on by default (`--resume`), implemented with HTTP `Range`
#   and an explicit check that the server honoured the range with a
#   `206 Partial Content`. If the server replies `200 OK` to a range
#   request, the script logs that the range was ignored and restarts
#   from scratch rather than producing a corrupt file.
#
# • Network safety: every URL is required to be HTTPS, including the
#   *final* URL after redirects (`require_https(r.url)`). A redirect
#   to plain HTTP is a hard failure, not a warning.
#
# • The script is idempotent: re-running it after a successful download
#   is a no-op if the file's size matches the expected size and the
#   checksum manifest is satisfied. `--force` is required to overwrite
#   an existing file; `--no-resume` disables the HTTP Range path (useful
#   for debugging a corrupt partial download).
#
# • A `--dry-run` mode prints what would be downloaded without opening
#   any network connection. Use it to confirm the DOIs / URLs before a
#   multi-hour full-mode download.
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

# ─── External packages (pinned in requirements.txt, File 4) ────────────────
try:
    import requests
    import yaml
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover — environment misconfiguration
    print(
        f"FATAL: missing required dependency: {exc.name}. "
        "Install the pinned dependencies first:\n"
        "    pip install -r requirements.txt\n"
        "or, for a conda environment:\n"
        "    conda env create -f environment.yml",
        file=sys.stderr,
    )
    raise


# ─── Local import — support both `python data/download_datasets.py` and
#     `python -m data.download_datasets` from the repo root ─────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # package-qualified (works when imported as `data.download_datasets`)
    from data.data_utils import (
        compute_sha256,
        resolve_artifact_root,
        verify_checksums,
    )
except ImportError:  # script-qualified (works with `python data/download_datasets.py`)
    from data_utils import (  # type: ignore[no-redef]
        compute_sha256,
        resolve_artifact_root,
        verify_checksums,
    )


# ─── Constants ─────────────────────────────────────────────────────────────

ARTIFACT_VERSION = "1.0.0"
THESIS_CHAPTER = "Chapter 4 (PAFA-DoH)"
CITATION_DOI = "10.1016/j.neunet.2025.108343"

DEFAULT_TIMEOUT_S: int = 60
DEFAULT_CHUNK_SIZE: int = 1 << 20  # 1 MiB
DEFAULT_USER_AGENT: str = (
    "pafa-doh-reproducibility/1.0.0 "
    "(+https://github.com/<your-org>/PAFA-DoH-Reproducibility)"
)


# ─── Logging ───────────────────────────────────────────────────────────────

log = logging.getLogger("data.download_datasets")


def configure_logging(verbose: bool) -> None:
    """Configure the module logger.

    Uses a single `basicConfig` call so that importing this module from
    another script (e.g. `train_pafa.py`) does not install a second
    handler on the root logger. The `human+json` format from the master
    config (File 9, `logging.format`) is implemented by
    `src/utils/logger.py` (File 11d) in the full pipeline; this script
    uses the simpler human-readable form because it runs before the
    pipeline is entered.
    """
    level = logging.DEBUG if verbose else logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        log.setLevel(level)


# ─── Master config loader ─────────────────────────────────────────────────

def load_config(root: Path) -> dict[str, Any]:
    """Read `configs/default_pafa_config.yaml` from the artifact root.

    Raises:
        FileNotFoundError: if the master config is missing. A missing
            master config is a broken checkout, not a missing download;
            the error message names the path so the user can `git pull`
            or re-clone.
        yaml.YAMLError: if the file is present but malformed. The
            caller is expected to catch and print this.
    """
    cfg_path = root / "configs" / "default_pafa_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Master config not found at {cfg_path}. "
            "This file is committed to the repository; a missing copy "
            "usually means the checkout is incomplete. Re-clone, or "
            "restore it with: git checkout -- configs/default_pafa_config.yaml"
        )
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


# ─── URL safety ───────────────────────────────────────────────────────────

def require_https(url: str) -> None:
    """Abort the program if `url` is not an HTTPS URL.

    Called before every network request *and* on the final URL after
    redirects (`require_https(r.url)`). The artifact downloads only
    from HTTPS sources; a redirect to plain HTTP is treated as a hard
    failure, not a warning, because a plain-HTTP download would defeat
    the integrity guarantees the checksums exist to provide.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(
            f"Refusing to use a non-HTTPS URL: {url!r}. "
            "All dataset downloads in this artifact must use HTTPS. "
            "If a redirect from a trusted HTTPS source lands on plain "
            "HTTP, treat the source as untrusted and report the issue."
        )


# ─── Core single-file downloader ──────────────────────────────────────────

def download_file(
    url: str,
    dest: Path,
    *,
    resume: bool = True,
    force: bool = False,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    expected_size: int | None = None,
    session: requests.Session | None = None,
) -> Path:
    """Download `url` to `dest` and return `dest`.

    Args:
        url: The HTTPS URL to fetch. Checked with :func:`require_https`
            before the request and again after redirects.
        dest: Destination path. Parent directories are created.
        resume: If True (default), send an HTTP `Range` header if a
            partial file already exists at `dest`. If the server replies
            `200 OK` instead of `206 Partial Content`, the script logs
            that the range was ignored and restarts from byte 0 (a
            corrupt partial file is worse than a re-download).
        force: If True, delete an existing `dest` before downloading.
            If False and `dest` exists, the function either skips (if
            the size matches `expected_size`) or raises.
        timeout_s: Per-request timeout in seconds. Applies to connect
            and to read between chunks.
        expected_size: If provided and the existing file already has
            this exact size, skip the download. Used by the Zenodo
            path, where the API publishes per-file sizes.
        session: Optional pre-configured `requests.Session`. If None,
            a fresh session is created (and not closed explicitly; the
            garbage collector will release it).

    Raises:
        FileExistsError: if `dest` exists, `force` is False, and the
            size does not match `expected_size`.
        ValueError: if `url` (or the final URL after redirects) is not
            HTTPS.
        RuntimeError: on any non-2xx HTTP status that is not a
            successful resume, or on a redirected-to-HTTP response.
        requests.RequestException: on network failures.
    """
    require_https(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Existing-file handling
    if dest.exists():
        if force:
            log.info("Removing existing %s (--force)", dest)
            dest.unlink()
        elif expected_size is not None and dest.stat().st_size == expected_size:
            log.info(
                "Already present and size-matched: %s (%.2f MB)",
                dest,
                expected_size / 1e6,
            )
            return dest
        elif not resume:
            raise FileExistsError(
                f"{dest} exists. Pass --force to overwrite, or "
                "--resume to continue a partial download."
            )
        elif expected_size is None:
            # No expected size to compare against: be conservative and
            # refuse rather than silently skip a partial download.
            raise FileExistsError(
                f"{dest} exists and no expected size was supplied. "
                "Pass --force to overwrite, or --resume to continue."
            )

    existing_bytes = dest.stat().st_size if dest.exists() and resume else 0

    headers: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    sess = session or requests.Session()
    with sess.get(
        url,
        headers=headers,
        stream=True,
        timeout=timeout_s,
        allow_redirects=True,
    ) as r:
        # The final URL after redirects must still be HTTPS.
        require_https(r.url)

        if r.status_code == 206 and existing_bytes > 0:
            mode = "ab"
            total = existing_bytes + int(r.headers.get("Content-Length", 0) or 0)
            log.info("Resuming %s from %d bytes", dest, existing_bytes)
        elif r.status_code == 200:
            mode = "wb"
            total = int(r.headers.get("Content-Length", 0) or 0) or None
            if existing_bytes > 0:
                log.warning(
                    "Server ignored the Range header for %s; restarting from byte 0",
                    dest,
                )
        else:
            raise RuntimeError(
                f"HTTP {r.status_code} for {url}. "
                f"Response headers: {dict(r.headers)}"
            )

        with dest.open(mode) as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=DEFAULT_CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))

    if expected_size is not None:
        observed = dest.stat().st_size
        if observed != expected_size:
            raise RuntimeError(
                f"Size mismatch after download of {dest}: "
                f"expected {expected_size} bytes, got {observed}. "
                "The file will not be used; delete it and re-run."
            )
    return dest


# ─── Zenodo ────────────────────────────────────────────────────────────────

def _parse_zenodo_record_id(doi: str) -> str:
    """Extract the numeric record ID from a Zenodo DOI.

    Zenodo DOIs take the form `10.5281/zenodo.<id>` (optionally with a
    trailing slash). This helper is deliberately strict: it raises if
    the input is not a Zenodo DOI, so that a misconfigured master config
    fails loudly at parse time instead of a cryptic API error later.
    """
    if "zenodo." not in doi:
        raise ValueError(
            f"{doi!r} is not a Zenodo DOI. "
            "Expected a string of the form '10.5281/zenodo.<id>'."
        )
    rid = doi.rsplit("zenodo.", 1)[1].strip("/")
    if not rid.isdigit():
        raise ValueError(
            f"Zenodo record ID {rid!r} is not numeric. "
            "Refusing to construct an API URL from a non-numeric ID."
        )
    return rid


def download_zenodo(
    cfg: dict[str, Any],
    root: Path,
    *,
    force: bool,
    dry_run: bool,
    resume: bool,
    timeout_s: int,
) -> Path:
    """Download every file in the pinned Zenodo record to `data/raw/zenodo/`.

    The record is resolved via the Zenodo REST API at
    `https://zenodo.org/api/records/{id}`. The API returns JSON whose
    `files` array contains, for each file:
      * `key`   — filename
      * `size`  — bytes
      * `links.self` / `links.download` — direct HTTPS URL

    Returns the destination directory.
    """
    zen = cfg["datasets"]["zenodo"]
    doi = zen["doi"]
    record_id = _parse_zenodo_record_id(doi)

    raw_root = root / cfg["paths"]["raw_datasets"]
    dest_dir = raw_root / "zenodo"
    api_url = f"https://zenodo.org/api/records/{record_id}"

    log.info("Zenodo DOI:        %s", doi)
    log.info("Zenodo record ID:  %s", record_id)
    log.info("Zenodo API:        %s", api_url)
    log.info("Destination:       %s", dest_dir)

    if dry_run:
        log.info(
            "DRY RUN — would GET %s and download each file in record['files']",
            api_url,
        )
        return dest_dir

    require_https(api_url)
    r = requests.get(
        api_url,
        timeout=timeout_s,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Zenodo API returned HTTP {r.status_code} for {api_url}. "
            "The record may have been renamed, or Zenodo may be rate-"
            "limiting the request. Re-try after a short pause; if the "
            "problem persists, check the DOI in a browser."
        )
    record = r.json()
    files = record.get("files", []) or []
    if not files:
        raise RuntimeError(
            f"Zenodo record {record_id} has no files in its `files` array. "
            "The record may be empty, restricted, or the DOI may point to "
            "a parent record rather than the file-containing record."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        key = f["key"]
        size = f.get("size")
        url = f["links"].get("self") or f["links"].get("download")
        if not url:
            raise RuntimeError(
                f"Zenodo record {record_id} file {key!r} has no download link"
            )
        log.info("Zenodo file: %s (%s bytes)", key, size)
        download_file(
            url,
            dest_dir / key,
            resume=resume,
            force=force,
            timeout_s=timeout_s,
            expected_size=size,
        )
    return dest_dir


# ─── CIC-DoH2023 ───────────────────────────────────────────────────────────

def download_cic_doh2023(
    cfg: dict[str, Any],
    root: Path,
    *,
    force: bool,
    dry_run: bool,
    resume: bool,
    timeout_s: int,
) -> Path:
    """Download the CIC-DoH2023 landing page to `data/raw/cic_doh2023/`.

    CIC-DoH2023 is distributed as individual CSV files behind a landing
    page that, in the general case, requires a browser to navigate. The
    script downloads the landing page and prints a clear manual-
    extraction instruction. Integrity is still enforced: the checksum
    verification step (`verify_all`) runs against whatever CSVs the user
    places in the directory.

    Returns the destination directory.
    """
    cic = cfg["datasets"]["cic_doh2023"]
    url = cic["url"]
    raw_root = root / cfg["paths"]["raw_datasets"]
    dest_dir = raw_root / "cic_doh2023"

    log.info("CIC-DoH2023 URL:   %s", url)
    log.info("Destination:       %s", dest_dir)

    if dry_run:
        log.info("DRY RUN — would GET %s", url)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    landing = dest_dir / "_landing_page.html"
    download_file(
        url,
        landing,
        resume=resume,
        force=force,
        timeout_s=timeout_s,
    )

    log.warning(
        "CIC-DoH2023 is distributed as individual CSV files behind a "
        "landing page. The landing page has been saved to %s; open it "
        "in a browser to obtain the CSV files, and place them directly "
        "in %s before running the verification step. Checksums are in "
        "data/CHECKSUMS.txt.",
        landing,
        dest_dir,
    )
    return dest_dir


# ─── Toy dataset ───────────────────────────────────────────────────────────

def generate_toy_dataset(root: Path) -> Path:
    """Run `data/make_toy_dataset.py --mode synthetic --seed 42`.

    This wrapper does not implement its own overwrite policy: the
    generator (File 10f) decides whether to overwrite existing toy
    CSVs, and its policy is documented in that file. The wrapper's
    only job is to invoke the generator with the pinned seed and to
    surface a clean error if the generator is missing or returns
    non-zero.

    Returns the `data/toy_dataset/` directory.
    """
    script = root / "data" / "make_toy_dataset.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"Toy generator not found at {script}. "
            "File 10f of the 74-file plan must be uploaded before the "
            "toy sub-command can run."
        )
    cmd = [sys.executable, str(script), "--mode", "synthetic", "--seed", "42"]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(root), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Toy generator exited with code {result.returncode}. "
            "Re-run with --verbose for the generator's own error message."
        )
    return root / "data" / "toy_dataset"


# ─── Verification ──────────────────────────────────────────────────────────

def verify_all(root: Path, cfg: dict[str, Any]) -> None:
    """Verify every file in `data/CHECKSUMS.txt` against `data/raw/`.

    Delegates to :func:`data.data_utils.verify_checksums` (File 10a),
    which reads the manifest and raises `ValueError` on the first
    mismatch. A missing manifest is a warning, not an error: a user
    who only ran the toy sub-command has no raw data to verify, and a
    user who is doing a full-mode download before the manifest has
    been committed (rare, but possible during artifact development)
    should still get the rest of the pipeline.
    """
    manifest = root / "data" / "CHECKSUMS.txt"
    if not manifest.is_file():
        log.warning(
            "No CHECKSUMS.txt at %s; skipping integrity verification. "
            "In a complete checkout this file is committed and the "
            "verification runs automatically.",
            manifest,
        )
        return
    raw_dir = root / cfg["paths"]["raw_datasets"]
    if not raw_dir.is_dir():
        log.warning(
            "No raw-data directory at %s; skipping integrity verification.",
            raw_dir,
        )
        return
    log.info("Verifying checksums against %s", manifest)
    results = verify_checksums(manifest, raw_dir)
    log.info("Checksum verification passed for %d file(s).", len(results))
    for rel, digest in sorted(results.items()):
        log.debug("  %s  %s", digest[:16] + "…", rel)


# ─── argparse ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_datasets.py",
        description=(
            "Download and verify the two datasets used by Chapter 4 "
            "(PAFA-DoH): the Zenodo DoH corpus (DOI 10.5281/zenodo.5957677) "
            "and CIC-DoH2023. Also generates the toy dataset used by the "
            "toy demo in README.md §3. All DOIs / URLs are read from "
            "configs/default_pafa_config.yaml; nothing is hard-coded here."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/download_datasets.py zenodo --dry-run\n"
            "  python data/download_datasets.py zenodo\n"
            "  python data/download_datasets.py cic_doh2023\n"
            "  python data/download_datasets.py all\n"
            "  python data/download_datasets.py toy\n"
            "  python data/download_datasets.py verify\n"
        ),
    )
    p.add_argument(
        "target",
        choices=["zenodo", "cic_doh2023", "all", "toy", "verify"],
        help=(
            "Which target to fetch. 'zenodo' downloads the ≈ 32 GB corpus; "
            "'cic_doh2023' downloads the CIC landing page; 'all' does both; "
            "'toy' generates the 1,000 + 1,000 row synthetic toy dataset "
            "(no network); 'verify' only runs the checksum verification "
            "against data/CHECKSUMS.txt."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files. Without this flag, an existing "
             "file whose size does not match the expected size is a hard "
             "error (no silent partial download).",
    )
    # --resume / --no-resume: two flags so that argparse's `store_true`
    # + `store_false` pattern can share a single `resume` destination.
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume partial downloads via HTTP Range (default: on).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable HTTP Range resume; restart from byte 0.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without opening any network "
             "connection. Use this to confirm the DOIs / URLs before a "
             "multi-hour full-mode download.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"Per-request HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_S}).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


# ─── main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Program entry point. Returns 0 on success, 1 on any failure.

    The return code is used by the Docker HEALTHCHECK in File 6 (which
    only checks imports, not this script) and by `run_all_scenarios.sh`
    (File 15f), which chains the downloader to the preprocessor and
    stops on non-zero.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    try:
        root = resolve_artifact_root()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    log.info("Artifact root:     %s", root)
    log.info("Artifact version:  %s", ARTIFACT_VERSION)
    log.info("Thesis:            %s", THESIS_CHAPTER)
    log.info("Citation DOI:      %s", CITATION_DOI)

    try:
        cfg = load_config(root)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        log.error("Failed to load master config: %s", exc)
        return 1

    try:
        if args.target in ("zenodo", "all"):
            download_zenodo(
                cfg,
                root,
                force=args.force,
                dry_run=args.dry_run,
                resume=args.resume,
                timeout_s=args.timeout,
            )
        if args.target in ("cic_doh2023", "all"):
            download_cic_doh2023(
                cfg,
                root,
                force=args.force,
                dry_run=args.dry_run,
                resume=args.resume,
                timeout_s=args.timeout,
            )
        if args.target == "toy":
            generate_toy_dataset(root)
        if args.target in ("all", "verify") and not args.dry_run:
            verify_all(root, cfg)
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
        requests.RequestException,
    ) as exc:
        log.error("%s", exc)
        return 1

    if args.dry_run:
        log.info("DRY RUN complete. No files were written.")
        return 0

    # ─── NEXT STEP banner (consumed by run_all_scenarios.sh, File 15f) ────
    print()
    print("NEXT STEP:")
    if args.target in ("zenodo", "all"):
        print(
            "  python data/preprocess_zenodo.py "
            "--config configs/default_pafa_config.yaml"
        )
    if args.target in ("cic_doh2023", "all"):
        print(
            "  python data/preprocess_doh2023.py "
            "--config configs/default_pafa_config.yaml"
        )
    if args.target == "toy":
        print(
            "  python scripts/train_pafa.py "
            "--config configs/default_pafa_config.yaml --mode toy"
        )
    if args.target == "verify":
        print("  (verification only; nothing else to do)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
