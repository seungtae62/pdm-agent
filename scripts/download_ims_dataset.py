"""NASA IMS Bearing Dataset Download & Extraction Script.

Downloads the IMS Bearing Dataset from NASA's repository,
extracts it, and organizes it into the project's data directory.

Usage:
    python scripts/download_ims_dataset.py
    python scripts/download_ims_dataset.py --data-dir /path/to/data
    python scripts/download_ims_dataset.py --force

Prerequisites:
    - py7zr (pip install py7zr) for .7z extraction
    - 7-Zip (https://7-zip.org/) for .rar extraction

Target structure after completion:
    data/ims/
    ├── 1st_test/    # ~2,156 files, 8 channels (2 per bearing)
    ├── 2nd_test/    # ~984 files, 4 channels (1 per bearing)
    └── 3rd_test/    # ~6,324 files, 4 channels (1 per bearing)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Download URLs in priority order
DOWNLOAD_URLS = [
    {
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
        "name": "NASA PCoE S3",
    },
    {
        "url": "https://data.nasa.gov/docs/legacy/IMS.zip",
        "name": "NASA Open Data Portal",
    },
]

EXPECTED_TEST_SETS = ["1st_test", "2nd_test", "3rd_test"]

EXPECTED_FILE_COUNTS = {
    "1st_test": 2156,
    "2nd_test": 984,
    "3rd_test": 6324,
}

VALIDATION_TOLERANCE = 0.03

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ims"


def _rmtree(path: Path) -> None:
    """Remove a directory tree, handling read-only files on Windows."""

    def _on_error(func, fpath, _exc_info):
        os.chmod(fpath, 0o777)
        func(fpath)

    shutil.rmtree(path, onexc=_on_error)


def _count_test_set_files(data_dir: Path) -> dict[str, int]:
    """Count files in each test set directory.

    Returns a dict mapping test set name to file count.
    Missing directories are omitted from the result.
    """
    counts = {}
    for test_set in EXPECTED_TEST_SETS:
        test_dir = data_dir / test_set
        if test_dir.exists():
            counts[test_set] = sum(1 for f in test_dir.iterdir() if f.is_file())
    return counts


def check_existing(data_dir: Path) -> bool:
    """Check if the dataset already exists and appears complete."""
    if not data_dir.exists():
        return False

    counts = _count_test_set_files(data_dir)
    missing = [ts for ts in EXPECTED_TEST_SETS if ts not in counts]
    incomplete = []

    for test_set, file_count in counts.items():
        expected = EXPECTED_FILE_COUNTS[test_set]
        tolerance = int(expected * VALIDATION_TOLERANCE)
        if abs(file_count - expected) > tolerance:
            incomplete.append(f"{test_set} ({file_count}/{expected} files)")

    if missing:
        logger.info("Missing test sets: %s", ", ".join(missing))
        return False

    if incomplete:
        logger.warning("Incomplete test sets: %s", ", ".join(incomplete))
        return False

    logger.info("Dataset already exists and appears complete at %s", data_dir)
    return True


def download_with_progress(url: str, dest_path: Path, timeout: int = 60) -> bool:
    """Download a file with tqdm progress bar. Returns True on success."""
    tmp_path = dest_path.with_suffix(".tmp")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(tmp_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="  Downloading"
            ) as pbar:
                while chunk := resp.read(1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

        os.replace(str(tmp_path), str(dest_path))
        return True

    except (urllib.error.URLError, OSError) as e:
        logger.error("Download failed: %s", e)
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def try_download(dest_path: Path) -> bool:
    """Try downloading from multiple sources. Returns True on success."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for source in DOWNLOAD_URLS:
        logger.info("Trying %s ...", source["name"])
        logger.info("  URL: %s", source["url"])

        if download_with_progress(source["url"], dest_path):
            logger.info("Download complete: %s", dest_path)
            return True

        logger.warning("Failed from %s, trying next source...", source["name"])

    logger.error("All download sources failed.")
    logger.error("Please download the dataset manually from:")
    for source in DOWNLOAD_URLS:
        logger.error("  - %s", source["url"])
    logger.error("Then extract it to: %s", dest_path.parent / "ims")
    return False


def _find_7z_exe() -> str | None:
    """Find 7z executable on the system."""
    # Linux/macOS: search PATH for common variants
    for cmd in ("7z", "7za", "7zr"):
        if shutil.which(cmd):
            return cmd
    # Windows: check common install locations
    if sys.platform == "win32":
        for prog_dir in (r"C:\Program Files\7-Zip", r"C:\Program Files (x86)\7-Zip"):
            exe = os.path.join(prog_dir, "7z.exe")
            if os.path.isfile(exe):
                return exe
    return None


_7Z_RETURN_CODES = {
    1: "Warning (non-fatal, e.g. locked files)",
    2: "Fatal error",
    7: "Command line error",
    8: "Not enough memory",
    255: "User stopped the process",
}


def _extract_rar(rar_path: Path, dest_dir: Path) -> bool:
    """Extract a RAR archive using 7z."""
    exe = _find_7z_exe()
    if exe is None:
        if sys.platform == "win32":
            hint = "Install from https://7-zip.org/"
        else:
            hint = "Install with: apt install p7zip-full  (or equivalent)"
        logger.error("7-Zip is required for .rar extraction. %s", hint)
        return False
    try:
        result = subprocess.run(
            [exe, "x", "-y", f"-o{dest_dir}", str(rar_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("  Extracted nested: %s", rar_path.name)
            return True
        code_hint = _7Z_RETURN_CODES.get(result.returncode, "Unknown error")
        logger.error(
            "  7z extraction failed (exit %d: %s): %s",
            result.returncode,
            code_hint,
            result.stderr or result.stdout,
        )
        return False
    except Exception as e:
        logger.error("  Failed to extract %s: %s", rar_path.name, e)
        return False


def _find_test_set_dirs(search_dir: Path) -> dict[str, Path]:
    """Recursively find test set directories within extracted content.

    When multiple directories share the same name (e.g. nested RAR
    extraction produces 1st_test/1st_test/), prefer the one that
    contains the most data files.
    """
    found = {}
    for test_set in EXPECTED_TEST_SETS:
        candidates = [p for p in search_dir.rglob(test_set) if p.is_dir()]
        if not candidates:
            continue
        # Pick the candidate with the most files
        best = max(
            candidates, key=lambda c: sum(1 for f in c.iterdir() if f.is_file())
        )
        found[test_set] = best
    return found


def _extract_single(archive: Path, dest: Path) -> bool:
    """Extract a single archive (.zip / .7z / .rar) into dest. Returns True on success."""
    if archive.suffix == ".zip":
        try:
            with zipfile.ZipFile(str(archive), "r") as zf:
                zf.extractall(str(dest))
            logger.info("  Extracted nested: %s", archive.name)
            return True
        except (zipfile.BadZipFile, OSError, EOFError) as e:
            logger.error("  Nested archive corrupted: %s (%s)", archive.name, e)
            return False

    elif archive.suffix == ".7z":
        try:
            import py7zr

            with py7zr.SevenZipFile(str(archive), mode="r") as sz:
                sz.extractall(path=str(dest))
            logger.info("  Extracted nested: %s", archive.name)
            return True
        except ImportError:
            logger.error(
                "py7zr is required for .7z extraction. "
                "Install it with: pip install py7zr"
            )
            return False
        except Exception as e:
            logger.error("  Failed to extract %s: %s", archive.name, e)
            return False

    elif archive.suffix == ".rar":
        dest.mkdir(parents=True, exist_ok=True)
        return _extract_rar(archive, dest)

    logger.warning("  Unsupported archive format: %s", archive.name)
    return False


def _move_test_sets(found_dirs: dict[str, Path], data_dir: Path) -> None:
    """Move discovered test set directories into data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for test_set, src_path in found_dirs.items():
        dst_path = data_dir / test_set
        if dst_path.exists():
            _rmtree(dst_path)
        shutil.move(str(src_path), str(dst_path))
        logger.info("  Extracted: %s", test_set)

    missing = [ts for ts in EXPECTED_TEST_SETS if ts not in found_dirs]
    if missing:
        logger.warning("Missing test sets after extraction: %s", ", ".join(missing))


def extract_archive(archive_path: Path, data_dir: Path) -> bool:
    """Extract the dataset archive, handling nested structures."""
    logger.info("Extracting archive: %s", archive_path)

    with tempfile.TemporaryDirectory(dir=archive_path.parent) as tmp_extract:
        tmp_extract_path = Path(tmp_extract)

        # Extract outer zip
        try:
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                zf.extractall(str(tmp_extract_path))
        except (zipfile.BadZipFile, OSError, EOFError) as e:
            logger.error("Archive is corrupted: %s (%s)", archive_path, e)
            return False

        # Look for test set directories directly
        found_dirs = _find_test_set_dirs(tmp_extract_path)

        if len(found_dirs) >= len(EXPECTED_TEST_SETS):
            _move_test_sets(found_dirs, data_dir)
            return True

        # Iteratively extract nested archives until test sets are found
        extracted_files: set[str] = set()
        max_rounds = 3

        for round_num in range(max_rounds):
            found_dirs = _find_test_set_dirs(tmp_extract_path)
            if len(found_dirs) >= len(EXPECTED_TEST_SETS):
                break

            # Find nested archives not yet processed
            nested_archives = [
                a
                for a in (
                    list(tmp_extract_path.rglob("*.zip"))
                    + list(tmp_extract_path.rglob("*.7z"))
                    + list(tmp_extract_path.rglob("*.rar"))
                )
                if str(a) not in extracted_files
            ]

            if not nested_archives:
                if round_num == 0:
                    subdirs = [d for d in tmp_extract_path.iterdir() if d.is_dir()]
                    if len(subdirs) == 1:
                        found_dirs = _find_test_set_dirs(subdirs[0])
                        if found_dirs:
                            break

                    logger.error("Could not find test set directories in the archive.")
                    logger.error(
                        "Archive contents: %s",
                        list(tmp_extract_path.rglob("*"))[:20],
                    )
                    return False
                break

            logger.info(
                "Round %d: found nested archives: %s",
                round_num + 1,
                [a.name for a in nested_archives],
            )

            for nested in nested_archives:
                extracted_files.add(str(nested))
                nested_extract = tmp_extract_path / nested.stem
                _extract_single(nested, nested_extract)

        # Final check for test set directories
        found_dirs = _find_test_set_dirs(tmp_extract_path)

        if not found_dirs:
            logger.error("No test set directories found after extraction.")
            return False

        _move_test_sets(found_dirs, data_dir)

    return True


def _flatten_txt_subdir(test_dir: Path) -> None:
    """If a test set dir contains only a txt/ subdir, flatten it."""
    txt_dir = test_dir / "txt"
    if not txt_dir.is_dir():
        return
    # Only flatten if txt/ is the only content
    direct_files = [f for f in test_dir.iterdir() if f.is_file()]
    if direct_files:
        return
    # Move txt/ contents up
    for item in txt_dir.iterdir():
        shutil.move(str(item), str(test_dir / item.name))
    txt_dir.rmdir()
    logger.info("  Flattened txt/ subdirectory in %s", test_dir.name)


def handle_4th_test(data_dir: Path) -> None:
    """Handle the 3rd_test.rar naming anomaly.

    In the S3 archive, 3rd_test.rar contains only a 4th_test/ directory
    which is actually Test Set 3. This function:
    1. Moves 4th_test out of 3rd_test if nested
    2. If 3rd_test is empty and 4th_test exists, renames 4th_test -> 3rd_test
    3. Flattens any txt/ subdirectories
    """
    nested_4th = data_dir / "3rd_test" / "4th_test"
    target_4th = data_dir / "4th_test"
    target_3rd = data_dir / "3rd_test"

    # Step 1: Move 4th_test out of 3rd_test if nested
    if nested_4th.exists() and nested_4th.is_dir():
        if target_4th.exists():
            _rmtree(target_4th)
        shutil.move(str(nested_4th), str(target_4th))
        logger.info("Moved 4th_test from 3rd_test/ to data/ims/4th_test/")

    # Step 2: If 3rd_test is empty and 4th_test exists, rename 4th_test -> 3rd_test
    if target_3rd.exists() and target_4th.exists():
        third_files = sum(1 for _ in target_3rd.rglob("*") if _.is_file())
        if third_files == 0:
            _rmtree(target_3rd)
            shutil.move(str(target_4th), str(target_3rd))
            logger.info("Renamed 4th_test -> 3rd_test (S3 archive naming fix)")

    # Step 3: Flatten txt/ subdirectories
    for test_set in EXPECTED_TEST_SETS:
        test_dir = data_dir / test_set
        if test_dir.exists():
            _flatten_txt_subdir(test_dir)


def validate_dataset(data_dir: Path) -> bool:
    """Validate the extracted dataset."""
    logger.info("Validating dataset...")
    all_ok = True

    counts = _count_test_set_files(data_dir)

    logger.info("")
    logger.info("  Test Set     | Files  | Expected | Status")
    logger.info("  -------------|--------|----------|-------")

    for test_set in EXPECTED_TEST_SETS:
        expected = EXPECTED_FILE_COUNTS[test_set]

        if test_set not in counts:
            logger.info("  %s | -      | %8d | MISSING", test_set.ljust(13), expected)
            all_ok = False
            continue

        file_count = counts[test_set]
        tolerance = int(expected * VALIDATION_TOLERANCE)
        status = "OK" if abs(file_count - expected) <= tolerance else "MISMATCH"

        if status == "MISMATCH":
            all_ok = False

        logger.info(
            "  %s | %6d | %8d | %s",
            test_set.ljust(13),
            file_count,
            expected,
            status,
        )

    # Check for 4th_test (bonus)
    test_4th = data_dir / "4th_test"
    if test_4th.exists():
        file_count = sum(1 for f in test_4th.iterdir() if f.is_file())
        logger.info("  %s | %6d | %8s | BONUS", "4th_test".ljust(13), file_count, "N/A")

    logger.info("")

    # Sample validation: check a file from each test set
    for test_set in EXPECTED_TEST_SETS:
        test_dir = data_dir / test_set
        if not test_dir.exists():
            continue

        sample_file = next(
            (f for f in sorted(test_dir.iterdir()) if f.is_file()), None
        )
        if sample_file:
            try:
                with open(sample_file, "r") as f:
                    lines = f.readlines()
                line_count = len(lines)
                if line_count > 0:
                    col_count = len(lines[0].strip().split("\t"))
                    logger.info(
                        "  %s sample: %d lines, %d channels (%s)",
                        test_set,
                        line_count,
                        col_count,
                        sample_file.name,
                    )
            except Exception as e:
                logger.warning("  Could not read sample from %s: %s", test_set, e)

    return all_ok


def check_disk_space(target_dir: Path, required_gb: float = 12.0) -> bool:
    """Check if there's enough disk space."""
    try:
        stat = shutil.disk_usage(str(target_dir.parent))
        free_gb = stat.free / (1024**3)
        if free_gb < required_gb:
            logger.error(
                "Insufficient disk space: %.1f GB free, %.1f GB required",
                free_gb,
                required_gb,
            )
            return False
        logger.info("Disk space: %.1f GB free (%.1f GB required)", free_gb, required_gb)
        return True
    except OSError:
        logger.warning("Could not check disk space, proceeding anyway...")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract NASA IMS Bearing Dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Target directory for dataset (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    logger.info("Target directory: %s", data_dir)

    # Idempotency check
    if not args.force and check_existing(data_dir):
        logger.info("Use --force to re-download.")
        return 0

    # Force mode: clean up existing data
    if args.force and data_dir.exists():
        logger.info("Force mode: removing existing dataset...")
        _rmtree(data_dir)

    # Disk space check
    data_dir.mkdir(parents=True, exist_ok=True)
    if not check_disk_space(data_dir):
        return 1

    # Download
    zip_path = data_dir.parent / "ims_bearing_dataset.zip"

    if zip_path.exists() and not args.force:
        logger.info("Using existing archive: %s", zip_path)
    else:
        if not try_download(zip_path):
            return 1

    # Extract
    if not extract_archive(zip_path, data_dir):
        return 1

    # Handle 4th_test anomaly
    handle_4th_test(data_dir)

    # Validate
    if validate_dataset(data_dir):
        logger.info("Dataset is ready!")
    else:
        logger.warning("Dataset validation found issues. Check the output above.")

    # Clean up zip
    if not args.keep_zip and zip_path.exists():
        logger.info("Removing archive: %s", zip_path)
        zip_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
