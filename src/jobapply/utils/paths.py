"""Safe artifact path utilities and collision-resistant bounded identifier sanitization."""

import hashlib
import re
from pathlib import Path
from typing import Optional

# Hard component length limit
MAX_COMPONENT_LENGTH = 128

# Windows reserved device names
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}

_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,118}\.[a-zA-Z0-9]{1,8}$")
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")
_SAFE_EXTENSION_PATTERN = re.compile(r"^\.[a-zA-Z0-9]{1,8}$")


def sanitize_component(value: str, default: str = "item") -> str:
    """Deterministically sanitize an untrusted path component with strict 128-char bound and collision resistance.

    - If already strictly safe and <= 128 chars, returns unmodified.
    - If modified/unsafe, generates a short stable sha256 hash and bounds output strictly <= 128 chars.
    - Preserves a valid short file extension if present and safe.

    Args:
        value: Untrusted string (e.g. run_id, job_id, screenshot label, filename).
        default: Fallback name if the result is empty.

    Returns:
        Safe, collision-resistant component string strictly <= 128 characters.
    """
    if not isinstance(value, str):
        value = str(value) if value is not None else ""

    # Check if already strictly safe and bounded
    if len(value) <= MAX_COMPONENT_LENGTH:
        stem = value.split(".")[0].upper()
        if (
            (_SAFE_IDENTIFIER_PATTERN.match(value) or _SAFE_FILENAME_PATTERN.match(value))
            and stem not in _WINDOWS_RESERVED_NAMES
            and value.upper() not in _WINDOWS_RESERVED_NAMES
            and not value.startswith((".", "-"))
        ):
            return value

    # Compute a stable short hash of the raw input to guarantee distinct outputs
    raw_hash = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:8]

    # Strip control chars and null bytes
    cleaned = _CONTROL_CHARS_PATTERN.sub("", value)

    # Normalize slashes and remove drive/UNC patterns
    cleaned = cleaned.replace("\\", "/").strip()
    if re.match(r"^[a-zA-Z]:", cleaned):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")

    # Take the basename in case a path was supplied
    cleaned = Path(cleaned).name

    # Separate stem and extension
    path_obj = Path(cleaned)
    suffix = path_obj.suffix
    base_stem = path_obj.stem if suffix else cleaned

    # Check if suffix is a safe short extension
    clean_suffix = ""
    if suffix and _SAFE_EXTENSION_PATTERN.match(suffix):
        clean_suffix = suffix.lower()
    else:
        # If suffix was unsafe or long, merge back into base stem
        base_stem = cleaned

    # Sanitize stem
    clean_stem = _UNSAFE_CHARS_PATTERN.sub("_", base_stem).strip(". ")
    clean_stem = re.sub(r"_+", "_", clean_stem)

    # Check for Windows reserved device names
    if clean_stem.upper() in _WINDOWS_RESERVED_NAMES:
        clean_stem = f"safe_{clean_stem}"

    if not clean_stem or clean_stem in (".", ".."):
        clean_stem = default

    # Enforce hard length limit: max 128 chars total including '_h' (2 chars) + hash (8 chars) + suffix
    hash_tag = f"_h{raw_hash}"
    max_stem_len = MAX_COMPONENT_LENGTH - len(hash_tag) - len(clean_suffix)
    bounded_stem = clean_stem[:max_stem_len].rstrip("._")
    if not bounded_stem:
        bounded_stem = default[:max_stem_len]

    result = f"{bounded_stem}{hash_tag}{clean_suffix}"
    return result[:MAX_COMPONENT_LENGTH]


def validate_run_id(run_id: str) -> str:
    """Validate CLI-provided run_id strictly.

    Rejects traversal, path separators, absolute paths, Windows reserved names,
    control characters, and unsafe characters.

    Args:
        run_id: CLI-provided run identifier.

    Returns:
        Validated run_id string.

    Raises:
        ValueError: If run_id is invalid or dangerous.
    """
    if not run_id or not isinstance(run_id, str):
        raise ValueError("Run ID cannot be empty.")

    if len(run_id) > MAX_COMPONENT_LENGTH:
        raise ValueError(
            f"Invalid run_id: exceeds maximum length of {MAX_COMPONENT_LENGTH} characters."
        )

    if _CONTROL_CHARS_PATTERN.search(run_id):
        raise ValueError(f"Invalid run_id '{run_id}': contains control characters.")

    if any(sep in run_id for sep in ("/", "\\", ":", "..")):
        raise ValueError(
            f"Invalid run_id '{run_id}': path traversal and separators are not allowed."
        )

    # Check Windows reserved names
    stem = run_id.split(".")[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES or run_id.upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(
            f"Invalid run_id '{run_id}': Windows reserved device names cannot be used as run IDs."
        )

    if not _SAFE_IDENTIFIER_PATTERN.match(run_id):
        raise ValueError(
            f"Invalid run_id '{run_id}': must contain only alphanumeric characters, dashes, and underscores (max {MAX_COMPONENT_LENGTH} characters)."
        )

    return run_id


def get_outputs_root(base_dir: Optional[Path | str] = None) -> Path:
    """Get canonical absolute outputs root directory."""
    if base_dir is not None:
        root = Path(base_dir).resolve()
    else:
        root = Path("outputs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_run_output_dir(run_id: str, base_dir: Optional[Path | str] = None) -> Path:
    """Get safe resolved output directory for a specific run under outputs/.

    Guarantees the resulting path is strictly contained within outputs_root.

    Args:
        run_id: Run identifier (sanitized automatically).
        base_dir: Optional base outputs directory.

    Returns:
        Safe, contained Path object for the run directory.
    """
    root = get_outputs_root(base_dir)
    safe_run_id = sanitize_component(run_id, default="default_run")
    run_dir = (root / safe_run_id).resolve()

    try:
        run_dir.relative_to(root)
    except ValueError:
        raise RuntimeError(f"Path containment violation: '{run_dir}' is outside '{root}'")

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_safe_artifact_path(
    run_id: str,
    filename: str,
    subfolder: Optional[str] = None,
    base_dir: Optional[Path | str] = None,
) -> Path:
    """Build a safe artifact path guaranteed to resolve within outputs/<safe-run-id>/.

    Args:
        run_id: Untrusted run identifier.
        filename: Untrusted artifact filename.
        subfolder: Optional untrusted subfolder name (e.g. 'errors').
        base_dir: Optional base outputs directory.

    Returns:
        Safe, contained Path object.
    """
    run_dir = get_run_output_dir(run_id, base_dir)

    target_dir = run_dir
    if subfolder:
        safe_subfolder = sanitize_component(subfolder, default="subfolder")
        target_dir = (run_dir / safe_subfolder).resolve()
        try:
            target_dir.relative_to(run_dir)
        except ValueError:
            raise RuntimeError(f"Path containment violation: '{target_dir}' is outside '{run_dir}'")
        target_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = sanitize_component(filename, default="artifact")
    final_path = (target_dir / safe_filename).resolve()

    try:
        final_path.relative_to(target_dir)
    except ValueError:
        raise RuntimeError(f"Path containment violation: '{final_path}' is outside '{target_dir}'")

    return final_path


def get_cover_letter_path(run_id: str, job_id: str, base_dir: Optional[Path | str] = None) -> Path:
    """Build safe path for a generated cover letter."""
    safe_job_id = sanitize_component(job_id, default="job")
    return get_safe_artifact_path(
        run_id=run_id,
        filename=f"cover_letter_{safe_job_id}.txt",
        base_dir=base_dir,
    )


def get_edited_resume_path(run_id: str, job_id: str, base_dir: Optional[Path | str] = None) -> Path:
    """Build safe path for an edited resume PDF."""
    safe_job_id = sanitize_component(job_id, default="job")
    return get_safe_artifact_path(
        run_id=run_id,
        filename=f"edited_resume_{safe_job_id}.pdf",
        base_dir=base_dir,
    )


def get_error_screenshot_path(
    run_id: str, label: str, base_dir: Optional[Path | str] = None
) -> Path:
    """Build safe path for an error screenshot PNG."""
    safe_label = sanitize_component(label, default="error")
    return get_safe_artifact_path(
        run_id=run_id,
        filename=f"{safe_label}.png",
        subfolder="errors",
        base_dir=base_dir,
    )
