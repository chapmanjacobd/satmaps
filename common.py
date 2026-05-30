import math
import os

from osgeo import gdal


def format_eta(
    seconds_remaining: float | None,
    *,
    elapsed_seconds: float | None = None,
) -> str:
    """Return a short ETA string for progress logging."""
    if seconds_remaining is None or not math.isfinite(seconds_remaining):
        return "ETA: calculating..."

    rounded_seconds = max(0, round(seconds_remaining))
    if rounded_seconds == 0 and elapsed_seconds is not None and math.isfinite(elapsed_seconds):
        rounded_elapsed = max(0, round(elapsed_seconds))
        hours, remainder = divmod(rounded_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"Elapsed: {hours}h {minutes}m"
        if minutes:
            return f"Elapsed: {minutes}m {seconds}s"
        return f"Elapsed: {seconds}s"
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"ETA: {hours}h {minutes}m"
    if minutes:
        return f"ETA: {minutes}m {seconds}s"
    return f"ETA: {seconds}s"


class LiveProgressLine:
    """Render a single in-place progress line without hiding later errors."""

    def __init__(self) -> None:
        self._active = False
        self._last_width = 0

    def update(self, message: str) -> None:
        padded_message = message
        if len(message) < self._last_width:
            padded_message += " " * (self._last_width - len(message))
        else:
            self._last_width = len(message)
        print(f"\r{padded_message}", end="", flush=True)
        self._active = True

    def finish(self) -> None:
        if not self._active:
            return
        print()
        self._active = False
        self._last_width = 0


def build_staged_path(path: str) -> str:
    """Return the hidden staging path used before atomically publishing a file."""
    directory, basename = os.path.split(path)
    return os.path.join(directory, f".temp_{basename}")


def file_has_content(path: str) -> bool:
    """Return True when a path exists and has non-zero size."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def publish_staged_path(staged_path: str, final_path: str) -> str:
    """Atomically publish a staged file under its final name."""
    if not file_has_content(staged_path):
        raise RuntimeError(f"Refusing to publish empty staged file: {staged_path}")
    os.replace(staged_path, final_path)
    return final_path


def remove_if_exists(path: str) -> None:
    """Delete a file if it exists."""
    if os.path.exists(path):
        try:
            gdal.Unlink(path)
        except RuntimeError:
            os.remove(path)
