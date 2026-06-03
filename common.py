import contextlib
import math
import os
from typing import Iterator

from osgeo import gdal

# Active GDAL warp NUM_THREADS budget. Defaults to "ALL_CPUS" so sequential warps
# use the whole machine; worker pools temporarily lower it (see warp_thread_budget)
# so an outer pool of N workers each running an ALL_CPUS warp doesn't oversubscribe
# the CPU (N x ALL_CPUS threads = context-switch thrash, not speedup).
_WARP_NUM_THREADS = "ALL_CPUS"


def warp_thread_options() -> list[str]:
    """gdal.Warp ``warpOptions`` entry honoring the active per-worker thread budget."""
    return [f"NUM_THREADS={_WARP_NUM_THREADS}"]


def per_worker_warp_threads(parallel: int) -> int:
    """Split the machine's cores across ``parallel`` outer workers.

    Keeps the total warp thread count near the core count instead of
    ``parallel x cpu_count``. When ``parallel`` exceeds the core count this
    floors at 1 thread per warp, bounding contention to roughly ``parallel``.
    """
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count // max(1, parallel))


@contextlib.contextmanager
def warp_thread_budget(num_threads: "int | str") -> Iterator[None]:
    """Scope the warp NUM_THREADS budget for the duration of a worker pool.

    The budget is set by the calling (main) thread before workers start and
    restored after the pool's ``with`` block joins all workers, so worker
    threads only ever read a stable value.
    """
    global _WARP_NUM_THREADS
    previous = _WARP_NUM_THREADS
    _WARP_NUM_THREADS = str(num_threads)
    try:
        yield
    finally:
        _WARP_NUM_THREADS = previous


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
