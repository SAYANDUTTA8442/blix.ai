"""
Background Memory Processing — Blix v0.3  (Feature 6)

New pipeline:
    User → LLM → Response (returned immediately)
    Background Worker (thread):
        Conversation → Memory Extraction
                     → Profile Update (ProfileEvolver)
                     → Summary Generation (HierarchyManager)
                     → Graph Update (MemoryGraph)

Chat latency is NEVER blocked by memory extraction.

Architecture
------------
* ``MemoryTask``   — dataclass describing one unit of async work.
* ``BackgroundProcessor`` — thread-safe queue + worker thread + retry.
* ``ProcessorJob`` — enum of job types for extensibility.

Python 3.10 compatible — uses ``queue.Queue`` and ``threading.Thread``.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Job types
# ---------------------------------------------------------------------------


class ProcessorJob(Enum):
    EXTRACT_AND_UPDATE = auto()   # primary post-turn job
    REGENERATE_SUMMARY = auto()   # on-demand summary refresh
    UPDATE_GRAPH = auto()         # isolated graph update
    REBUILD_INDEX = auto()        # embedding index rebuild


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class MemoryTask:
    """One unit of background work."""

    job: ProcessorJob
    payload: dict = field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 3

    def can_retry(self) -> bool:
        return self.attempt < self.max_attempts


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


class BackgroundProcessor:
    """
    Thread-safe background worker for post-turn memory processing.

    Parameters
    ----------
    max_queue_size:
        Maximum pending in-memory tasks before new submissions overflow to
        durable disk storage (NOT dropped — see Issue 11 fix in v0.3.1).
    worker_count:
        Number of parallel worker threads (default 1 for simplicity/ordering).
    overflow_file:
        Optional path to a JSONL file used as durable overflow storage.
        When the in-memory queue is full, tasks are appended here instead
        of being dropped.  Call ``drain_overflow()`` (e.g. on startup or
        periodically) to requeue them.  If ``None``, overflow falls back
        to the v0.3 drop-with-warning behaviour.
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        worker_count: int = 1,
        overflow_file: Optional[Path] = None,
    ) -> None:
        self._queue: queue.Queue[Optional[MemoryTask]] = queue.Queue(maxsize=max_queue_size)
        self._handlers: dict[ProcessorJob, Callable[[dict], None]] = {}
        self._workers: list[threading.Thread] = []
        self._worker_count = worker_count
        self._running = False
        self._processed = 0
        self._failed = 0
        self._overflowed = 0
        self._lock = threading.Lock()
        self._overflow_file = overflow_file
        self._overflow_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register(self, job: ProcessorJob, handler: Callable[[dict], None]) -> None:
        """Register a callable handler for a given job type."""
        self._handlers[job] = handler
        log.debug("BackgroundProcessor: registered handler for %s", job.name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background worker thread(s). Safe to call multiple times."""
        if self._running:
            return
        self._running = True
        for i in range(self._worker_count):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"blix-bg-worker-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        log.info("BackgroundProcessor started (%d worker(s)).", self._worker_count)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Signal workers to stop and wait for them to drain.

        Sends ``None`` sentinel for each worker thread.
        """
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=timeout)
        self._workers.clear()
        log.info(
            "BackgroundProcessor stopped. processed=%d failed=%d",
            self._processed,
            self._failed,
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, job: ProcessorJob, payload: dict) -> bool:
        """
        Submit a task to the queue.

        Returns ``True`` if enqueued in-memory OR durably persisted to the
        overflow file.  Only returns ``False`` if the queue is full AND no
        ``overflow_file`` was configured (v0.3 legacy drop behaviour).

        v0.3.1: tasks are never silently lost when an overflow_file is set —
        see Issue 11.
        """
        task = MemoryTask(job=job, payload=payload)
        try:
            self._queue.put_nowait(task)
            log.debug("BackgroundProcessor: enqueued %s", job.name)
            return True
        except queue.Full:
            if self._overflow_file is not None:
                self._write_overflow(task)
                with self._lock:
                    self._overflowed += 1
                log.warning(
                    "BackgroundProcessor: queue full — persisted %s to overflow file.",
                    job.name,
                )
                return True
            log.warning("BackgroundProcessor: queue full — dropping task %s", job.name)
            return False

    # ------------------------------------------------------------------
    # Durable overflow (Issue 11)
    # ------------------------------------------------------------------

    def _write_overflow(self, task: MemoryTask) -> None:
        """Append a task to the JSONL overflow file."""
        assert self._overflow_file is not None
        self._overflow_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "job": task.job.name,
            "payload": task.payload,
            "attempt": task.attempt,
            "max_attempts": task.max_attempts,
        }
        with self._overflow_lock:
            with self._overflow_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

    def drain_overflow(self) -> int:
        """
        Re-enqueue all tasks from the overflow file into the in-memory queue.

        Call this on startup (or periodically) to recover overflowed tasks.
        Truncates the overflow file once all entries have been re-enqueued.
        Entries that still don't fit are left in the file for the next drain.

        Returns the number of tasks successfully re-enqueued.
        """
        if self._overflow_file is None or not self._overflow_file.exists():
            return 0

        with self._overflow_lock:
            try:
                with self._overflow_file.open("r", encoding="utf-8") as fh:
                    lines = [l for l in fh.read().splitlines() if l.strip()]
            except Exception as exc:
                log.warning("BackgroundProcessor: overflow read failed (%s)", exc)
                return 0

            requeued = 0
            remaining: list[str] = []
            for line in lines:
                try:
                    rec = json.loads(line)
                    task = MemoryTask(
                        job=ProcessorJob[rec["job"]],
                        payload=rec["payload"],
                        attempt=rec.get("attempt", 0),
                        max_attempts=rec.get("max_attempts", 3),
                    )
                    self._queue.put_nowait(task)
                    requeued += 1
                except queue.Full:
                    remaining.append(line)
                except Exception as exc:
                    log.warning("BackgroundProcessor: skipping malformed overflow entry (%s)", exc)

            if remaining:
                with self._overflow_file.open("w", encoding="utf-8") as fh:
                    fh.write("\n".join(remaining) + "\n")
            else:
                self._overflow_file.unlink(missing_ok=True)

        if requeued:
            log.info("BackgroundProcessor: drained %d task(s) from overflow.", requeued)
        return requeued

    @property
    def overflow_pending(self) -> int:
        """Number of lines currently sitting in the overflow file."""
        if self._overflow_file is None or not self._overflow_file.exists():
            return 0
        try:
            with self._overflow_file.open("r", encoding="utf-8") as fh:
                return sum(1 for l in fh if l.strip())
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Internal worker loop — runs in a daemon thread."""
        while True:
            try:
                task: Optional[MemoryTask] = self._queue.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if task is None:
                break  # stop sentinel

            self._process(task)
            self._queue.task_done()

    def _process(self, task: MemoryTask) -> None:
        """Execute one task with retry logic and failure isolation."""
        handler = self._handlers.get(task.job)
        if handler is None:
            log.warning("No handler for job %s — skipping.", task.job.name)
            return

        task.attempt += 1
        try:
            handler(task.payload)
            with self._lock:
                self._processed += 1
            log.debug("BackgroundProcessor: completed %s (attempt %d)", task.job.name, task.attempt)
        except Exception:
            with self._lock:
                self._failed += 1
            log.error(
                "BackgroundProcessor: %s failed (attempt %d/%d)\n%s",
                task.job.name,
                task.attempt,
                task.max_attempts,
                traceback.format_exc(),
            )
            if task.can_retry():
                try:
                    self._queue.put_nowait(task)
                    log.info("BackgroundProcessor: requeued %s for retry.", task.job.name)
                except queue.Full:
                    log.warning("BackgroundProcessor: queue full — cannot retry %s", task.job.name)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "queue_size": self._queue.qsize(),
                "processed": self._processed,
                "failed": self._failed,
                "overflowed": self._overflowed,
                "overflow_pending": self.overflow_pending,
            }
