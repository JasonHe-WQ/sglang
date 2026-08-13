import os
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range

PREFETCH_BATCH_NVTX_PREFIX = "sglang.prefetch_batch"


def format_prefetch_batch_nvtx_name(
    *, mode: str, layer_idx: int, chunk_size_bytes: int
) -> str:
    """Return the stable marker schema consumed by Nsys prefetch analyzers."""
    return (
        f"{PREFETCH_BATCH_NVTX_PREFIX}|mode={mode}|layer_idx={int(layer_idx)}"
        f"|chunk_size_bytes={int(chunk_size_bytes)}"
    )


class CooperativeCopyHandle:
    """Completion handle for a copy batch submitted by the host worker."""

    def __init__(self) -> None:
        self._submitted = threading.Event()
        self._completion_event: Any | None = None
        self._error: BaseException | None = None

    def _set_completion_event(self, event: Any) -> None:
        self._completion_event = event
        self._submitted.set()

    def _set_error(self, error: BaseException) -> None:
        self._error = error
        self._submitted.set()

    def wait_event(self, timeout: float | None = None):
        if not self._submitted.wait(timeout):
            raise TimeoutError("Timed out waiting for cooperative H2D submission")
        if self._error is not None:
            raise RuntimeError("Cooperative H2D submission failed") from self._error
        return self._completion_event


@dataclass
class _CopyBatch:
    pairs: list[tuple[torch.Tensor, torch.Tensor]]
    chunk_size_bytes: int
    layer_idx: int
    ready_event: Any
    enable_prefetch_nvtx: bool = False
    handle: CooperativeCopyHandle = field(default_factory=CooperativeCopyHandle)


class CooperativePrefetchScheduler:
    """Bounded H2D producer coordinated with high-priority A2A collectives.

    A daemon host thread submits H2D blocks on one low-priority CUDA stream. It
    fences after ``queue_depth`` blocks, bounding work that can already be in the
    copy engine when an A2A requests the gate. The A2A caller waits only for that
    bounded window to drain; H2D submission resumes after an event recorded after
    the collective on the caller's current CUDA stream completes.
    """

    def __init__(
        self,
        device_index: int,
        queue_depth: int,
        *,
        device_module=None,
    ) -> None:
        if queue_depth < 1:
            raise ValueError("queue_depth must be at least 1")
        self.device_index = int(device_index)
        self.queue_depth = int(queue_depth)
        self._device_module = device_module or torch.cuda
        self.stream = self._device_module.Stream(device=self.device_index, priority=0)
        self._fence_event = self._device_module.Event()

        self._condition = threading.Condition()
        self._jobs: deque[_CopyBatch] = deque()
        self._stopping = False
        self._pause_requested = False
        self._drained = False
        self._gate_generation = 0
        self._active_a2a: set[int] = set()
        self._resume_events: list[tuple[int, Any]] = []
        self._inflight_blocks = 0
        self._copy_active = False

        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"sglang-prefetch-cuda-{self.device_index}",
            daemon=True,
        )
        self._worker.start()

    @staticmethod
    def _storage_view(tensor: torch.Tensor) -> torch.Tensor:
        storage_numel = tensor.untyped_storage().nbytes() // tensor.element_size()
        return tensor.as_strided((storage_numel,), (1,), storage_offset=0)

    def submit(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        chunk_size_bytes: int,
        *,
        layer_idx: int,
        enable_prefetch_nvtx: bool = False,
    ) -> CooperativeCopyHandle:
        if chunk_size_bytes < 1:
            raise ValueError("chunk_size_bytes must be at least 1")
        ready_event = self._device_module.Event()
        ready_event.record(self._device_module.current_stream())
        batch = _CopyBatch(
            pairs=list(pairs),
            chunk_size_bytes=int(chunk_size_bytes),
            layer_idx=int(layer_idx),
            ready_event=ready_event,
            enable_prefetch_nvtx=bool(enable_prefetch_nvtx),
        )
        with self._condition:
            if self._stopping:
                raise RuntimeError("Cooperative prefetch scheduler is stopped")
            self._jobs.append(batch)
            self._condition.notify_all()
        return batch.handle

    def begin_a2a(self) -> int:
        """Pause H2D submission and wait for the bounded in-flight window."""
        with self._condition:
            if self._stopping:
                raise RuntimeError("Cooperative prefetch scheduler is stopped")
            self._gate_generation += 1
            generation = self._gate_generation
            self._active_a2a.add(generation)
            self._pause_requested = True
            if not self._copy_active and not self._jobs and self._inflight_blocks == 0:
                self._drained = True
            self._condition.notify_all()
            while not self._drained and not self._stopping:
                self._condition.wait()
            if self._stopping:
                raise RuntimeError("Cooperative prefetch scheduler stopped at A2A gate")
            return generation

    def end_a2a(self, generation: int, completion_event: Any) -> None:
        """Let the worker resume after the A2A completion event fires."""
        with self._condition:
            if generation not in self._active_a2a:
                return
            self._active_a2a.remove(generation)
            self._resume_events.append((generation, completion_event))
            self._pause_requested = bool(self._active_a2a)
            self._condition.notify_all()

    def abort_a2a(self, generation: int) -> None:
        """Stop the scheduler when an A2A completion fence cannot be recorded."""
        with self._condition:
            self._active_a2a.discard(generation)
            self._stopping = True
            self._condition.notify_all()

    def record_a2a_completion(self):
        event = self._device_module.Event()
        event.record(self._device_module.current_stream())
        return event

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if wait and threading.current_thread() is not self._worker:
            self._worker.join()

    def _copy_block(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        destination[start:end].copy_(source[start:end], non_blocking=True)

    def _fence_inflight(self, *, force: bool = False) -> None:
        if self._inflight_blocks == 0 and not force:
            return
        self._fence_event.record(self.stream)
        self._fence_event.synchronize()
        self._inflight_blocks = 0

    def _cooperate_with_a2a(self) -> bool:
        """Return False when shutdown was requested, otherwise True."""
        while True:
            with self._condition:
                stopping = self._stopping
                pause_requested = self._pause_requested
                resume_events = tuple(self._resume_events)

            if stopping:
                self._fence_inflight()
                return False

            if pause_requested:
                self._fence_inflight()
                with self._condition:
                    if self._stopping:
                        return False
                    if self._pause_requested:
                        self._drained = True
                        self._condition.notify_all()
                        while self._pause_requested and not self._stopping:
                            self._condition.wait()
                        if self._stopping:
                            return False
                continue

            if resume_events:
                for _, event in resume_events:
                    event.synchronize()
                with self._condition:
                    if self._stopping:
                        return False
                    completed_tokens = {token for token, _ in resume_events}
                    self._resume_events = [
                        item
                        for item in self._resume_events
                        if item[0] not in completed_tokens
                    ]
                    if not self._pause_requested and not self._resume_events:
                        self._drained = False
                        self._condition.notify_all()
                        return True
                continue

            return True

    def _next_job(self) -> _CopyBatch | None:
        while True:
            if not self._cooperate_with_a2a():
                return None
            with self._condition:
                if self._stopping:
                    return None
                if self._jobs:
                    self._copy_active = True
                    return self._jobs.popleft()
                self._condition.wait()

    def _run_batch_body(self, batch: _CopyBatch) -> None:
        self.stream.wait_event(batch.ready_event)
        storage_pairs = []
        for destination, source in batch.pairs:
            destination_storage = self._storage_view(destination)
            source_storage = self._storage_view(source)
            if destination_storage.numel() != source_storage.numel():
                raise RuntimeError(
                    "Cooperative H2D source and destination storage sizes differ: "
                    f"{source_storage.numel()} != {destination_storage.numel()}"
                )
            storage_pairs.append((destination_storage, source_storage))

        for destination_storage, source_storage in storage_pairs:
            chunk_numel = max(
                1, batch.chunk_size_bytes // source_storage.element_size()
            )
            for start in range(0, source_storage.numel(), chunk_numel):
                if not self._cooperate_with_a2a():
                    raise RuntimeError("Cooperative prefetch scheduler stopped")
                end = min(start + chunk_numel, source_storage.numel())
                self._copy_block(destination_storage, source_storage, start, end)
                self._inflight_blocks += 1
                if self._inflight_blocks >= self.queue_depth:
                    self._fence_inflight()

        # Finish the tail shorter than queue_depth so an idle scheduler has no
        # hidden copy-engine work and the next A2A can take the fast gate path.
        self._fence_inflight()
        completion_event = self._device_module.Event()
        completion_event.record(self.stream)
        batch.handle._set_completion_event(completion_event)

    def _run_batch(self, batch: _CopyBatch) -> None:
        if not batch.enable_prefetch_nvtx:
            self._run_batch_body(batch)
            return

        marker = format_prefetch_batch_nvtx_name(
            mode="cooperative",
            layer_idx=batch.layer_idx,
            chunk_size_bytes=batch.chunk_size_bytes,
        )
        with maybe_nvtx_range(marker):
            self._run_batch_body(batch)

    def _worker_loop(self) -> None:
        try:
            self._device_module.set_device(self.device_index)
            with (
                torch.inference_mode(False),
                torch.no_grad(),
                self._device_module.stream(self.stream),
            ):
                while True:
                    batch = self._next_job()
                    if batch is None:
                        return
                    try:
                        self._run_batch(batch)
                    # Fail the handle for any submission/runtime error raised at
                    # this worker-thread boundary.
                    except Exception as error:  # noqa: BLE001
                        # Retain the batch tensors until every descriptor issued
                        # before the failure has stopped using their storage.
                        try:
                            self._fence_inflight(force=True)
                        except RuntimeError as fence_error:
                            error.add_note(
                                f"Failed to fence cooperative H2D work: {fence_error}"
                            )
                        batch.handle._set_error(error)
                        with self._condition:
                            self._stopping = True
                            self._condition.notify_all()
                        return
                    finally:
                        with self._condition:
                            self._copy_active = False
                            self._condition.notify_all()
        finally:
            with self._condition:
                self._stopping = True
                while self._jobs:
                    self._jobs.popleft().handle._set_error(
                        RuntimeError("Cooperative prefetch scheduler stopped")
                    )
                self._condition.notify_all()


_REGISTRY_LOCK = threading.Lock()
_SCHEDULERS: dict[tuple[int, int], CooperativePrefetchScheduler] = {}


def get_cooperative_prefetch_scheduler(
    device_index: int, queue_depth: int
) -> CooperativePrefetchScheduler:
    key = (os.getpid(), int(device_index))
    with _REGISTRY_LOCK:
        scheduler = _SCHEDULERS.get(key)
        if scheduler is None:
            scheduler = CooperativePrefetchScheduler(device_index, queue_depth)
            _SCHEDULERS[key] = scheduler
        elif scheduler.queue_depth != queue_depth:
            raise ValueError(
                "All cooperative offload managers on one CUDA device must use "
                f"the same queue depth: {scheduler.queue_depth} != {queue_depth}"
            )
        return scheduler


def get_active_cooperative_prefetch_scheduler():
    # Keep the default-disabled USP path to a single dictionary check, without
    # touching the CUDA runtime or taking a lock for every collective.
    if not _SCHEDULERS:
        return None
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    key = (os.getpid(), torch.cuda.current_device())
    with _REGISTRY_LOCK:
        return _SCHEDULERS.get(key)


def shutdown_cooperative_prefetch_schedulers(wait: bool = True) -> None:
    """Stop cooperative workers owned by this process after fencing H2D work."""
    pid = os.getpid()
    with _REGISTRY_LOCK:
        schedulers = [
            _SCHEDULERS.pop(key) for key in list(_SCHEDULERS) if key[0] == pid
        ]
    for scheduler in schedulers:
        scheduler.shutdown(wait=wait)


@contextmanager
def cooperative_a2a_gate():
    """Drain cooperative H2D work around one USP all-to-all operation."""
    scheduler = get_active_cooperative_prefetch_scheduler()
    if scheduler is None:
        yield
        return

    generation = scheduler.begin_a2a()
    try:
        yield
    finally:
        try:
            completion_event = scheduler.record_a2a_completion()
        except Exception:
            scheduler.abort_a2a(generation)
            raise
        scheduler.end_a2a(generation, completion_event)
