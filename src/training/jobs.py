"""Background training jobs for the dashboard.

Training runs in a separate Python process (``python -m src.training.experiment``)
so Streamlit reruns, browser refreshes or closing the tab never kill or block it.

* **One job at a time.** ``launch_job`` refuses to start while another experiment
  is active, and an exclusive lock file stops two browser sessions racing.
* **Cancellation** is cooperative: the UI writes ``cancel.flag`` and the trainer stops
  at the next batch, keeping the last epoch's checkpoint. ``force_stop`` terminates the
  process if it is unresponsive.
* **Crash detection.** If status says active but the recorded PID is gone, the
  experiment is marked ``interrupted`` (resumable when a checkpoint exists).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

from src.persistence.storage import ACTIVE_STATES, ExperimentStore, utc_now
from src.utils.config import PROJECT_ROOT, outputs_dir

LOCK_STALE_SECONDS = 60


class JobConflictError(RuntimeError):
    """Another training job is already active."""


def is_process_alive(pid: int | None) -> bool:
    """Portable liveness check that never signals the process.

    (On Windows ``os.kill(pid, 0)`` would *terminate* the process, so the Win32 API is used.)
    """
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def job_pid(store: ExperimentStore, exp_id: str) -> int | None:
    status = store.read_status(exp_id)
    job = store.read_job(exp_id) or {}
    # job.json is written by whoever launched the process and is authoritative;
    # status.json may still hold the PID of a previous (resumed) run.
    return job.get("pid") or status.get("pid")


def reconcile(store: ExperimentStore) -> list[str]:
    """Mark experiments whose process died without reaching a terminal state. Returns their IDs."""
    interrupted = []
    for exp_id in store.experiment_ids():
        status = store.read_status(exp_id)
        if status.get("state") not in ACTIVE_STATES:
            continue
        job = store.read_job(exp_id)
        if job is None:
            continue  # created but never launched (e.g. CLI run in progress records its own pid)
        pid = job.get("pid") or status.get("pid")
        if not is_process_alive(pid):
            store.update_status(
                exp_id, state="interrupted", finished_at=utc_now(),
                error="Training process exited unexpectedly (crash, reboot or forced stop). See train.log / job_stdout.log.",
                resumable=store.paths(exp_id).checkpoint.exists(),
            )
            interrupted.append(exp_id)
    return interrupted


def active_job(store: ExperimentStore) -> str | None:
    reconcile(store)
    for exp_id in store.experiment_ids():
        if store.read_status(exp_id).get("state") in ACTIVE_STATES and store.read_job(exp_id) is not None:
            return exp_id
    return None


class _LaunchLock:
    """Exclusive lock file created with O_EXCL; stale locks (crashed launcher) are broken."""

    def __init__(self) -> None:
        self.path = outputs_dir() / "jobs" / "launch.lock"

    def __enter__(self) -> "_LaunchLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() - self.path.stat().st_mtime > LOCK_STALE_SECONDS:
                self.path.unlink(missing_ok=True)
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                raise JobConflictError("another job launch is in progress; try again in a moment")
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.path.unlink(missing_ok=True)


def launch_job(store: ExperimentStore, exp_id: str, resume: bool = False, device: str = "auto") -> int:
    """Start training in a detached child process and return its PID."""
    store.paths(exp_id)  # validates id / existence
    with _LaunchLock():
        running = active_job(store)
        if running:
            raise JobConflictError(f"experiment {running} is still running; wait for it or cancel it first")
        cmd = [sys.executable, "-m", "src.training.experiment", "--experiment-id", exp_id, "--device", device]
        if resume:
            cmd.append("--resume")
        paths = store.paths(exp_id)
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        store.clear_cancel(exp_id)
        if resume:
            store.update_status(exp_id, state="queued", error=None, pid=None)
        with open(paths.stdout_log, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUNBUFFERED": "1"}, **kwargs)
        store.write_job(exp_id, {"pid": proc.pid, "command": cmd, "launched_at": utc_now(), "resume": resume})
        return proc.pid


def request_cancel(store: ExperimentStore, exp_id: str) -> None:
    store.request_cancel(exp_id)


def force_stop(store: ExperimentStore, exp_id: str) -> bool:
    """Terminate an unresponsive job. Returns True if a live process was stopped."""
    pid = job_pid(store, exp_id)
    if not is_process_alive(pid):
        reconcile(store)
        return False
    assert pid is not None
    os.kill(pid, signal.SIGTERM)  # TerminateProcess on Windows
    for _ in range(50):
        if not is_process_alive(pid):
            break
        time.sleep(0.1)
    store.update_status(exp_id, state="cancelled", finished_at=utc_now(), error="Force-stopped from the dashboard",
                        resumable=store.paths(exp_id).checkpoint.exists())
    store.clear_cancel(exp_id)
    return True
