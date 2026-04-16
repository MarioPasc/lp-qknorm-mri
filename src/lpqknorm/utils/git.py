"""Git state capture for reproducibility.

Captures the current commit SHA, branch name, dirty status, and diff text
from the repository so that every training run can be traced back to an
exact code state. Uses ``subprocess`` to call the ``git`` binary directly
(GitPython is not a project dependency).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitState:
    """Snapshot of the repository git state.

    Parameters
    ----------
    sha : str
        Full commit SHA, or ``"UNKNOWN"`` if not in a git repo.
    branch : str
        Current branch name, or ``"UNKNOWN"``.
    dirty : bool
        ``True`` if the working tree has uncommitted changes.
    diff : str
        Output of ``git diff HEAD`` (empty string if clean).
    """

    sha: str
    branch: str
    dirty: bool
    diff: str


def capture_git_state(repo_root: str | None = None) -> GitState:
    """Capture the current git state of the repository.

    Parameters
    ----------
    repo_root : str or None
        Path to the repository root.  If ``None``, uses the current working
        directory.

    Returns
    -------
    GitState
        Frozen dataclass with ``sha``, ``branch``, ``dirty``, ``diff``.

    Notes
    -----
    Returns sentinel values (``"UNKNOWN"``, ``False``, ``""``) if ``git``
    is unavailable or the directory is not a git repository.
    """

    def _run(cmd: list[str]) -> str:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_root,
                timeout=10,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    sha = _run(["git", "rev-parse", "HEAD"]) or "UNKNOWN"
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "UNKNOWN"
    diff = _run(["git", "diff", "HEAD"])
    dirty = bool(diff) or bool(_run(["git", "status", "--porcelain"]))

    state = GitState(sha=sha, branch=branch, dirty=dirty, diff=diff)
    logger.info(
        "Git state: sha=%s, branch=%s, dirty=%s",
        sha[:8],
        branch,
        dirty,
    )
    return state
