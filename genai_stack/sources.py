"""Fetching a repository to index from a URL.

Only clones and parses - nothing from the cloned repository is ever executed.
No install step, no build step, no lifecycle scripts.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

# Hosts a URL may point at. Override for self-hosted GitHub/GitLab.
DEFAULT_ALLOWED_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")

# owner/repo, each segment restricted to what these hosts actually allow.
_SEGMENT = r"[A-Za-z0-9._-]+"
_SHORTHAND_RE = re.compile(rf"^(?P<owner>{_SEGMENT})/(?P<name>{_SEGMENT})$")

CLONE_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class RepoRef:
    """A validated, cloneable repository reference."""

    host: str
    owner: str
    name: str
    branch: str | None = None

    @property
    def clone_url(self) -> str:
        return f"https://{self.host}/{self.owner}/{self.name}.git"

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


def allowed_hosts() -> tuple[str, ...]:
    configured = os.getenv("ALLOWED_GIT_HOSTS", "")
    if not configured:
        return DEFAULT_ALLOWED_HOSTS
    return tuple(h.strip().lower() for h in configured.split(",") if h.strip())


def parse_repo_url(value: str, branch: str | None = None) -> RepoRef:
    """Turn user input into a RepoRef, or raise ValueError explaining why not.

    Accepts `owner/repo`, `github.com/owner/repo`, a full https URL with or
    without `.git`, and a `/tree/<branch>` browser URL.
    """
    value = (value or "").strip()
    if not value:
        raise ValueError("Enter a repository URL")

    # A leading dash would be read by git as an option, not a URL.
    if value.startswith("-"):
        raise ValueError("Repository URL may not start with '-'")

    hosts = allowed_hosts()

    shorthand = _SHORTHAND_RE.match(value)
    if shorthand:
        return RepoRef(
            host=hosts[0],
            owner=shorthand["owner"],
            name=_strip_git_suffix(shorthand["name"]),
            branch=branch or None,
        )

    candidate = value if "://" in value else f"https://{value}"
    parsed = urlparse(candidate)

    if parsed.scheme != "https":
        raise ValueError(
            f"Only https:// URLs are supported (got '{parsed.scheme or value}'). "
            "SSH and local paths are not accepted here."
        )
    # Credentials in the URL would be written to the container's git config
    # and shown back in error messages.
    if parsed.username or parsed.password:
        raise ValueError("Remove credentials from the URL")

    host = (parsed.hostname or "").lower()
    if host not in hosts:
        raise ValueError(
            f"Host '{host or value}' is not allowed. Allowed: {', '.join(hosts)}. "
            "Set ALLOWED_GIT_HOSTS to change this."
        )

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("URL must include an owner and a repository name")

    owner, name = parts[0], _strip_git_suffix(parts[1])
    if not _SEGMENT_RE_FULL.match(owner) or not _SEGMENT_RE_FULL.match(name):
        raise ValueError("Owner and repository name contain unsupported characters")

    # https://github.com/owner/repo/tree/some-branch
    url_branch = None
    if len(parts) >= 4 and parts[2] in ("tree", "src"):
        url_branch = "/".join(parts[3:])

    return RepoRef(host=host, owner=owner, name=name, branch=branch or url_branch)


_SEGMENT_RE_FULL = re.compile(rf"^{_SEGMENT}$")


def _strip_git_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".git") else name


def clone_repository(ref: RepoRef, destination: Path) -> Path:
    """Shallow-clone `ref` into `destination`. Returns the checkout path."""
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / ref.name

    command = ["git", "clone", "--depth", "1", "--single-branch"]
    if ref.branch:
        command += ["--branch", ref.branch]
    # `--` stops git reading anything that follows as an option.
    command += ["--", ref.clone_url, str(target)]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=CLONE_TIMEOUT_SECONDS,
        env={
            **os.environ,
            # Without this a private or misspelled repo blocks forever on a
            # username prompt instead of failing.
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "",
            "GCM_INTERACTIVE": "never",
        },
    )
    if result.returncode != 0:
        raise RuntimeError(_clone_error_message(ref, result.stderr))
    return target


def _clone_error_message(ref: RepoRef, stderr: str) -> str:
    detail = (stderr or "").strip().splitlines()
    tail = detail[-1] if detail else "unknown error"
    if "Authentication failed" in stderr or "could not read Username" in stderr:
        return (
            f"Could not read {ref.slug}: it is private or does not exist. "
            "Only public repositories are supported."
        )
    if "not found in upstream origin" in stderr or "Remote branch" in stderr:
        return f"Branch '{ref.branch}' does not exist in {ref.slug}."
    return f"Could not clone {ref.slug}: {tail}"


@contextmanager
def cloned_repository(ref: RepoRef) -> Iterator[Path]:
    """Clone into a temporary directory and remove it afterwards.

    The checkout is only read by the parser, and a full source tree is large,
    so it is not worth keeping once the index has been built.
    """
    with tempfile.TemporaryDirectory(prefix="genai-stack-clone-") as tmp:
        yield clone_repository(ref, Path(tmp))
