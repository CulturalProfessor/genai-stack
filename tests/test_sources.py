import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.sources import RepoRef, clone_repository, parse_repo_url


def test_accepts_a_plain_https_url():
    ref = parse_repo_url("https://github.com/owner/repo")
    assert (ref.host, ref.owner, ref.name) == ("github.com", "owner", "repo")
    assert ref.clone_url == "https://github.com/owner/repo.git"


def test_accepts_a_dot_git_url():
    assert parse_repo_url("https://github.com/owner/repo.git").name == "repo"


def test_accepts_a_host_relative_url():
    assert parse_repo_url("github.com/owner/repo").owner == "owner"


def test_accepts_owner_slash_repo_shorthand():
    ref = parse_repo_url("owner/repo")
    assert ref.slug == "owner/repo"
    assert ref.host == "github.com"


def test_reads_the_branch_out_of_a_browser_url():
    assert parse_repo_url("https://github.com/owner/repo/tree/develop").branch == "develop"


def test_explicit_branch_wins_over_the_url():
    ref = parse_repo_url("https://github.com/owner/repo/tree/develop", branch="main")
    assert ref.branch == "main"


def test_trailing_path_segments_are_ignored():
    ref = parse_repo_url("https://github.com/owner/repo/blob/main/src/index.ts")
    assert (ref.owner, ref.name) == ("owner", "repo")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "not a url",
        "https://github.com/owner",
    ],
)
def test_rejects_incomplete_input(value):
    with pytest.raises(ValueError):
        parse_repo_url(value)


def test_ssh_url_says_so_instead_of_blaming_credentials():
    with pytest.raises(ValueError, match="SSH URLs are not supported"):
        parse_repo_url("git@github.com:owner/repo.git")


@pytest.mark.parametrize(
    "value",
    [
        "ssh://git@github.com/owner/repo",
        "file:///etc/passwd",
        "http://github.com/owner/repo",
    ],
)
def test_rejects_non_https_schemes(value):
    """SSH and local paths would read keys or files from the container."""
    with pytest.raises(ValueError):
        parse_repo_url(value)


def test_rejects_hosts_outside_the_allowlist():
    with pytest.raises(ValueError, match="not allowed"):
        parse_repo_url("https://evil.example/owner/repo")


def test_rejects_credentials_in_the_url():
    with pytest.raises(ValueError, match="credentials"):
        parse_repo_url("https://user:token@github.com/owner/repo")


def test_rejects_leading_dash():
    """A leading dash would be read by git as an option, not a URL."""
    with pytest.raises(ValueError, match="may not start"):
        parse_repo_url("--upload-pack=touch /tmp/pwned")


def test_allowlist_is_configurable(monkeypatch):
    monkeypatch.setenv("ALLOWED_GIT_HOSTS", "git.internal.example")
    assert parse_repo_url("https://git.internal.example/o/r").host == "git.internal.example"
    with pytest.raises(ValueError):
        parse_repo_url("https://github.com/o/r")


def test_clone_command_terminates_options_and_disables_prompts(monkeypatch):
    """The URL must sit after `--`, and git must never wait on a password."""
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = RepoRef(host="github.com", owner="o", name="r", branch="dev")
    clone_repository(ref, Path("/tmp/does-not-matter"))

    command = captured["command"]
    assert command[:2] == ["git", "clone"]
    assert "--depth" in command and "--single-branch" in command
    assert command[command.index("--branch") + 1] == "dev"
    assert command[command.index("--") + 1] == ref.clone_url
    assert captured["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_failure_explains_a_private_repo(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 128, "", "fatal: could not read Username for 'https://github.com'"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="private or does not exist"):
        clone_repository(RepoRef("github.com", "o", "r"), Path("/tmp/x"))


def test_clone_failure_explains_a_missing_branch(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 128, "", "fatal: Remote branch nope not found in upstream origin"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Branch 'nope' does not exist"):
        clone_repository(RepoRef("github.com", "o", "r", branch="nope"), Path("/tmp/x"))
