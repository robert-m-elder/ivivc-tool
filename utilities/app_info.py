"""Application version, source revision, and support-contact helpers."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlencode

CONTACT_EMAIL = "robert.elder@fda.hhs.gov"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = PROJECT_ROOT / "VERSION"


def _read_packaged_version() -> str:
    """Read the packaged application version, with a conservative fallback."""
    configured = os.getenv("IVIVC_APP_VERSION", "").strip()
    if configured:
        return configured

    try:
        version = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        version = ""
    return version or "unknown"


def _configured_revision() -> str:
    """Return a revision supplied by a deployment platform, when available."""
    for variable in ("IVIVC_GIT_COMMIT", "GITHUB_SHA", "COMMIT_SHA", "SOURCE_VERSION"):
        value = os.getenv(variable, "").strip()
        if value:
            return value[:12]
    return ""


def _git_revision() -> str:
    """Return the local Git revision, including a dirty-worktree marker."""
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        if not revision:
            return ""

        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def get_app_info() -> dict[str, str]:
    """Return display-ready application and support-contact information."""
    version = _read_packaged_version()
    revision = _configured_revision() or _git_revision()
    revision_display = revision or "Not available in packaged build"
    display = f"{version} ({revision})" if revision else version

    subject = f"IVIVC App feedback - version {display}"
    body = "\n".join(
        [
            f"App version: {version}",
            f"Source revision: {revision_display}",
            "Issue type: Bug report / feature request",
            "",
            "What happened or what would you like added?",
            "",
            "Steps to reproduce, if applicable:",
            "",
            "Expected result:",
            "",
            "Please do not include sensitive or confidential data in this email.",
        ]
    )
    mailto_query = urlencode(
        {'subject': subject, 'body': body},
        quote_via=quote,
    )
    mailto = f"mailto:{CONTACT_EMAIL}?{mailto_query}"

    return {
        "version": version,
        "revision": revision_display,
        "revision_raw": revision,
        "display": display,
        "contact_email": CONTACT_EMAIL,
        "contact_mailto": mailto,
    }
