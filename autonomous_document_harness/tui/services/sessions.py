"""Session persistence: one JSON document per session, OS app-data dir.

Persists document references (+fingerprints), conversation turns, contract
ledger, evidence references, metrics and exports. NEVER secrets:
stored payloads are scanned for secret values on save and redacted.
"""

from __future__ import annotations

import datetime
import json
import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_NAME = "kynex"
_OLD_APP_NAME = "harness"


def sessions_dir() -> Path:
    try:
        from platformdirs import user_data_dir

        root = Path(user_data_dir(APP_NAME, appauthor=False))
    except ImportError:
        root = Path.home() / f".{APP_NAME}"
    if not root.exists():
        try:
            from platformdirs import user_data_dir as _udd

            old_root = Path(_udd(_OLD_APP_NAME, appauthor=False))
        except ImportError:
            old_root = Path.home() / f".{_OLD_APP_NAME}"
        if old_root.exists():
            import shutil

            shutil.copytree(old_root, root)
    root.mkdir(parents=True, exist_ok=True)
    sessions = root / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_session_id() -> str:
    words = (
        "amber",
        "brisk",
        "clear",
        "delta",
        "ember",
        "flint",
        "grove",
        "harbor",
        "iron",
        "juniper",
        "kestrel",
        "lumen",
        "meadow",
        "north",
        "onyx",
        "prairie",
        "quartz",
        "river",
        "stone",
        "timber",
    )
    return f"{secrets.choice(words)}-{secrets.choice(words)}"


class SessionService:
    """CRUD + persistence for harness sessions."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = Path(directory) if directory else sessions_dir()

    # -- lifecycle ------------------------------------------------------
    def create(self, name: str = "", workspace_name: str = "") -> Dict[str, Any]:
        session = {
            "id": new_session_id(),
            "name": name or "",
            "workspace_name": workspace_name or "",
            "created_utc": _utcnow(),
            "updated_utc": _utcnow(),
            "documents": [],
            "turns": [],
            "ledger": None,
            "metrics": {},
            "exports": [],
            "thread_ids": [],
        }
        self.save(session)
        return session

    def path_for(self, session_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id)[:64] or "session"
        return self.directory / f"{safe}.json"

    def save(self, session: Dict[str, Any]) -> Path:
        from ..sanitize import scrub_secrets

        session["updated_utc"] = _utcnow()
        payload = json.dumps(session, indent=2, default=str)
        payload = scrub_secrets(payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session["id"])
        path.write_text(payload, encoding="utf-8")
        return path

    def load(self, session_id: str) -> Dict[str, Any]:
        path = self.path_for(session_id)
        if not path.exists():
            # Allow lookup by name as well.
            for candidate in self.list():
                if (
                    candidate.get("name") == session_id
                    or candidate.get("id") == session_id
                ):
                    return candidate
            raise FileNotFoundError(f"unknown session '{session_id}'")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"session checkpoint '{session_id}' is corrupted: {exc}")

    def delete(self, session_id: str) -> bool:
        session = self.load(session_id)
        self.path_for(session["id"]).unlink(missing_ok=True)
        return True

    def rename(self, session_id: str, name: str) -> Dict[str, Any]:
        session = self.load(session_id)
        session["name"] = name
        self.save(session)
        return session

    def list(self) -> List[Dict[str, Any]]:
        sessions = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                sessions.append(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        sessions.sort(key=lambda s: s.get("updated_utc", ""), reverse=True)
        return sessions

    def most_recent(self) -> Optional[Dict[str, Any]]:
        sessions = self.list()
        return sessions[0] if sessions else None

    def search(self, query: str) -> List[Dict[str, Any]]:
        needle = query.lower()
        hits = []
        for session in self.list():
            haystack = json.dumps(session, default=str).lower()
            if needle in haystack:
                hits.append(session)
        return hits
