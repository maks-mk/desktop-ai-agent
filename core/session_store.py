import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_CHAT_TITLE = "New Chat"
INDEX_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_project_path(path: str | Path | None = None) -> str:
    base = Path(path) if path is not None else Path.cwd()
    return str(base.resolve())


@dataclass
class SessionListEntry:
    session_id: str
    thread_id: str
    project_path: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class SessionSnapshot:
    session_id: str
    thread_id: str
    checkpoint_backend: str
    checkpoint_target: str
    created_at: str
    updated_at: str
    project_path: str
    approval_mode: str = "prompt"
    title: str = DEFAULT_CHAT_TITLE

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_list_entry(self) -> SessionListEntry:
        return SessionListEntry(
            session_id=self.session_id,
            thread_id=self.thread_id,
            project_path=self.project_path,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class SessionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path = self.path.parent / "session_index.json"

    def _read_json(self, path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text("utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _coerce_snapshot(self, payload: dict | None) -> Optional[SessionSnapshot]:
        if not isinstance(payload, dict):
            return None
        try:
            payload = dict(payload)
            payload.setdefault("approval_mode", "prompt")
            payload["project_path"] = normalize_project_path(payload.get("project_path"))
            payload["title"] = str(payload.get("title") or DEFAULT_CHAT_TITLE).strip() or DEFAULT_CHAT_TITLE
            return SessionSnapshot(**payload)
        except TypeError:
            return None

    def _load_active_session_file(self) -> Optional[SessionSnapshot]:
        return self._coerce_snapshot(self._read_json(self.path))

    def _default_index_payload(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "sessions": [],
            "active_session_by_project": {},
            "last_active_session_id": "",
        }

    def _load_index_payload(self) -> dict:
        payload = self._read_json(self.index_path) or self._default_index_payload()
        payload.setdefault("version", INDEX_VERSION)
        payload.setdefault("sessions", [])
        payload.setdefault("active_session_by_project", {})
        payload.setdefault("last_active_session_id", "")
        return payload

    def _save_index_payload(self, payload: dict) -> None:
        payload["version"] = INDEX_VERSION
        payload["sessions"] = list(payload.get("sessions", []))
        payload["active_session_by_project"] = dict(payload.get("active_session_by_project", {}))
        payload["last_active_session_id"] = str(payload.get("last_active_session_id") or "")
        self._write_json(self.index_path, payload)

    def _all_snapshots_from_index(self) -> list[SessionSnapshot]:
        payload = self._load_index_payload()
        snapshots: list[SessionSnapshot] = []
        for raw in payload.get("sessions", []):
            snapshot = self._coerce_snapshot(raw)
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def _ensure_index_initialized(self) -> None:
        payload = self._load_index_payload()
        sessions = payload.get("sessions", [])
        known_ids = {
            str(entry.get("session_id"))
            for entry in sessions
            if isinstance(entry, dict) and entry.get("session_id")
        }
        active_snapshot = self._load_active_session_file()
        changed = not self.index_path.exists()
        if active_snapshot is not None and active_snapshot.session_id not in known_ids:
            sessions.append(asdict(active_snapshot))
            changed = True
        if active_snapshot is not None:
            active_map = dict(payload.get("active_session_by_project", {}))
            project_key = normalize_project_path(active_snapshot.project_path)
            if active_map.get(project_key) != active_snapshot.session_id:
                active_map[project_key] = active_snapshot.session_id
                payload["active_session_by_project"] = active_map
                changed = True
            if payload.get("last_active_session_id") != active_snapshot.session_id:
                payload["last_active_session_id"] = active_snapshot.session_id
                changed = True
        payload["sessions"] = sessions
        if changed:
            self._save_index_payload(payload)

    def _upsert_session(self, snapshot: SessionSnapshot, *, set_active: bool, touch: bool) -> None:
        self._ensure_index_initialized()
        if touch:
            snapshot.touch()
        snapshot.project_path = normalize_project_path(snapshot.project_path)
        snapshot.title = str(snapshot.title or DEFAULT_CHAT_TITLE).strip() or DEFAULT_CHAT_TITLE

        payload = self._load_index_payload()
        sessions: list[dict] = []
        replaced = False
        for raw in payload.get("sessions", []):
            existing = self._coerce_snapshot(raw)
            if existing is None:
                continue
            if existing.session_id == snapshot.session_id:
                sessions.append(asdict(snapshot))
                replaced = True
            else:
                sessions.append(asdict(existing))
        if not replaced:
            sessions.append(asdict(snapshot))

        payload["sessions"] = sessions
        if set_active:
            active_map = dict(payload.get("active_session_by_project", {}))
            active_map[normalize_project_path(snapshot.project_path)] = snapshot.session_id
            payload["active_session_by_project"] = active_map
            payload["last_active_session_id"] = snapshot.session_id
        self._save_index_payload(payload)

    def load_active_session(self) -> Optional[SessionSnapshot]:
        self._ensure_index_initialized()
        return self._load_active_session_file()

    def save_active_session(
        self,
        snapshot: SessionSnapshot,
        *,
        touch: bool = True,
        set_active: bool = True,
    ) -> None:
        snapshot.project_path = normalize_project_path(snapshot.project_path)
        snapshot.title = str(snapshot.title or DEFAULT_CHAT_TITLE).strip() or DEFAULT_CHAT_TITLE
        if touch:
            snapshot.touch()
        self._write_json(self.path, asdict(snapshot))
        self._upsert_session(snapshot, set_active=set_active, touch=False)

    def list_sessions(self, project_path: str | Path | None = None) -> list[SessionListEntry]:
        self._ensure_index_initialized()
        snapshots = self._all_snapshots_from_index()
        if project_path is None:
            matches = [snapshot.to_list_entry() for snapshot in snapshots]
        else:
            project_key = normalize_project_path(project_path)
            matches = [
                snapshot.to_list_entry()
                for snapshot in snapshots
                if normalize_project_path(snapshot.project_path) == project_key
            ]
        return sorted(matches, key=lambda item: (item.updated_at, item.created_at, item.session_id), reverse=True)

    def get_session(self, session_id: str) -> Optional[SessionSnapshot]:
        self._ensure_index_initialized()
        for snapshot in self._all_snapshots_from_index():
            if snapshot.session_id == session_id:
                return snapshot
        return None

    def get_active_session_for_project(self, project_path: str | Path | None = None) -> Optional[SessionSnapshot]:
        self._ensure_index_initialized()
        project_key = normalize_project_path(project_path)
        payload = self._load_index_payload()
        active_map = payload.get("active_session_by_project", {})
        session_id = active_map.get(project_key)
        if session_id:
            session = self.get_session(str(session_id))
            if session is not None and normalize_project_path(session.project_path) == project_key:
                return session
        sessions = self.list_sessions(project_key)
        if not sessions:
            return None
        return self.get_session(sessions[0].session_id)

    def get_last_active_session(self) -> Optional[SessionSnapshot]:
        self._ensure_index_initialized()
        payload = self._load_index_payload()
        last_active_session_id = str(payload.get("last_active_session_id") or "").strip()
        if last_active_session_id:
            session = self.get_session(last_active_session_id)
            if session is not None:
                return session

        sessions = self.list_sessions()
        if not sessions:
            return None
        return self.get_session(sessions[0].session_id)

    def update_session_title(self, session_id: str, title: str) -> Optional[SessionSnapshot]:
        snapshot = self.get_session(session_id)
        if snapshot is None:
            return None
        snapshot.title = str(title or DEFAULT_CHAT_TITLE).strip() or DEFAULT_CHAT_TITLE
        active = self._load_active_session_file()
        if active is not None and active.session_id == session_id:
            active.title = snapshot.title
            self.save_active_session(active, touch=False, set_active=True)
            return active
        self._upsert_session(snapshot, set_active=False, touch=False)
        return snapshot

    def delete_session(self, session_id: str) -> bool:
        if not session_id:
            return False

        self._ensure_index_initialized()
        payload = self._load_index_payload()
        sessions = payload.get("sessions", [])

        remaining_raw: list[dict] = []
        removed_any = False
        for raw in sessions:
            snapshot = self._coerce_snapshot(raw)
            if snapshot is None:
                continue
            if snapshot.session_id == session_id:
                removed_any = True
                continue
            remaining_raw.append(asdict(snapshot))

        if not removed_any:
            return False

        payload["sessions"] = remaining_raw
        remaining_snapshots = [snapshot for snapshot in (self._coerce_snapshot(raw) for raw in remaining_raw) if snapshot is not None]
        remaining_snapshots.sort(
            key=lambda snapshot: (snapshot.updated_at, snapshot.created_at, snapshot.session_id),
            reverse=True,
        )

        active_map = {
            project: sid
            for project, sid in dict(payload.get("active_session_by_project", {})).items()
            if str(sid) != session_id
        }
        payload["active_session_by_project"] = active_map

        if str(payload.get("last_active_session_id") or "") == session_id:
            payload["last_active_session_id"] = remaining_snapshots[0].session_id if remaining_snapshots else ""

        self._save_index_payload(payload)

        active_snapshot = self._load_active_session_file()
        if active_snapshot is not None and active_snapshot.session_id == session_id:
            replacement = remaining_snapshots[0] if remaining_snapshots else None
            if replacement is not None:
                self._write_json(self.path, asdict(replacement))
            else:
                try:
                    self.path.unlink(missing_ok=True)
                except Exception:
                    pass

        return True

    def new_session(
        self,
        checkpoint_backend: str,
        checkpoint_target: str,
        *,
        project_path: str | Path | None = None,
        title: str = DEFAULT_CHAT_TITLE,
    ) -> SessionSnapshot:
        session_id = uuid.uuid4().hex
        now = _utc_now()
        return SessionSnapshot(
            session_id=session_id,
            thread_id=f"session_{session_id}",
            checkpoint_backend=checkpoint_backend,
            checkpoint_target=checkpoint_target,
            created_at=now,
            updated_at=now,
            project_path=normalize_project_path(project_path),
            title=str(title or DEFAULT_CHAT_TITLE).strip() or DEFAULT_CHAT_TITLE,
        )
