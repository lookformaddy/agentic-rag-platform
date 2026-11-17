"""SHA-256 hash-chained audit log (tamper-evident, append-only).

Every request, retrieval round, tool call, policy change and eval run
appends an entry whose hash commits to the previous entry's hash, so any
later edit, deletion or reordering is detectable by ``verify()``.
"""
from __future__ import annotations

import hashlib
import json

from ragagent.models import AuditEntry


def _entry_hash(seq: int, event: str, request_id: str, tenant: str,
                detail: dict, prev_hash: str) -> str:
    payload = json.dumps({
        "seq": seq, "event": event, "request_id": request_id,
        "tenant": tenant, "detail": detail, "prev_hash": prev_hash,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only hash-chained audit log."""

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def append(self, event: str, request_id: str, tenant: str,
               detail: dict | None = None) -> AuditEntry:
        detail = detail or {}
        prev_hash = self.entries[-1].hash if self.entries else "GENESIS"
        entry = AuditEntry(
            seq=len(self.entries) + 1, event=event, request_id=request_id,
            tenant=tenant, detail=detail, prev_hash=prev_hash,
            hash=_entry_hash(len(self.entries) + 1, event, request_id,
                             tenant, detail, prev_hash))
        self.entries.append(entry)
        return entry

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain. Returns (ok, message)."""
        prev = "GENESIS"
        for entry in self.entries:
            if entry.prev_hash != prev:
                return False, f"entry {entry.seq}: prev_hash mismatch"
            expected = _entry_hash(entry.seq, entry.event, entry.request_id,
                                   entry.tenant, entry.detail, entry.prev_hash)
            if entry.hash != expected:
                return False, f"entry {entry.seq}: hash mismatch (tampered)"
            prev = entry.hash
        return True, f"chain valid ({len(self.entries)} entries)"

    def events(self, event: str | None = None) -> list[AuditEntry]:
        if event is None:
            return list(self.entries)
        return [e for e in self.entries if e.event == event]
