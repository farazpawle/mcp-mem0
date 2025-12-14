"""Utilities for structured, low-noise "vibe" memories.

This module is intentionally dependency-light so it can be unit-tested without
requiring Mem0 or a running database.

Core idea: store memories in a predictable, searchable mini-schema and mirror
important fields into metadata for filtering/hygiene.

Schema (plain text):
- TYPE: CONVENTION | DECISION | GOTCHA
- SCOPE: repo/module/feature/etc
- TRIGGER: when to recall this
- CONTENT: the rule/decision/gotcha itself
- LAST_VALIDATED: YYYY-MM-DD
- REVIEW_AFTER: YYYY-MM-DD (optional)
- EXPIRES: YYYY-MM-DD (optional)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

MemoryType = Literal["CONVENTION", "DECISION", "GOTCHA"]

ALLOWED_MEMORY_TYPES: set[str] = {"CONVENTION", "DECISION", "GOTCHA"}


@dataclass(frozen=True)
class VibeMemory:
    memory_type: str
    scope: str
    trigger: str | None
    content: str
    last_validated: str
    review_after: str | None = None
    expires: str | None = None


_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}\b", re.IGNORECASE)),
    ("Private key block", re.compile(r"-----BEGIN\s+.*PRIVATE\s+KEY-----")),
    (
        "DB connection string",
        re.compile(r"\b(postgres|postgresql|mysql|mongodb)://[^\s]+", re.IGNORECASE),
    ),
    (
        "Supabase key",
        re.compile(r"\b(anon|service)_key\b\s*[:=]\s*[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
    ),
    (
        "Generic API key assignment",
        re.compile(r"\b(api[_-]?key|token|secret)\b\s*[:=]\s*\S{8,}", re.IGNORECASE),
    ),
]


def today_yyyy_mm_dd() -> str:
    return _dt.date.today().isoformat()


def detect_potential_secret(text: str) -> str | None:
    """Return a human-readable reason if text looks like it contains secrets."""
    if not text:
        return None

    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return label

    return None


def normalize_csv_tags(tags: str | None) -> str | None:
    if not tags:
        return None

    parts = [p.strip() for p in tags.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None

    # Stable ordering improves dedupe and makes tags predictable.
    normalized = sorted(set(parts), key=str.lower)
    return ", ".join(normalized)


def ensure_allowed_type(memory_type: str) -> str:
    value = (memory_type or "").strip().upper()
    if value not in ALLOWED_MEMORY_TYPES:
        raise ValueError(
            f"Invalid memory_type '{memory_type}'. Allowed: {sorted(ALLOWED_MEMORY_TYPES)}"
        )
    return value


def make_vibe_key(
    *,
    memory_type: str,
    scope: str,
    trigger: str | None,
    content: str,
) -> str:
    """Compute a stable key for deduplication/upserts."""
    normalized = "\n".join(
        [
            ensure_allowed_type(memory_type),
            (scope or "").strip().lower(),
            (trigger or "").strip().lower(),
            " ".join((content or "").split()).strip().lower(),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def format_vibe_text(memory: VibeMemory) -> str:
    trigger = memory.trigger.strip() if memory.trigger else "(none)"

    lines: list[str] = [
        f"TYPE: {ensure_allowed_type(memory.memory_type)}",
        f"SCOPE: {memory.scope.strip()}",
        f"TRIGGER: {trigger}",
        f"CONTENT: {memory.content.strip()}",
        f"LAST_VALIDATED: {memory.last_validated}",
    ]

    if memory.review_after:
        lines.append(f"REVIEW_AFTER: {memory.review_after}")

    if memory.expires:
        lines.append(f"EXPIRES: {memory.expires}")

    return "\n".join(lines)


def parse_vibe_text(text: str) -> dict[str, str]:
    """Parse schema fields from a vibe memory text.

    Best-effort: returns only the fields it recognizes.
    """
    if not text:
        return {}

    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in {
            "TYPE",
            "SCOPE",
            "TRIGGER",
            "CONTENT",
            "LAST_VALIDATED",
            "REVIEW_AFTER",
            "EXPIRES",
        }:
            fields[key] = value

    return fields


def build_vibe_memory(
    *,
    memory_type: str,
    scope: str,
    trigger: str | None,
    content: str,
    last_validated: str | None,
    review_after: str | None,
    expires: str | None,
) -> VibeMemory:
    memory_type = ensure_allowed_type(memory_type)
    if not scope or not scope.strip():
        raise ValueError("scope is required")
    if not content or not content.strip():
        raise ValueError("content is required")

    return VibeMemory(
        memory_type=memory_type,
        scope=scope,
        trigger=trigger,
        content=content,
        last_validated=last_validated or today_yyyy_mm_dd(),
        review_after=review_after,
        expires=expires,
    )


def build_vibe_metadata(
    *,
    memory: VibeMemory,
    file_path: str | None,
    tags: str | None,
    schema_version: int = 1,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "schema": "vibe_memory",
        "schema_version": schema_version,
        "memory_type": ensure_allowed_type(memory.memory_type),
        "scope": memory.scope.strip(),
        "trigger": (memory.trigger or "").strip() or None,
        "last_validated": memory.last_validated,
        "review_after": memory.review_after,
        "expires": memory.expires,
    }

    if file_path:
        meta["file_path"] = file_path

    normalized_tags = normalize_csv_tags(tags)
    if normalized_tags:
        meta["tags"] = normalized_tags

    meta["vibe_key"] = make_vibe_key(
        memory_type=memory.memory_type,
        scope=memory.scope,
        trigger=memory.trigger,
        content=memory.content,
    )

    return meta
