import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from daccord.validation import ValidatedModel, validated

MANIFEST_SCHEMA_VERSION = 1


class ManifestEntry(ValidatedModel):
    framework: str
    jurisdiction: str
    filename: str
    source_url: str | None
    local_path: str
    sha256: str
    content_length: int
    retrieved_at: datetime
    manual: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.framework, self.filename)


class Manifest(ValidatedModel):
    schema_version: int = MANIFEST_SCHEMA_VERSION
    generated_at: datetime
    entries: list[ManifestEntry]

    @classmethod
    def load(cls, path: Path) -> Self:
        if not path.exists():
            return cls(generated_at=datetime.now(UTC), entries=[])
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        sorted_entries = sorted(self.entries, key=lambda e: e.key)
        payload = self.model_copy(update={"entries": sorted_entries})
        text = json.dumps(
            payload.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    @validated
    def upsert(self, entry: ManifestEntry) -> None:
        self.entries = [e for e in self.entries if e.key != entry.key] + [entry]

    @validated
    def find(self, framework: str, filename: str) -> ManifestEntry | None:
        for e in self.entries:
            if e.framework == framework and e.filename == filename:
                return e
        return None
