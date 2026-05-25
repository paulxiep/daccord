from pathlib import Path
from typing import Self

import yaml
from pydantic import HttpUrl, model_validator

from daccord.validation import ValidatedModel


class Source(ValidatedModel):
    framework: str
    jurisdiction: str
    filename: str
    description: str
    url: HttpUrl | None = None
    manual: bool = False

    @model_validator(mode="after")
    def _exactly_one_of_url_or_manual(self) -> Self:
        if self.manual and self.url is not None:
            raise ValueError(
                f"{self.framework}/{self.filename}: 'url' and 'manual=true' are mutually exclusive"
            )
        if not self.manual and self.url is None:
            raise ValueError(
                f"{self.framework}/{self.filename}: must set either 'url' or 'manual: true'"
            )
        return self


class SourcesSpec(ValidatedModel):
    sources: list[Source]

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def filter_frameworks(self, frameworks: list[str] | None) -> Self:
        if not frameworks:
            return self
        wanted = set(frameworks)
        return type(self)(sources=[s for s in self.sources if s.framework in wanted])
