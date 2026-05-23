import pytest
from pydantic import ValidationError

from daccord.validation import ValidatedModel, validated


@validated
def _add(x: int, y: int) -> int:
    return x + y


def test_validated_accepts_correct_types() -> None:
    assert _add(2, 3) == 5


def test_validated_rejects_wrong_arg_type() -> None:
    with pytest.raises(ValidationError):
        _add("two", 3)  # type: ignore[arg-type]


def test_validated_model_validates_on_construction() -> None:
    class Mapping(ValidatedModel):
        source_jurisdiction: str
        article_id: str

    m = Mapping(source_jurisdiction="GDPR", article_id="Art. 32")
    assert m.source_jurisdiction == "GDPR"

    with pytest.raises(ValidationError):
        Mapping(source_jurisdiction=123, article_id="Art. 32")  # type: ignore[arg-type]
