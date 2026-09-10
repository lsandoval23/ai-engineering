"""The data contract: what the schema accepts and, more importantly, rejects."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import CriticalityLevel, TechnicalEntities

VALID = {
    "technologies": ["FastAPI", "Redis"],
    "criticality_level": "high",
    "technical_summary": "The API times out because the Redis cache saturates.",
}


def test_valid_payload_builds_a_typed_instance():
    entities = TechnicalEntities(**VALID)
    assert entities.technologies == ["FastAPI", "Redis"]
    assert entities.criticality_level is CriticalityLevel.HIGH
    assert entities.model_dump()["criticality_level"] == "high"


def test_empty_technologies_list_is_rejected():
    with pytest.raises(ValidationError, match="technologies"):
        TechnicalEntities(**{**VALID, "technologies": []})


def test_blank_only_technologies_are_rejected_by_the_validator():
    """``min_length=1`` is satisfied here; only the semantic validator catches it."""
    with pytest.raises(ValidationError, match="non-blank"):
        TechnicalEntities(**{**VALID, "technologies": ["  ", ""]})


def test_technologies_are_stripped_and_deduplicated_in_order():
    entities = TechnicalEntities(
        **{**VALID, "technologies": [" Redis", "FastAPI", "Redis ", "", "PostgreSQL"]}
    )
    assert entities.technologies == ["Redis", "FastAPI", "PostgreSQL"]


@pytest.mark.parametrize("bad_level", ["HIGH", "alta", "critical", 3])
def test_criticality_outside_the_closed_enum_is_rejected(bad_level):
    with pytest.raises(ValidationError, match="criticality_level"):
        TechnicalEntities(**{**VALID, "criticality_level": bad_level})


def test_short_summary_is_rejected():
    with pytest.raises(ValidationError, match="technical_summary"):
        TechnicalEntities(**{**VALID, "technical_summary": "too short"})


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID.items() if k != "technical_summary"}
    with pytest.raises(ValidationError, match="technical_summary"):
        TechnicalEntities(**payload)
