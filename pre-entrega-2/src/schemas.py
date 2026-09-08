"""Data contract for the technical-entity extraction pipeline.

The Pydantic model here is what the LLM is forced to fill via
``with_structured_output``: the field names, types and ``description``s are
shown to the model through the provider's tool-calling machinery, so the
descriptions double as prompt instructions.

Three validation layers apply, from the course's taxonomy:

- **syntactic** — the provider guarantees parseable JSON;
- **structural** — required fields, types and the closed enum are enforced by
  the schema itself;
- **semantic** — the ``technologies`` validator applies a business rule the
  schema alone cannot express (whitespace-only entries are not technologies).
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class CriticalityLevel(str, Enum):
    """Closed set of severity values.

    Inheriting from ``str`` keeps members JSON-serializable and comparable to
    plain strings; the ``Enum`` makes the set closed, so the model cannot
    invent values such as ``"HIGH"`` or ``"critical"``.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TechnicalEntities(BaseModel):
    """Validated output of the pipeline: the entities found in a technical text."""

    technologies: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Technologies, frameworks, tools or services explicitly mentioned in "
            "the text (e.g. FastAPI, Redis, PostgreSQL). Use the canonical name."
        ),
    )
    criticality_level: CriticalityLevel = Field(
        ...,
        description=(
            "Severity of the problem, or relevance of the architecture, described "
            "in the text: 'high' when production users are affected, 'medium' for "
            "degraded but working systems, 'low' for informational content."
        ),
    )
    technical_summary: str = Field(
        ...,
        min_length=10,
        description=(
            "One or two sentence technical summary of the text, written in the "
            "same language as the input text."
        ),
    )

    @field_validator("technologies")
    @classmethod
    def strip_and_deduplicate(cls, value: List[str]) -> List[str]:
        """Semantic rule: drop blank entries, then deduplicate preserving order.

        ``min_length=1`` alone would accept ``["  ", ""]`` — a non-empty list
        with no usable content. Raising ``ValueError`` here surfaces as a
        ``ValidationError`` naming this field, which is exactly what the
        error-aware retry feeds back to the model.
        """
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("technologies must contain at least one non-blank entry")
        return list(dict.fromkeys(cleaned))
