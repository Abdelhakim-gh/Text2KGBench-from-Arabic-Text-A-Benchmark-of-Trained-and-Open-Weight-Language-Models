"""The common triple schema, shared by gold facts (from ArSRED), mREBEL's
output, and every LLM's output, so the scorer never needs to know which
source a triple came from.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    subject_type: str | None = None  # known for gold and mREBEL, unknown (None) for LLM output
    object_type: str | None = None
