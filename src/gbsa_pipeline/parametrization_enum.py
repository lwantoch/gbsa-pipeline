"""Force field and charge method enumerations for parametrization."""

from __future__ import annotations

from enum import StrEnum


class ProteinFF(StrEnum):
    """Supported protein force fields."""

    FF14SB = "ff14SB"
    FF19SB = "ff19SB"
    FF99SB = "ff99SB"

    @classmethod
    def from_str(cls, value: str) -> ProteinFF:
        """Case-insensitive lookup by value string."""
        normalized = value.lower().strip()
        for member in cls:
            if member.value.lower() == normalized:
                return member
        supported = ", ".join(m.value for m in cls)
        raise ValueError(f"Unsupported protein FF '{value}'. Supported: {supported}")


class LigandFF(StrEnum):
    """Supported small-molecule force fields."""

    GAFF = "gaff"
    GAFF2 = "gaff2"


class ChargeMethod(StrEnum):
    """Partial charge assignment methods for ligands.

    AM1BCC  -- AM1-BCC via sqm (AmberTools). Default, no extra dependencies.
    NAGL    -- Graph neural network trained to reproduce AM1-BCC charges.
               Requires ``openff-nagl`` and a model file.
    ESPALOMA -- End-to-end ML charges. Requires ``espaloma-charge``.
    """

    AM1BCC = "am1bcc"
    NAGL = "nagl"
    ESPALOMA = "espaloma-am1bcc"
