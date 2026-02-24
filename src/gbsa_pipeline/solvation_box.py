"""Solvation helpers with configurable solvent, box shape, padding, and ions."""

from __future__ import annotations

from enum import StrEnum
from typing import Callable

import BioSimSpace as BSS
from pydantic import BaseModel, ConfigDict, model_validator

WaterBuilder = Callable[..., BSS._SireWrappers.System]


class BoxShape(StrEnum):
    """Supported solvent box shapes."""

    TRUNCATED_OCTAHEDRON = "truncated_octahedron"
    CUBIC = "cubic"


class WaterModel(StrEnum):
    """Supported water models for solvation."""

    TIP3P = "tip3p"
    TIP4P = "tip4p"
    TIP5P = "tip5p"
    SPC = "spc"
    SPCE = "spce"

    def builder(self) -> WaterBuilder:
        """Return the BioSimSpace builder for this water model."""
        return _BUILDERS[self]


_BUILDERS = {
    WaterModel.TIP3P: BSS.Solvent.tip3p,
    WaterModel.TIP4P: BSS.Solvent.tip4p,
    WaterModel.TIP5P: BSS.Solvent.tip5p,
    WaterModel.SPC: BSS.Solvent.spc,
    WaterModel.SPCE: BSS.Solvent.spce,
}


def box_parameters(
    box_size: float,
    shape: BoxShape | str = BoxShape.TRUNCATED_OCTAHEDRON,
) -> tuple[BSS.Box, BSS.Types.Angle]:
    """Return a solvated box of the requested shape and size (nm)."""
    shape_enum = _coerce_box_shape(shape)
    length = box_size * BSS.Units.Length.nanometer

    if shape_enum is BoxShape.TRUNCATED_OCTAHEDRON:
        return BSS.Box.truncatedOctahedron(length)
    if shape_enum is BoxShape.CUBIC:
        return BSS.Box.cubic(length)

    raise ValueError(f"Unsupported box shape: {shape}")


class SolvationParams(BaseModel):
    """Validated container for solvation options.

    You can specify either an absolute `box_size` (nm) or a `padding` (nm).
    When both are provided, `padding` takes precedence to match the intent
    of distance-based sizing.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    water_model: WaterModel | str = WaterModel.TIP3P
    shape: BoxShape | str = BoxShape.TRUNCATED_OCTAHEDRON
    box_size: float | None = 8.0
    padding: float | None = None
    solvent: WaterBuilder | None = None
    ion_concentration: float | None = None  # mol/L
    is_neutral: bool = True

    @model_validator(mode="before")
    @classmethod
    def _normalise_strings(cls, values: dict[str, object]) -> dict[str, object]:
        for key in ("water_model", "shape"):
            val = values.get(key)
            if isinstance(val, str):
                values[key] = val.lower()
        return values

    @model_validator(mode="after")
    def _check_dimensions(self) -> SolvationParams:
        if self.box_size is None and self.padding is None:
            raise ValueError("either box_size or padding must be provided")

        if self.box_size is not None and self.box_size <= 0:
            raise ValueError("box_size must be positive (nanometers)")

        if self.padding is not None and self.padding <= 0:
            raise ValueError("padding must be positive (nanometers)")

        return self

    def solvent_builder(self) -> WaterBuilder:
        """Resolve a solvent builder (custom override or named water model)."""
        water = _coerce_water_model(self.water_model)
        return self.solvent or water.builder()

    def box(self) -> tuple[BSS.Box, BSS.Types.Angle]:
        """Build the solvent box for the configured shape/size."""
        if self.box_size is None:
            raise ValueError("box_size not set; use padding or supply box_size")

        return box_parameters(self.box_size, shape=self.shape)


def run_solvation(system: BSS._SireWrappers.System, params: SolvationParams) -> BSS._SireWrappers.System:
    """Prepare and run solvation with a parameter object."""
    water = _coerce_water_model(params.water_model)

    kwargs = {"molecule": system, "is_neutral": params.is_neutral}

    if params.ion_concentration is not None:
        kwargs["ion_conc"] = params.ion_concentration

    if params.padding is not None:
        kwargs["shell"] = params.padding * BSS.Units.Length.nanometer
    elif params.box_size is not None:
        box, angles = params.box()
        kwargs["box"] = box
        kwargs["angles"] = angles
    else:
        raise ValueError("either padding or box_size must be provided")

    return BSS.Solvent.solvate(water.value, **kwargs)


def _coerce_box_shape(shape: BoxShape | str) -> BoxShape:
    if isinstance(shape, BoxShape):
        return shape

    return BoxShape(shape.lower().strip())


def _coerce_water_model(model: WaterModel | str) -> WaterModel:
    if isinstance(model, WaterModel):
        return model

    try:
        return WaterModel(model.lower().strip())
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Water model is not supported: {model}") from exc
