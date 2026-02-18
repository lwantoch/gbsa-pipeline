"""Classes for availible options in gms parameters."""

from enum import StrEnum


class Integrator(StrEnum):
    """Integration algorithm for time propagation of equations of motion."""

    LEAP_FROG = "md"
    VELOCITY_VERLET = "md-vv"
    LEAP_FROG_STOCHASTIC = "sd"
    LANGEVIN = "bd"


class CommMode(StrEnum):
    """Center-of-mass motion removal mode."""

    LINEAR = "Linear"
    ANGULAR = "Angular"
    LINEAR_ACC_CORRECTION = "Linear-acceleration-correction"
    NONE = "None"


class NghCutoffScheme(StrEnum):
    """Neighbor-list and cutoff handling scheme."""

    VERLET = "Verlet"
    GROUP = "group"


class CoulombType(StrEnum):
    """Electrostatics method for long-range Coulomb interactions."""

    CUT_OFF = "Cut-off"
    EWALD = "Ewald"
    PME = "PME"
    PM3AD = "PM3-AD"
    REACTION_FIELD = "Reaction-field"


class CoulombModifier(StrEnum):
    """Modifier applied to short-range Coulomb interactions near cutoff."""

    POTENTIAL_SHIFT = "Potential-shift"
    NONE = "None"


class VDWType(StrEnum):
    """Method for computing van der Waals interactions."""

    CUT_OFF = "Cut-off"
    SHIFT = "Shift"
    PME = "PME"
    Switch = "Switch"


class DispCorr(StrEnum):
    """Long-range dispersion correction mode."""

    NO = "no"
    ENERGY_PRESSURE = "EnerPres"
    ENERGY = "Energy"


class VDWModifier(StrEnum):
    """Modifier applied to Lennard-Jones interactions near cutoff."""

    POTENTIAL_SHIFT = "Potential_Shift"
    NONE = "None"
    FORCE_SWITCH = "Force-switch"
    POTENTIAL_SWITCH = "potential-switch"


class LJPMECombination(StrEnum):
    """Combination rule for Lennard-Jones PME interactions."""

    GEOMETRIC = "Geometric"
    LORENTZ_BERTHELOT = "Lorentz-Berthelot"


class EnsembleTempSetting(StrEnum):
    """Temperature control strategy for ensemble definition."""

    AUTO = "auto"
    CONSTANT = "constant"
    VARIABLE = "variable"


class Thermostat(StrEnum):
    """Thermostat algorithm for temperature coupling."""

    NO = "no"
    BERENDSEN = "berendsen"
    NOSE_HOOVER = "nose-hoover"
    ANDERSEN = "andersen"
    ANDERSEN_MASSIVE = "andersen-massive"
    VRESCALE = "v-rescale"


class Barostat(StrEnum):
    """Barostat algorithm for pressure coupling."""

    NO = "no"
    BERENDSEN = "Berendsen"
    CRESCALE = "C-rescale"
    PARRINELLO_RAHMAN = "Parrinello-Rahman"
    MARTYNA_TUCKERMAN_TK = "MTTK"


class PCoupleType(StrEnum):
    """Pressure coupling geometry mode."""

    ISOTROPIC = "isotropic"
    SEMIISOTROPIC = "semiisotropic"
    ANISOTROPIC = "anisotropic"
    SURFACE_TENSION = "surface-tension"


class VelocityGeneration(StrEnum):
    """Velocity generation flag at simulation start."""

    NO = "no"
    YES = "yes"


class Constraints(StrEnum):
    """Bond/angle constraint type applied during simulation."""

    NONE = "none"
    HYDROGENS_BONDS = "h-bonds"
    ALL_BONDS = "all-bonds"
    HANGLES = "h-angles"
    AANGLES = "all-angles"


class ConstraintsAlgorithms(StrEnum):
    """Constraint solver algorithm."""

    LINCS = "LINCS"
    SHAKE = "SHAKE"
