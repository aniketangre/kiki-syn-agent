# utils/bounds.py
#
# Single source of truth for all input field bounds used by the KIKI optimizer.
#
# HOW TO USE:
#   from utils.bounds import BOUNDS
#
#   BOUNDS.body_wt.min      # minimum allowed value
#   BOUNDS.body_wt.max      # maximum allowed value
#   BOUNDS.body_wt.default  # suggested starting value
#   BOUNDS.body_wt.step     # slider step size (used in the Gradio UI)
#
# WHY A CENTRAL BOUNDS FILE?
#   Keeping all bounds here means you only need to change one place to update
#   validation in both the REST API (Pydantic) and the browser UI (Gradio).
#   No more hunting through main.py to find where a limit was set.

from typing import NamedTuple


# ---------------------------------------------------------------------------
# FieldBounds — stores the range and display hints for a single input field
# ---------------------------------------------------------------------------

class FieldBounds(NamedTuple):
    """
    Describes the allowed range for one numeric input field.

    Attributes:
        min     : Smallest value the field may take (inclusive).
        max     : Largest value the field may take (inclusive).
        default : Sensible starting value shown in the UI and API examples.
        step    : Increment used by the Gradio slider (controls slider precision).
    """
    min:     float
    max:     float
    default: float
    step:    float


# ---------------------------------------------------------------------------
# InputBounds — collects FieldBounds for every input the model expects
# ---------------------------------------------------------------------------

class InputBounds:
    """
    All input bounds for the KIKI Lattice Structure Optimizer.

    Each attribute is a FieldBounds named-tuple for one model input.
    Edit the numbers below to change allowed ranges everywhere at once.
    """

    # -----------------------------------------------------------------------
    # Patient & Loading
    # -----------------------------------------------------------------------
    body_wt = FieldBounds(
        min=55.0, max=95.0, default=75.0, step=1.0
        # Unit: kg  |  Range from database (55–95 kg)
    )

    # -----------------------------------------------------------------------
    # Young's Moduli  (GPa)
    # How stiff the material is along each principal axis.
    # Higher value = harder to stretch in that direction.
    # -----------------------------------------------------------------------
    E1 = FieldBounds(
        min=0.0, max=50.0, default=21.0, step=0.5
        # Stiffness along the x-axis
    )
    E2 = FieldBounds(
        min=0.0, max=50.0, default=10.0, step=0.5
        # Stiffness along the y-axis
    )
    E3 = FieldBounds(
        min=0.0, max=50.0, default=10.0, step=0.5
        # Stiffness along the z-axis
    )

    # -----------------------------------------------------------------------
    # Shear Moduli  (GPa)
    # Resistance to sliding deformation in each material plane.
    # -----------------------------------------------------------------------
    G12 = FieldBounds(
        min=0.0, max=30.0, default=6.0, step=0.5
        # Shear in the x-y plane
    )
    G23 = FieldBounds(
        min=0.0, max=30.0, default=4.0, step=0.5
        # Shear in the y-z plane
    )
    G31 = FieldBounds(
        min=0.0, max=30.0, default=6.0, step=0.5
        # Shear in the z-x plane
    )

    # -----------------------------------------------------------------------
    # Poisson's Ratios  (dimensionless)
    # How much the material contracts sideways when stretched along one axis.
    # NOTE: physics-based pairwise and global constraints are checked separately
    #       in main.py (they depend on the combination of E values at runtime).
    # -----------------------------------------------------------------------
    NU12 = FieldBounds(
        min=0.1, max=0.45, default=0.3, step=0.01
        # Contraction in y when stretched in x
    )
    NU23 = FieldBounds(
        min=0.1, max=0.45, default=0.2, step=0.01
        # Contraction in z when stretched in y
    )
    NU31 = FieldBounds(
        min=0.1, max=0.45, default=0.3, step=0.01
        # Contraction in x when stretched in z
    )

    # -----------------------------------------------------------------------
    # Structural Constraints
    # -----------------------------------------------------------------------
    max_disp = FieldBounds(
        min=0.0, max=25, default=5.0, step=0.5
        # Unit: mm  |  Range from database (0.005–310.21 mm)
    )
    max_stress = FieldBounds(
        min=25, max=250, default=50, step=1.0
        # Unit: MPa  |  Range from database (34.2–6992.87 MPa)
    )
    vol_frac = FieldBounds(
        min=0.20, max=0.48, default=0.25, step=0.01
        # Dimensionless  |  Range from database (0.20–0.48)
    )
    vox_to_surf = FieldBounds(
        min=60.0, max=200.0, default=100.0, step=1.0
        # Unit: mm  |  Range from database (60.33–240.76 mm)
    )

# Singleton — import this object wherever bounds are needed.
BOUNDS = InputBounds()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_bounds_summary() -> str:
    """
    Human-readable summary of user-facing parameter bounds.
    Inject this into the SYSTEM_PROMPT so the agent knows valid ranges
    before calling kiki_recommend.
    """
    b = BOUNDS
    return (
        f"    body_wt    : {b.body_wt.min}–{b.body_wt.max} kg   (default {b.body_wt.default})\n"
        f"    max_stress : {b.max_stress.min}–{b.max_stress.max} MPa (default {b.max_stress.default})\n"
        f"    max_disp   : {b.max_disp.min}–{b.max_disp.max} mm   (default {b.max_disp.default})\n"
        f"    vol_frac   : {b.vol_frac.min}–{b.vol_frac.max}        (default {b.vol_frac.default})\n"
        f"    vox_to_surf: {b.vox_to_surf.min}–{b.vox_to_surf.max} mm (default {b.vox_to_surf.default})"
    )


def validate_inputs(
    body_wt: float,
    max_disp: float,
    max_stress: float,
    vol_frac: float,
    vox_to_surf: float,
) -> list:
    """
    Validate user-facing inputs against BOUNDS.
    Returns a list of human-readable violation messages (empty if all valid).
    """
    b = BOUNDS
    violations = []
    checks = [
        ("body_wt",     body_wt,     b.body_wt,     "kg"),
        ("max_disp",    max_disp,    b.max_disp,    "mm"),
        ("max_stress",  max_stress,  b.max_stress,  "MPa"),
        ("vol_frac",    vol_frac,    b.vol_frac,    ""),
        ("vox_to_surf", vox_to_surf, b.vox_to_surf, "mm"),
    ]
    for name, value, bounds, unit in checks:
        if not (bounds.min <= value <= bounds.max):
            suffix = f" {unit}" if unit else ""
            violations.append(
                f"{name}={value}{suffix} is out of range "
                f"({bounds.min}–{bounds.max}{suffix})"
            )
    return violations
