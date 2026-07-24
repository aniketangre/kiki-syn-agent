"""
kiki_recommend_tool.py
----------------------
Defines the `kiki_recommend` tool for the LangGraph agent.

Calls the KIKI ML API to predict the optimal lattice orientation and cell type.
Material constants are resolved from a bone preset (osteoporotic / elderly /
normal / athletic) so the user only needs to provide clinical context, not raw
engineering numbers.

Inputs:
    bone_preset  — str,   required  — bone condition key: "osteoporotic", "elderly",
                                      "normal", or "athletic". Free-text is also
                                      accepted and resolved via keyword matching.
    body_wt      — float, optional  — patient body weight (kg),          range 55–95,   default 75
    max_disp     — float, optional  — max allowable displacement (mm),   range 0–25,    default 5
    max_stress   — float, optional  — max allowable stress (MPa),        range 25–250,  default 50
    vol_frac     — float, optional  — volume fraction,                   range 0.2–0.48, default 0.25
    vox_to_surf  — float, optional  — voxel-to-surface distance (mm),   range 60–200,  default 100

Output (from KIKI API + pass-through values):
    recommended_cell_type — lattice cell type tag (e.g. "BCC", "GYR")
    x_rotation_deg        — optimal rotation around X axis (degrees)
    y_rotation_deg        — optimal rotation around Y axis (degrees)
    z_rotation_deg        — optimal rotation around Z axis (degrees)
    vol_frac              — passed through for use by reconstruct_lattice
    preset_used           — which bone preset was resolved and applied

Folder structure:
    config/
        presets.py               <- bone material presets
    tools/kiki_recommend/
        kiki_recommend_tool.py   <- this file
"""

import json
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from config.presets import BONE_PRESETS, resolve_preset
from tools.kiki_recommend.bounds import validate_inputs

load_dotenv()

KIKI_API_URL = os.environ.get("KIKI_API_URL", "http://127.0.0.1:8000")


@tool
def kiki_recommend(
    bone_preset: str = "normal",
    body_wt: float = 75.0,
    max_disp: float = 5.0,
    max_stress: float = 50.0,
    vol_frac: float = 0.25,
    vox_to_surf: float = 100.0,
) -> str:
    """
    Calls the KIKI ML model to predict the optimal lattice cell type and
    orientation for a patient based on their bone condition and structural
    design constraints.

    Use this tool when the user wants to find the best lattice structure for
    a specific patient profile. After this tool returns, ALWAYS call
    reconstruct_lattice immediately using the returned recommended_cell_type,
    rotation angles, and vol_frac — do not ask the user for confirmation.

    Args:
        bone_preset:  Bone condition of the patient. One of:
                        "osteoporotic" — severely reduced bone density (osteoporosis)
                        "elderly"      — age-related density reduction (65+ years)
                        "normal"       — healthy adult bone (default)
                        "athletic"     — dense, high-stiffness bone (young/active)
                      Free-text descriptions are also accepted (e.g. "fragile bones",
                      "senior patient") and matched to the closest preset.
        body_wt:      Patient body weight in kg. Range: 55–95. Default: 75.
        max_disp:     Maximum allowable displacement under load in mm. Range: 0–25. Default: 5.
        max_stress:   Maximum allowable internal stress in MPa. Range: 25–250. Default: 50.
        vol_frac:     Volume fraction (solid/total volume ratio). Range: 0.2–0.48. Default: 0.25.
        vox_to_surf:  Distance from voxel grid to part surface in mm. Range: 60–200. Default: 100.

    Returns:
        A JSON string with the recommended cell_type, rotation angles, and
        vol_frac — ready to pass directly to reconstruct_lattice.
    """
    # Validate user-facing inputs against allowed bounds before calling the API
    violations = validate_inputs(body_wt, max_disp, max_stress, vol_frac, vox_to_surf)
    if violations:
        return json.dumps({
            "success": False,
            "error": "Input validation failed: " + "; ".join(violations),
        })

    # Resolve bone preset — accept exact key or free-text description
    preset_key = bone_preset.lower().strip()
    if preset_key not in BONE_PRESETS:
        preset_key = resolve_preset(bone_preset) or "normal"

    material = BONE_PRESETS[preset_key]["material"]

    payload = {
        # Material constants from preset
        "E1":   material["E1"],
        "E2":   material["E2"],
        "E3":   material["E3"],
        "G12":  material["G12"],
        "G23":  material["G23"],
        "G31":  material["G31"],
        "NU12": material["NU12"],
        "NU23": material["NU23"],
        "NU31": material["NU31"],
        # Structural parameters from user
        "body_wt":     body_wt,
        "max_disp":    max_disp,
        "max_stress":  max_stress,
        "vol_frac":    vol_frac,
        "vox_to_surf": vox_to_surf,
    }

    try:
        response = requests.post(
            f"{KIKI_API_URL}/predict",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        prediction = response.json()
    except requests.exceptions.ConnectionError:
        return json.dumps({
            "success": False,
            "error": "Could not connect to the KIKI API. Make sure the API server is running.",
        })
    except requests.exceptions.Timeout:
        return json.dumps({
            "success": False,
            "error": "The KIKI API did not respond within 30 seconds.",
        })
    except requests.exceptions.HTTPError as e:
        try:
            detail = response.json()
        except Exception:
            detail = str(e)
        return json.dumps({
            "success": False,
            "error": f"KIKI API rejected the request: {detail}",
        })

    return json.dumps({
        "success":               True,
        "recommended_cell_type": prediction["recommended_cell_type"],
        "x_rotation_deg":        prediction["x_rotation_deg"],
        "y_rotation_deg":        prediction["y_rotation_deg"],
        "z_rotation_deg":        prediction["z_rotation_deg"],
        "vol_frac":              vol_frac,
        "preset_used":           preset_key,
    }, indent=2)
