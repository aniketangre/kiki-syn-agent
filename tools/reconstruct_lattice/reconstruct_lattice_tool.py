"""
reconstruct_lattice_tool.py
---------------------------
Defines the `reconstruct_lattice` tool for the LangGraph agent.

Inputs (from reconstruct_lattice_input.json):
    cell_type   — Text,  required  — lattice cell type tag (e.g. "GYR", "BCC")
    vol_frac    — float, optional  — volume fraction, default 0.25 (range 0–1)
    x_rotation  — float, optional  — rotation around X axis in degrees, default 0
    y_rotation  — float, optional  — rotation around Y axis in degrees, default 0
    z_rotation  — float, optional  — rotation around Z axis in degrees, default 0

Exports produced by this tool:
    Results/pattern_image.png   — rendered preview image of the lattice

Valid cell types (16 options):
    ID  | Tag
    ----+-------
     0  | GYR
     1  | SCH
     2  | DTPMS
     3  | SPP
     4  | SC
     5  | BCC
     6  | FCC
     7  | DIA
     8  | FLU
     9  | OCT
    10  | KEV
    11  | DDK2
    12  | HCG
    13  | PSM2
    14  | RDO
    15  | TEG3

Folder structure:
    tools/reconstruct_lattice/
        reconstruct_lattice.syn         <- the Synera workflow
        reconstruct_lattice_input.json  <- input template (5 inputs)
        reconstruct_lattice_tool.py     <- this file
        Results/
            pattern_image.png
"""

import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Paths — everything lives inside this tool's folder
# ---------------------------------------------------------------------------

TOOL_DIR      = Path(__file__).parent
SYN_FILE      = str(TOOL_DIR / "reconstruct_lattice.syn")
TEMPLATE_FILE = TOOL_DIR / "reconstruct_lattice_input.json"
OUTPUT_FILE   = str(TOOL_DIR / "reconstruct_lattice_output.json")
RESULTS_DIR   = str(TOOL_DIR / "Results")

load_dotenv()
SYNERA_EXE = os.environ.get("SYNERA_EXE", "syneraheadless.exe")

# ---------------------------------------------------------------------------
# Valid cell types — 16 options, addressable by tag string or numeric ID
# ---------------------------------------------------------------------------

CELL_TYPES = [
    "GYR",   # 0
    "SCH",   # 1
    "DTPMS", # 2
    "SPP",   # 3
    "SC",    # 4
    "BCC",   # 5
    "FCC",   # 6
    "DIA",   # 7
    "FLU",   # 8
    "OCT",   # 9
    "KEV",   # 10
    "DDK2",  # 11
    "HCG",   # 12
    "PSM2",  # 13
    "RDO",   # 14
    "TEG3",  # 15
]

_ID_TO_TAG = {str(i): tag for i, tag in enumerate(CELL_TYPES)}


def _resolve_cell_type(cell_type: str) -> str | None:
    """
    Accept a tag ("GYR") — case-insensitive.
    Returns the canonical uppercase tag, or None if not recognised.
    """
    tag = cell_type.strip().upper()
    return tag if tag in CELL_TYPES else None


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def reconstruct_lattice(
    cell_type: str,
    vol_frac: float = 0.25,
    x_rotation: float = 0.0,
    y_rotation: float = 0.0,
    z_rotation: float = 0.0,
) -> str:
    """
    Reconstructs a 3D lattice structure in Synera and exports a rendered
    preview image (PNG).

    Use this tool when the user wants to generate, visualize, or reconstruct
    a lattice or unit cell structure.

    Args:
        cell_type:  REQUIRED. The lattice cell type as a string tag. Must be
                    one of the following values exactly:

                    Tag    | Full name
                    -------+----------------------
                    GYR    | Gyroid
                    SCH    | Schwartz
                    DTPMS  | Double TPMS
                    SPP    | Split P
                    SC     | Simple Cubic
                    BCC    | Body-Centred Cubic
                    FCC    | Face-Centred Cubic
                    DIA    | Diamond
                    FLU    | Fluorite
                    OCT    | Octet
                    KEV    | Kelvin
                    DDK2   | Dodekaeder V2
                    HCG    | Honeycomb Graph
                    PSM2   | Polymer Shell Mesh V2
                    RDO    | Rhombic Dodecahedron
                    TEG3   | Tesseract Graph V3

        vol_frac:   Volume fraction of the lattice (0.0 – 1.0). Default: 0.25
                    Lower values = more open/porous; higher values = denser.

        x_rotation: Rotation around the X axis in degrees. Default: 0.0
        y_rotation: Rotation around the Y axis in degrees. Default: 0.0
        z_rotation: Rotation around the Z axis in degrees. Default: 0.0

    Returns:
        A JSON string with the result, including whether it succeeded,
        the inputs used, and the path to the exported pattern_image.png.
    """

    # Validate cell_type
    resolved = _resolve_cell_type(cell_type)
    if resolved is None:
        valid = ", ".join(f"{i}={t}" for i, t in enumerate(CELL_TYPES))
        return json.dumps({
            "success": False,
            "error": f"Unknown cell_type '{cell_type}'. Valid options: {valid}",
        })

    # Step 1: Load the input template.
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        template = json.load(f)

    # Step 2: Fill in all five parameter values.
    filled_template = _fill_template(template, {
        "cell_type":  resolved,
        "vol_frac":   vol_frac,
        "x_rotation": x_rotation,
        "y_rotation": y_rotation,
        "z_rotation": z_rotation,
    })

    # Step 3: Write filled template to a temporary file for syneraheadless.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(filled_template, tmp, indent=2)
        input_path = tmp.name

    # Step 4: Ensure the Results directory exists before running.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 5: Run syneraheadless.exe.
    subprocess.run(
        [SYNERA_EXE, "execute",
         "--file",            SYN_FILE,
         "--jsoninput",       input_path,
         "--output",          OUTPUT_FILE,
         "--exportformat",    ".png",
         "--exportdirectory", RESULTS_DIR],
        capture_output=True, text=True,
    )

    # Step 6: Read the output JSON that Synera wrote.
    output_path = Path(OUTPUT_FILE)
    if not output_path.exists():
        return json.dumps({"success": False, "error": "Synera did not produce an output file."})

    with open(output_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Step 7: Format and return the result.
    result = _format_result(raw)
    result["inputs_used"] = {
        "cell_type":  resolved,
        "vol_frac":   vol_frac,
        "x_rotation": x_rotation,
        "y_rotation": y_rotation,
        "z_rotation": z_rotation,
    }
    result["exports"] = {
        "lattice_image": str(Path(RESULTS_DIR) / "lattice_image.png"),
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_template(template: list, values: dict) -> list:
    """Replace named inputs in the Synera template with the provided values."""
    filled = copy.deepcopy(template)
    for inp in filled:
        name = inp.get("name")
        if name in values:
            inp["value"] = {
                "stringValue": {
                    "0": [str(values[name])]
                }
            }
    return filled


def _format_result(raw: dict) -> dict:
    """Strip the raw Synera output down to the fields useful for the agent."""
    outputs = {}
    for out in raw.get("outputs", []):
        str_vals = out.get("value", {}).get("stringValue", {}).get("0", [])
        outputs[out["name"]] = {
            "type":  out.get("typeName", ""),
            "value": str_vals[0] if len(str_vals) == 1 else str_vals,
        }

    return {
        "success":       raw.get("success", False),
        "solution_time": raw.get("solutionTime"),
        "outputs":       outputs,
        "errors":        raw.get("errors", []) + raw.get("nodeErrors", []),
    }
