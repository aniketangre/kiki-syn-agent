"""
create_sphere_tool.py
---------------------
Defines the `create_sphere` tool for the LangGraph agent.

Inputs:
    radius      — float, required — sphere radius
    center_x    — float, optional — X coordinate of center (default 0.0)
    center_y    — float, optional — Y coordinate of center (default 0.0)
    center_z    — float, optional — Z coordinate of center (default 0.0)

Outputs (from Synera):
    volume      — computed volume of the sphere

Exports produced by this tool:
    Results/geometry.step   — sphere geometry in STEP format

Folder structure:
    tools/create_sphere/
        create_sphere.syn               <- the Synera workflow
        create_sphere_input.json        <- input template
        create_sphere_tool.py           <- this file
        Results/
            geometry.step
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
# Paths
# ---------------------------------------------------------------------------

TOOL_DIR      = Path(__file__).parent
SYN_FILE      = str(TOOL_DIR / "create_sphere.syn")
TEMPLATE_FILE = TOOL_DIR / "create_sphere_input.json"
OUTPUT_FILE   = str(TOOL_DIR / "create_sphere_output.json")
RESULTS_DIR   = str(TOOL_DIR / "Results")

load_dotenv()
SYNERA_EXE = os.environ.get("SYNERA_EXE", "syneraheadless.exe")


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def create_sphere(
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
) -> str:
    """
    Creates a sphere in Synera and exports it as a STEP geometry file.

    Use this tool when the user wants to create a sphere or ball geometry.

    Args:
        radius:   REQUIRED. The radius of the sphere (floating point, > 0).
        center_x: X coordinate of the sphere center. Default: 0.0
        center_y: Y coordinate of the sphere center. Default: 0.0
        center_z: Z coordinate of the sphere center. Default: 0.0

    Returns:
        A JSON string with the result, including whether it succeeded,
        the computed volume, and the path to geometry.step.
    """

    # Step 1: Load input template.
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        template = json.load(f)

    # Step 2: Fill in values.
    # Center must be formatted as Synera's Point 3D string.
    center_str = f"Point(X = {center_x}, Y = {center_y}, Z = {center_z})"
    filled_template = _fill_template(template, {
        "center": center_str,
        "radius": radius,
    })

    # Step 3: Write filled template to a temporary file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(filled_template, tmp, indent=2)
        input_path = tmp.name

    # Step 4: Ensure Results directory exists.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 5: Run syneraheadless.exe.
    subprocess.run(
        [SYNERA_EXE, "execute",
         "--file",            SYN_FILE,
         "--jsoninput",       input_path,
         "--output",          OUTPUT_FILE,
         "--exportformat",    ".step",
         "--exportdirectory", RESULTS_DIR],
        capture_output=True, text=True,
    )

    # Step 6: Read the output JSON.
    output_path = Path(OUTPUT_FILE)
    if not output_path.exists():
        return json.dumps({"success": False, "error": "Synera did not produce an output file."})

    with open(output_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Step 7: Format and return result.
    result = _format_result(raw)
    result["inputs_used"] = {
        "radius":   radius,
        "center_x": center_x,
        "center_y": center_y,
        "center_z": center_z,
    }
    result["exports"] = {
        "geometry": str(Path(RESULTS_DIR) / "geometry.step"),
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_template(template: list, values: dict) -> list:
    """Replace named inputs in the Synera template with provided values."""
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
