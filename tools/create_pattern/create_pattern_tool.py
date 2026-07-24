"""
create_pattern_tool.py
----------------------
Defines the `create_pattern` tool for the LangGraph agent.

The tool accepts pattern parameters from the agent, fills those values into
the Synera input template, runs create_pattern.syn via syneraheadless, and
returns the result as a JSON string.

Exports produced by this tool:
    Results/geometry.step       — the pattern geometry in STEP format
    Results/pattern_image.png   — a rendered preview image of the pattern

Folder structure:
    tools/create_pattern/
        create_pattern.syn              <- the Synera workflow
        create_pattern_input.json       <- input template (all inputs optional)
        create_pattern_tool.py          <- this file
        Results/
            geometry.step
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

# The folder containing this file
TOOL_DIR = Path(__file__).parent

# The Synera workflow file for this tool
SYN_FILE = str(TOOL_DIR / "create_pattern.syn")

# The input template — contains the correct JSON structure with default values
TEMPLATE_FILE = TOOL_DIR / "create_pattern_input.json"

# Output JSON from syneraheadless (named after the .syn file)
OUTPUT_FILE = str(TOOL_DIR / (Path(SYN_FILE).stem + "_output.json"))

# Exported geometry and image files land here
RESULTS_DIR = str(TOOL_DIR / "Results")

# Read SYNERA_EXE from .env (or fall back to expecting it on PATH)
load_dotenv()
SYNERA_EXE = os.environ.get("SYNERA_EXE", "syneraheadless.exe")


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@tool
def create_pattern(
    u_spacing: float = 0.5,
    v_spacing: float = 0.5,
    u_size: float = 1.0,
    v_size: float = 1.0,
    pattern_type: int = 1,
) -> str:
    """
    Creates a 2D surface pattern in Synera and exports it as a STEP geometry
    file and a PNG preview image.

    Use this tool when the user wants to create or generate a surface pattern.
    The tool runs the Synera CAD workflow and returns information about the
    result. Exported files are saved to the Results folder inside the tool
    directory.

    All parameters are optional — Synera will use the template defaults if
    they are not provided.

    Args:
        u_spacing:    Spacing between elements in the U direction. Default: 0.5
        v_spacing:    Spacing between elements in the V direction. Default: 0.5
        u_size:       Size of the pattern in the U direction (floating point). Default: 1.0
        v_size:       Size of the pattern in the V direction (floating point). Default: 1.0
        pattern_type: Integer selecting the pattern type. Default: 1
                      0 = Curved Beam
                      1 = Glass Sponge 1
                      2 = Glass Sponge 2

    Returns:
        A JSON string with the result, including whether it succeeded,
        and the file paths to geometry.step and pattern_image.png.
    """

    # Step 1: Load the input template.
    # The template already has the correct Synera JSON structure with defaults.
    with open(TEMPLATE_FILE, encoding="utf-8") as f:
        template = json.load(f)

    # Step 2: Fill in the values the agent provided.
    # Input names must match exactly what is in create_pattern_input.json.
    # Boolean must be lowercase string: True → "true", False → "false"
    # NOTE: "plane" is an internal workflow connection in Synera — do NOT pass it
    # via --jsoninput or Synera will fail to parse the string and yield null.
    filled_template = _fill_template(template, {
        "u-size":       u_size,
        "v-size":       v_size,
        "u-spacing":    u_spacing,
        "v-spacing":    v_spacing,
        "pattern-type": pattern_type,
    })

    # Step 3: Write the filled template to a temporary file.
    # syneraheadless reads inputs from a JSON file via --jsoninput.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(filled_template, tmp, indent=2)
        input_path = tmp.name

    # Step 4: Ensure the Results directory exists before running.
    # Synera will fail silently if the export directory does not exist.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Step 5: Run syneraheadless.exe.
    # --exportformat .step .png  exports geometry.step and pattern_image.png
    # --exportdirectory          controls where those files land (Results subfolder)
    subprocess.run(
        [SYNERA_EXE, "execute",
         "--file",            SYN_FILE,
         "--jsoninput",       input_path,
         "--output",          OUTPUT_FILE,
         "--exportformat",    ".step", ".png",
         "--exportdirectory", RESULTS_DIR],
        capture_output=True, text=True,
    )

    # Step 6: Read the output JSON that Synera wrote.
    output_path = Path(OUTPUT_FILE)
    if not output_path.exists():
        return json.dumps({"success": False, "error": "Synera did not produce an output file."})

    with open(output_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Step 7: Clean up the raw output and attach the export file paths.
    result = _format_result(raw)
    result["exports"] = {
        "geometry":      str(Path(RESULTS_DIR) / "geometry.step"),
        "pattern_image": str(Path(RESULTS_DIR) / "pattern_image.png"),
    }
    # Include the values actually sent to Synera so the agent can confirm
    # what was used — useful for debugging parameter mapping issues.
    result["inputs_used"] = {
        "pattern_type": pattern_type,
        "u_spacing":    u_spacing,
        "v_spacing":    v_spacing,
        "u_size":       u_size,
        "v_size":       v_size,
    }
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Helper: fill values into the template
# ---------------------------------------------------------------------------

def _fill_template(template: list, values: dict) -> list:
    """
    Replace values in the Synera input template.

    template : the list loaded from create_pattern_input.json
    values   : dict mapping input name to value

    For each input in the template, if its name appears in `values`,
    the existing value is replaced with the one provided.
    Inputs NOT in `values` are left unchanged — Synera keeps the template default.
    """
    # Deep copy so the original template in memory is never modified
    filled = copy.deepcopy(template)

    for inp in filled:
        name = inp.get("name")
        if name in values:
            # Synera always expects values as strings under key "0"
            inp["value"] = {
                "stringValue": {
                    "0": [str(values[name])]
                }
            }

    return filled


# ---------------------------------------------------------------------------
# Helper: clean up raw Synera output
# ---------------------------------------------------------------------------

def _format_result(raw: dict) -> dict:
    """
    Strip the raw Synera output down to the fields that are useful
    for the agent to read and report back to the user.
    """
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
