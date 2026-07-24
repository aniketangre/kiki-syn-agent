"""
run_workflow.py
---------------
Use this script to execute a Synera .syn workflow with a JSON input file
and get the result back as JSON.

Usage:
    python run_workflow.py  sphere_workflow.syn  sphere_workflow_template.json

    The second argument is the filled-in template from get_template.py.

Output:
    - The result is printed to the console as JSON
    - The result is also saved to  output.json  (or a custom path you specify)

Optional third argument — custom output path:
    python run_workflow.py  sphere_workflow.syn  my_input.json  my_result.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env (SYNERA_EXE can be set there)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — set SYNERA_EXE in your .env file if not on your PATH
# ---------------------------------------------------------------------------
SYNERA_EXE = os.environ.get("SYNERA_EXE", "syneraheadless.exe")

# Default file where the result JSON will be saved
DEFAULT_OUTPUT_FILE = "output.json"


# ---------------------------------------------------------------------------
# Execute the workflow
# ---------------------------------------------------------------------------

def run_workflow(syn_file: str, input_json: str, output_file: str) -> dict:
    """
    Runs:  syneraheadless execute
              --file      <syn_file>
              --jsoninput <input_json>
              --output    <output_file>

    Returns the result as a clean Python dict (see _format_result below).
    """

    cmd = [
        SYNERA_EXE, "execute",
        "--file",      syn_file,
        "--jsoninput", input_json,
        "--output",    output_file,
    ]

    print(f"Running workflow:  {Path(syn_file).name}")
    print(f"Input file:        {input_json}")
    print(f"Output file:       {output_file}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print any console output from Synera (useful for debugging)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Read the output JSON that Synera wrote
    output_path = Path(output_file)
    if not output_path.exists():
        # If the output file was not created, Synera failed before it could write anything
        return {
            "success": False,
            "error":   "Synera did not produce an output file.",
            "hint":    "Check that the .syn file path is correct and syneraheadless.exe is reachable.",
        }

    with open(output_path, encoding="utf-8") as f:
        raw = json.load(f)

    return _format_result(raw)


# ---------------------------------------------------------------------------
# Clean up the raw Synera output into something readable
# ---------------------------------------------------------------------------

def _format_result(raw: dict) -> dict:
    """
    The raw output from Synera contains a lot of internal .NET type information.
    This function strips that down to just what is useful:

    {
        "success": true,
        "solution_time": "00:00:00.017",
        "outputs": {
            "sphere": {
                "type": "General graph data",
                "value": "Solid Body"
            }
        },
        "errors": []
    }
    """

    # Pull the human-readable string value out of each output
    outputs = {}
    for out in raw.get("outputs", []):

        # Synera stores values as a list under key "0" (the root tree branch)
        str_vals = out.get("value", {}).get("stringValue", {}).get("0", [])

        outputs[out["name"]] = {
            "type":  out.get("typeName", ""),
            # If there is only one value, return it as a plain string
            # If there are multiple (a list), return the whole list
            "value": str_vals[0] if len(str_vals) == 1 else str_vals,
        }

    # Combine workflow-level errors and node-level errors into one list
    all_errors = raw.get("errors", []) + raw.get("nodeErrors", [])

    return {
        "success":       raw.get("success", False),
        "solution_time": raw.get("solutionTime"),
        "outputs":       outputs,
        "errors":        all_errors,
    }


# ---------------------------------------------------------------------------
# Main — run this script directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage:  python run_workflow.py  <workflow.syn>  <input.json>  [output.json]")
        sys.exit(1)

    syn_file    = sys.argv[1]
    input_json  = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_FILE

    # Run the workflow
    result = run_workflow(syn_file, input_json, output_file)

    # Print the clean result as formatted JSON
    print(json.dumps(result, indent=2))

    # Exit with code 0 on success, 1 on failure (useful for scripting)
    sys.exit(0 if result["success"] else 1)
