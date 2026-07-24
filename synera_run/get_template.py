"""
get_template.py
---------------
Use this script to inspect a Synera .syn workflow file before running it.

It does two things:
  1. Prints a summary of the workflow: what inputs it needs and what it outputs.
  2. Saves a template JSON file that shows you exactly how to write your input.json.

Usage:
    python get_template.py sphere_workflow.syn

Output:
    - A printed summary in the console
    - A file called  sphere_workflow_template.json  in the same folder
      (fill in your values there, then use it with run_workflow.py)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env (SYNERA_EXE can be set there)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — set SYNERA_EXE in your .env file if not on your PATH
# ---------------------------------------------------------------------------
SYNERA_EXE = os.environ.get("SYNERA_EXE", "syneraheadless.exe")


# ---------------------------------------------------------------------------
# Step 1: Ask Synera for information about the workflow
# ---------------------------------------------------------------------------

def get_workflow_info(syn_file: str) -> dict:
    """
    Runs:  syneraheadless info --file <syn_file> --output <temp.json>

    Returns the parsed info as a Python dict.
    The info contains the list of inputs (with types and defaults) and outputs.
    """

    # We write the result to a temp file so we can read it back as JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        info_path = tmp.name

    cmd = [SYNERA_EXE, "info", "--file", syn_file, "--output", info_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Exit code 1 means something went wrong (wrong path, corrupt file, etc.)
    if result.returncode != 0:
        print("ERROR: syneraheadless could not read the workflow file.")
        print(result.stderr)
        sys.exit(1)

    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 2: Ask Synera to generate the input template file
# ---------------------------------------------------------------------------

def save_input_template(syn_file: str, template_path: str):
    """
    Runs:  syneraheadless info --file <syn_file> --template <template_path>

    This tells Synera to write a ready-to-fill JSON template.
    The template already has the correct structure — you just replace
    the placeholder values with your actual numbers or strings.

    Optional inputs appear in the template without a "value" field.
    You can add a value for them manually, or leave them out entirely
    to let Synera use the default.
    """

    cmd = [SYNERA_EXE, "info", "--file", syn_file, "--template", template_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("ERROR: Could not generate template.")
        print(result.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 3: Print a human-readable summary to the console
# ---------------------------------------------------------------------------

def print_summary(info: dict, template_path: str):
    """
    Reads the info dict returned by get_workflow_info() and prints a
    clear, readable summary of the workflow's inputs and outputs.
    """

    print("\n" + "=" * 50)
    print(f"  Workflow summary")
    print("=" * 50)
    print(f"  Document ID : {info.get('documentId', 'n/a')}")

    # ── Inputs ──────────────────────────────────────────────────────────
    inputs = info.get("inputs", [])
    print(f"\n  INPUTS  ({len(inputs)} total)")
    print("  " + "-" * 40)

    for inp in inputs:
        name      = inp.get("name", "?")
        type_name = inp.get("typeName", "?")

        # An input is optional if Synera flags it, or if it has a default value
        has_default = inp.get("defaultValue") is not None
        is_optional = inp.get("isOptional", False) or has_default

        # Try to read the default value as a plain string for display
        default_str = _read_default(inp)

        # Build the status label shown next to the input name
        if is_optional and default_str:
            status = f"optional  (default: {default_str})"
        elif is_optional:
            status = "optional"
        else:
            status = "REQUIRED"

        print(f"  {name}")
        print(f"    type   : {type_name}")
        print(f"    status : {status}")

    # ── Outputs ─────────────────────────────────────────────────────────
    outputs = info.get("outputs", [])
    print(f"\n  OUTPUTS  ({len(outputs)} total)")
    print("  " + "-" * 40)

    for out in outputs:
        print(f"  {out.get('name', '?')}")
        print(f"    type : {out.get('typeName', '?')}")

    # ── Next steps ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  Template saved to:  {template_path}")
    print(f"  Fill in your values, then run:")
    print(f"    python run_workflow.py  <workflow.syn>  {template_path}")
    print("=" * 50 + "\n")


def _read_default(inp: dict) -> str:
    """
    Try to extract a human-readable default value string from an input definition.
    Returns an empty string if no default is available or the format is unexpected.
    """
    try:
        return inp["defaultValue"]["stringValue"]["0"][0]
    except (KeyError, IndexError, TypeError):
        return ""


# ---------------------------------------------------------------------------
# Main — run this script directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:  python get_template.py  <workflow.syn>")
        sys.exit(1)

    syn_file = sys.argv[1]

    # Build a template file name next to the .syn file, e.g. sphere_workflow_template.json
    template_path = str(Path(syn_file).parent / (Path(syn_file).stem + "_template.json"))

    # Get info and generate the template
    info = get_workflow_info(syn_file)
    save_input_template(syn_file, template_path)

    # Print the summary
    print_summary(info, template_path)
