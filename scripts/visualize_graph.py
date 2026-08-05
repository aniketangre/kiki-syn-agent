"""
visualize_graph.py
------------------
Save PNG diagrams of the agent2 multi-agent graph to the diagrams/ directory.

Outputs:
    diagrams/graph_supervisor.png  — top-level supervisor graph
    diagrams/graph_expanded.png    — sub-agent internals expanded (xray=True)
    diagrams/graph_kiki.png        — KIKI sub-agent ReAct loop only
    diagrams/graph_synera.png      — Synera sub-agent ReAct loop only

draw_mermaid_png() calls the free mermaid.ink API — requires internet access.
If offline, the script falls back to ASCII art printed to the terminal.

Usage:
    python visualize_graph.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so agent.py can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import graph, kiki_subgraph, synera_subgraph

OUT_DIR = Path("diagrams")
OUT_DIR.mkdir(exist_ok=True)

OUTPUT = {
    "graph_supervisor.png": graph.get_graph(),
    "graph_expanded.png":   graph.get_graph(xray=True),
    "graph_kiki.png":       kiki_subgraph.get_graph(),
    "graph_synera.png":     synera_subgraph.get_graph(),
}

for filename, g in OUTPUT.items():
    path = OUT_DIR / filename
    try:
        png = g.draw_mermaid_png()
        path.write_bytes(png)
        print(f"Saved: {path}")
    except Exception as e:
        print(f"Could not save {path}: {e}")
        print("  ASCII fallback:")
        print(g.draw_ascii())
