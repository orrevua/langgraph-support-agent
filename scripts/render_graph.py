"""Render the compiled agent graph as mermaid: `make graph` -> docs/graph.mmd.

The README embeds this file's content. Regenerate it whenever you change the graph wiring.
"""

from pathlib import Path

from supportagent.config import Settings
from supportagent.runtime import build_agent

graph, _ = build_agent(Settings(storage="memory", llm_provider="fake"))
mermaid = graph.get_graph().draw_mermaid()
out = Path(__file__).resolve().parents[1] / "docs" / "graph.mmd"
out.write_text(mermaid)
print(mermaid)
