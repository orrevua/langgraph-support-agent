from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from supportagent.llm import build_fake_model
from supportagent.triage import TicketTriage


def test_structured_output_offline():
    model = build_fake_model().with_structured_output(TicketTriage)
    result = model.invoke("My card was stolen!")
    assert isinstance(result, TicketTriage)
    assert result.category == "card"
    assert result.urgency == "high"


def test_generic_policy_calls_matching_tool():
    @tool
    def pix_limit(period: str) -> str:
        """Return the Pix limit for a period."""
        return "R$ 1.000,00"

    ai = build_fake_model().bind_tools([pix_limit]).invoke([HumanMessage("what is my pix limit?")])
    assert ai.tool_calls[0]["name"] == "pix_limit"
