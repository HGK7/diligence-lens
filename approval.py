from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    report: str
    approved: bool | None


def approval_node(state: ApprovalState) -> dict:
    approved = interrupt(
        {
            "question": "Approve this Diligence Lens report for release?",
            "report_preview": state["report"][:1200],
        }
    )
    return {"approved": bool(approved)}


_builder = StateGraph(ApprovalState)
_builder.add_node("approval", approval_node)
_builder.add_edge(START, "approval")
_builder.add_edge("approval", END)
APPROVAL_GRAPH = _builder.compile(checkpointer=InMemorySaver())


def request_approval(report: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return APPROVAL_GRAPH.invoke(
        {"report": report, "approved": None}, config=config
    )
    
def resume_approval(thread_id: str, approved: bool) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return APPROVAL_GRAPH.invoke(Command(resume=approved), config=config)