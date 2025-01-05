from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from .graph_state import GraphState
from .nodes import call_model, tool_node, should_use_tools, remove_messages, should_summarize
from .nodes import summarize_conversation, formulate_query
from langgraph.graph import StateGraph

memory = MemorySaver()

workflow = StateGraph(GraphState)
workflow.add_node("formulate_query", formulate_query)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("delete_messages", remove_messages)

workflow.add_edge(START, "formulate_query")
workflow.add_edge("formulate_query", "agent")
workflow.add_conditional_edges("agent", should_use_tools, [
                               "tools", "delete_messages"])
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "delete_messages", should_summarize, [END, "summarize_conversation"])

app = workflow.compile(checkpointer=memory)


def create_workflow():
    
    return app


