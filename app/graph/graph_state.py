from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    input : str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary : str
    
