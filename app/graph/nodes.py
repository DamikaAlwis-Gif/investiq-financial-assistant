from .graph_state import GraphState
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, SystemMessage, ToolMessage, BaseMessage
from .chains import get_formulated_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import logging
from .tools import retrieve_news_data, retrieve_stocks_data, retreive_stock_indicators_for_single_stock
from langgraph.prebuilt import ToolNode
from langgraph.graph import  END

from typing import Literal

tools = [
    retrieve_news_data,
    retrieve_stocks_data,
    retreive_stock_indicators_for_single_stock,
    
]
model_with_tools = ChatGroq(
    model="llama-3.1-8b-instant", temperature=0.0).bind_tools(tools)

tool_node = ToolNode(tools)

def call_model(state: GraphState):
    
    
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    system_message = (
        "You are a expert in stock data and financial news data analys."
        "Give answers in a proffesional tone."
        "Provide a detailed and insightful answer."
        "Give answers in a structured format and use tables to represent data when applicable."
    )   

    messages =[SystemMessage(content=system_message)] + messages
    print(messages)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_use_tools(state: GraphState) ->Literal["tools","delete_messages"] :
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "delete_messages"


def remove_messages(state: GraphState):
   messages = state["messages"]

   def should_delete(msg: BaseMessage):
      return isinstance(msg, ToolMessage) or (isinstance(msg, AIMessage) and not msg.content)
   remaining_messages = [RemoveMessage(id=m.id)
                         for m in messages if should_delete(m)]
   return {
       "messages": remaining_messages
   }


def summarize_conversation(state: GraphState):
  """Summarize a conversation and keep only limited messages in history"""
  
  
  
  llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

  summary = state.get("summary", "")
  if summary:
    summary_message = (
        f"This is summary of the conversation to date: {summary}\n\n"
        "Extend the summary by taking into account the new messages above:"
    )

  else:
    summary_message = "Create a summary of the conversation above."
  messages = state["messages"] + [HumanMessage(summary_message)]
  
  chain = llm | StrOutputParser()
  logging.info("---Generating summary of the conversation---")
  summary = chain.invoke(messages)
  logging.info("---Completed summary of the conversation---")
  # keep only the last 2 messages only
  delete_messages = [RemoveMessage(id=m.id)
                     for m in state["messages"][:-2]]

  # remove analysis results and vector store documents from storing in the memory
  return {
    "summary": summary,
    "messages": delete_messages
    }


def should_summarize(state: GraphState):
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than 4 messages, then summarize the conversation
    if len(messages) > 4:
        return "summarize_conversation"
    # Otherwise end
    return END


def formulate_query(state: GraphState):
  """Genarate a standalone query based on original query, summary and chat history"""

  chain = get_formulated_query_chain()

  chat_history = state["messages"]
  input = state["input"]
  summary = state.get("summary", "")

  # if there is no chat history
  if not chat_history:
     formatted_query = input
  else:
    logging.info("---Genarating formatted query---")
    formatted_query = chain.invoke(
        {
            "chat_history": chat_history,
            "input": input,
            "summary": summary,
        }
    )
  print(f"Formatted query : {formatted_query}")
  logging.info(f"Formatted query : {formatted_query}")  

  
  return {
     "formatted_query": formatted_query,
     "messages": [HumanMessage(state["input"])],
     "summary": summary,
       }


# def extract_context(state: GraphState):
#     """Extract relevant context like stock symbols, period from the question"""

#     context_chain = get_extract_context_chain()
#     # extract data from the formatted query
#     logging.info("---Extracting Context---")
#     context = context_chain.invoke({
#         "question": state["formatted_query"]
#     })

#     logging.info(f"Extracted Context : {context}")

#     return {
#         "context": context
#     }





