from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(model="moonshotai/kimi-k2-instruct-0905")


# Configuration of MCP client for MCP servers
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",          
            "args": ["/home/neurobuntu/My_Files/Coding_Files/Tutotials/LangGraph-tutorial/08_MCP/01.1_mcp_server.py"],
        },
    }
)


# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    # retrive all the tools present in mcp server
    tools = await client.get_tools()

    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    # nodes (need to make each node execution async)
    async def chat_node(state: ChatState):

        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {'messages': [response]}
    
    # by default async
    tool_node = ToolNode(tools)

    # defining graph and nodes
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # defining graph connections
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    # running the graph
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Add 45 and 56")]})

    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())