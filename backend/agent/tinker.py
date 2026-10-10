import os

from agent.state import AgentState
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from agent.prompts import DEFAULT_SYSTEM_INSTRUCTION
from agent.lang_tools import tools

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"), temperature=0.1
).bind_tools(tools)


async def tinker(state: AgentState, config: RunnableConfig):
    # Per-session prompt comes through the graph config (see LangGraphProcessor);
    # fall back to the packaged default when it isn't supplied.
    instruction = (config.get("configurable") or {}).get("system_instruction")
    system = SystemMessage(instruction or DEFAULT_SYSTEM_INSTRUCTION)
    return {"messages": [await llm.ainvoke([system] + state["messages"])]}
