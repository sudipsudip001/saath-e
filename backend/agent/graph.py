from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.tinker import tinker
from agent.lang_tools import tools


class Graph:
    def build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", tinker)
        g.add_node("tools", ToolNode(tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
        return g.compile(checkpointer=MemorySaver())
