from loguru import logger
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame, TranscriptionFrame, InputTextRawFrame, TTSTextFrame, LLMRunFrame
from langchain_core.messages import HumanMessage


class LangGraphProcessor(FrameProcessor):
    """Bridges the Pipecat pipeline to the LangGraph agent.

    Turns user speech/text into ``HumanMessage``s, runs the graph, and pushes
    the agent's reply back downstream as TTS text.
    """

    def __init__(
        self,
        graph,
        session_id: str,
        greeting: str,
        system_instruction: str | None = None,
    ):
        super().__init__()
        self.graph = graph
        self.session_id = session_id
        self.greeting = greeting
        # Forwarded to the ``tinker`` node so per-session prompts still work.
        self.system_instruction = system_instruction

    async def _run_graph(self, text: str):
        try:
            result = await self.graph.ainvoke(
                {"messages": [HumanMessage(content=text)]},
                {
                    "configurable": {
                        "thread_id": self.session_id,
                        "system_instruction": self.system_instruction,
                    }
                },
            )
            reply = self._extract_reply(result)
            if reply:
                await self.push_frame(TTSTextFrame(text=reply, aggregated_by="sentence"))
        except Exception as e:
            # Never let a graph/Ollama failure stall the pipeline silently: the
            # user must still hear *something*.
            logger.exception(f"Graph invocation failed: {e}")
            await self.push_frame(
                TTSTextFrame(text="Sorry, I had a hiccup. Could you say that again?", aggregated_by="sentence")
            )

    @staticmethod
    def _extract_reply(result) -> str:
        """Flatten the final message's content into plain text for the TTS."""
        content = result["messages"][-1].content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return (content or "").strip()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Must be called or the base class bookkeeping (start/stop) breaks.
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Empty transcripts are noise; drop them but keep the pipeline alive.
            if frame.text.strip():
                await self._run_graph(frame.text)
        elif isinstance(frame, InputTextRawFrame):
            text = frame.text.strip()
            if text:
                await self._run_graph(text)
        elif isinstance(frame, LLMRunFrame):
            await self._run_graph(self.greeting)
        else:
            # EndFrame/CancelFrame/VAD/audio must reach transport.output() or
            # the worker never shuts down.
            await self.push_frame(frame, direction)
