import os
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
# from pipecat.services.cartesia.tts import CartesiaTTSService# NEED TO CHANGE THIS
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language
from pipecat.services.keenable.search import KeenableWebSearch
# from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from dotenv import load_dotenv
from typing import Any
from loguru import logger

load_dotenv(override=True)

DEFAULT_SYSTEM_INSTRUCTION = (
    """
    You are saath-e a helpful voice assistant.
    Help user to perform tasks using the knowledge and resources that you have.
    Use the search tool for current events/weather/news or anything uncertain.
    Keep the responses brief and avoid using long conversation format.
    """
)

DEFAULT_GREETING = "Start by concisely introducing yourself as Saath E."

DEFAULT_VOICE_ID = "86e30c1d-714b-4074-a1f2-1cb6b552fb49"
search = KeenableWebSearch()

def _str_option(body: dict[str, Any], key: str, fallback: str) -> str:
    value = body.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def session_settings(runner_args: RunnerArguments) -> dict[str, str]:
    body = getattr(runner_args, "body", None)
    if not isinstance(body, dict):
        body = {}

    return {
        "system_instruction": _str_option(
            body,
            "system_instruction",
            os.getenv("BOT_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION),
        ),
        "greeting": _str_option(body, "greeting", os.getenv("BOT_GREETING", DEFAULT_GREETING)),
        "voice_id": _str_option(body, "voice_id", os.getenv("CARTESIA_VOICE_ID", DEFAULT_VOICE_ID)),
        "llm_model": _str_option(body, "llm_model", os.getenv("OLLAMA_MODEL", "llama3.2:latest")),
    }

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    settings = session_settings(runner_args)
    session_id = getattr(runner_args, "session_id", None) or "unknown"
    logger.info(f'Starting bot session {session_id}')

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model="base"),
        device="cuda",
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice="af_heart",
            language=Language.EN_US,
        ),
    )

    llm = OLLamaLLMService(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        settings=OLLamaLLMService.Settings(
            model=settings["llm_model"],
            system_instruction=settings["system_instruction"],
        ),
    )

    context = LLMContext(tools=await search.tools())
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[],
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        context.add_message({"role": "developer", "content": settings["greeting"]})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Session {session_id}: client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Session {session_id}: client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)

if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
