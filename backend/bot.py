import os
from pipecat.pipeline.pipeline import Pipeline
from pipecat.workers.runner import WorkerRunner
from pipecat.frames.frames import Frame, LLMRunFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
# from pipecat.services.cartesia.tts import CartesiaTTSService# NEED TO CHANGE THIS
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language
# from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from agent.service import LangGraphProcessor
from agent.graph import Graph
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from dotenv import load_dotenv
from typing import Any
from loguru import logger
from agent.prompts import DEFAULT_SYSTEM_INSTRUCTION


load_dotenv(override=True)


DEFAULT_GREETING = "Start by concisely introducing yourself as Saath E."

DEFAULT_VOICE_ID = "86e30c1d-714b-4074-a1f2-1cb6b552fb49"


class TranscriptLogger(FrameProcessor):
    """Log what the STT actually heard, so speech-clarity issues are visible.

    Sits right after the STT service and passes every frame through unchanged.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info(f"STT final : {frame.text!r}")

        await self.push_frame(frame, direction)


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
    graph = Graph().build()
    settings = session_settings(runner_args)
    session_id = getattr(runner_args, "session_id", None) or "unknown"
    logger.info(f'Starting bot session {session_id}')

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(
            # "base" was a downgrade from Pipecat's default (distil-medium.en);
            # "small" is a much better accuracy/speed trade-off for names and
            # spelled-out email addresses. Override with WHISPER_MODEL if needed.
            model=os.getenv("WHISPER_MODEL", "small"),
            # Pin the language instead of auto-detecting: faster, and much more
            # stable on proper nouns and spelled letters. (Passing `language=`
            # straight to the constructor is deprecated; put it in Settings.)
            language=Language.EN,
            # Bias the decoder toward dictation so spelled letters and "@"/"."
            # come through correctly.
            initial_prompt=(
                "The user dictates emails and names. Write addresses like "
                "john.smith@gmail.com, joining letters without spaces, and "
                "spell proper nouns correctly."
            ),
            hotwords="gmail.com yahoo.com outlook.com email subject reminder",
        ),
        device="cuda",
    )

    transcript_logger = TranscriptLogger()

    # WhisperSTTService subclasses SegmentedSTTService, which only transcribes
    # when it sees VADUserStarted/StoppedSpeakingFrames. The old pipeline got
    # those from LLMUserAggregatorParams; with the aggregators gone we need this
    # standalone VAD processor or the STT never emits a TranscriptionFrame.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
    )

    tts = KokoroTTSService(
        settings=KokoroTTSService.Settings(
            voice="af_heart",
            language=Language.EN_US,
        ),
        # kokoro-onnx synthesizes the entire utterance before yielding its first
        # (and only) audio chunk, so time-to-first-audio grows with reply length
        # (~0.23x realtime). Pipecat's 3.0s default closes the audio context
        # before longer replies produce any audio -> "completed with no audio",
        # and after 3 in a row the service is marked unusable. Give synth room.
        stop_frame_timeout_s=20.0,
    )

    processor = LangGraphProcessor(
        graph,
        session_id,
        settings["greeting"],
        system_instruction=settings["system_instruction"],
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            transcript_logger,
            processor,
            tts,
            transport.output(),
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
