import uuid
from collections import OrderedDict
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

import bot as bot_module

STUN_SERVER = {"urls": ["stun:stun.l.google.com:19302"]}

webrtc = SmallWebRTCRequestHandler(ice_servers=[IceServer(**STUN_SERVER)])

app = FastAPI(title="saath-e voice bot (WebRTC)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SESSIONS = 100
sessions: OrderedDict[str, Any] = OrderedDict()


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


async def _answer_offer(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str | None = None,
    settings: Any | None = None,
) -> dict:
    body = await _json_body(request)
    sdp, sdp_type = body.get("sdp"), body.get("type")
    if not sdp or not sdp_type:
        raise HTTPException(
            status_code=422,
            detail='Expected a WebRTC offer such as {"sdp": "...", "type": "offer"}.',
        )

    session_id = session_id or str(uuid.uuid4())
    if settings is None:
        settings = body.get("request_data") or body.get("requestData") or {}

    async def on_connected(connection: SmallWebRTCConnection):
        logger.info(f"Session {session_id}: WebRTC client connected")
        background_tasks.add_task(
            bot_module.bot,
            SmallWebRTCRunnerArguments(
                webrtc_connection=connection,
                session_id=session_id,
                body=settings,
            ),
        )

    return await webrtc.handle_web_request(
        request=SmallWebRTCRequest(
            sdp=sdp,
            type=sdp_type,
            pc_id=body.get("pc_id"),
            restart_pc=body.get("restart_pc"),
            request_data=settings,
        ),
        webrtc_connection_callback=on_connected,
    )


@app.post("/start")
async def start(request: Request):
    body = await _json_body(request)
    if body.get("transport", "webrtc") != "webrtc" or body.get("createDailyRoom"):
        raise HTTPException(
            status_code=400, detail="This server hosts the WebRTC transport only."
        )

    session_id = str(uuid.uuid4())
    sessions[session_id] = body.get("body") or {}
    if len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)
    logger.info(f"Session {session_id}: registered")

    return {"sessionId": session_id, "iceConfig": {"iceServers": [STUN_SERVER]}}


@app.post("/api/connect")
async def connect(request: Request, background_tasks: BackgroundTasks):
    return await _answer_offer(request, background_tasks)


@app.post("/sessions/{session_id}/api/offer")
async def connect_session(
    session_id: str, request: Request, background_tasks: BackgroundTasks
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Unknown session_id; call POST /start first.")
    return await _answer_offer(
        request, background_tasks, session_id=session_id, settings=sessions[session_id]
    )


@app.patch("/api/connect")
@app.patch("/sessions/{session_id}/api/offer")
async def ice_candidates(request: Request):
    body = await _json_body(request)
    pc_id = body.get("pc_id")
    if not pc_id:
        raise HTTPException(status_code=400, detail="Missing 'pc_id' in request body")

    candidates = [
        IceCandidate(
            candidate=c.get("candidate", ""),
            sdp_mid=c.get("sdp_mid", ""),
            sdp_mline_index=c.get("sdp_mline_index", 0),
        )
        for c in (body.get("candidates") or [])
        if isinstance(c, dict)
    ]

    await webrtc.handle_patch_request(
        SmallWebRTCPatchRequest(pc_id=pc_id, candidates=candidates)
    )
    return {"status": "success"}


app.mount("/client", PipecatPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
