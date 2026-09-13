"""Standalone Shazam recognition microservice.

Wraps ShazamIO behind a small FastAPI app so the main Telegram bot can
send it an audio/video file and get back track metadata (title, artist,
cover art, Apple Music / Spotify / YouTube links when available).

Designed to be deployed as its own Render Web Service, independent from
the bot itself — mirrors how COBALT_API_URL is a separate proxied service
in the bot's config.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from shazamio import Shazam

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("shazam_service")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Optional shared-secret auth. If API_KEY is unset, the service accepts
# unauthenticated requests (useful for local testing) — but for a public
# Render deployment you should always set this.
API_KEY = (os.getenv("API_KEY") or "").strip()

# Reject files above this size before ever touching disk/Shazam. Shazam only
# needs a short audio fingerprint, so callers should already be trimming
# audio, but this is a hard backstop against abuse.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))  # 30 MB

# Recognition itself can be slow on a cold Render instance; keep this
# generous but bounded so a stuck request can't hang forever.
RECOGNIZE_TIMEOUT_SECONDS = float(os.getenv("RECOGNIZE_TIMEOUT_SECONDS", "45"))

ALLOWED_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga", ".opus", ".flac",
    ".mp4", ".mov", ".mkv", ".webm", ".3gp", ".amr",
}

app = FastAPI(
    title="Shazam Recognition Service",
    description="Minimal ShazamIO wrapper for audio/video track recognition.",
    version="1.0.0",
)

_shazam = Shazam()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


def _check_api_key(x_api_key: Optional[str]) -> None:
    """Validate the caller-supplied API key.

    IMPORTANT: this is read from the `X-API-Key` HTTP header (via FastAPI's
    Header() dependency below), not from a query parameter — query params
    end up logged in access logs / proxies, which would leak the key.
    """
    if not API_KEY:
        # No key configured on this deployment: allow all requests.
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _extract_track_info(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Flatten the (fairly deep) ShazamIO response into a small, stable
    payload the bot can render without needing to know ShazamIO's schema."""
    track = result.get("track")
    if not track:
        return None

    title = track.get("title")
    subtitle = track.get("subtitle")  # usually the artist name

    images = track.get("images") or {}
    cover_art = images.get("coverarthq") or images.get("coverart") or images.get("background")

    genre = (track.get("genres") or {}).get("primary")

    # Sections/hub can carry Apple Music, Shazam, and provider deep-links.
    apple_music_url: Optional[str] = None
    shazam_url: Optional[str] = track.get("url")
    spotify_url: Optional[str] = None
    youtube_url: Optional[str] = None

    hub = track.get("hub") or {}
    for action in hub.get("actions", []) or []:
        uri = action.get("uri")
        if not uri:
            continue
        if "music.apple.com" in uri and not apple_music_url:
            apple_music_url = uri

    for provider in hub.get("providers", []) or []:
        provider_type = (provider.get("type") or "").upper()
        for action in provider.get("actions", []) or []:
            uri = action.get("uri")
            if not uri:
                continue
            if provider_type == "SPOTIFY" and not spotify_url:
                spotify_url = uri
            elif provider_type in {"YOUTUBEMUSIC", "YOUTUBE"} and not youtube_url:
                youtube_url = uri

    # Some responses expose YouTube data separately under "sections".
    for section in track.get("sections", []) or []:
        if section.get("type") == "YOUTUBE" and not youtube_url:
            yt = section.get("youtubeurl") or {}
            youtube_url = yt.get("url") or youtube_url

    return {
        "title": title,
        "artist": subtitle,
        "album": None,
        "genre": genre,
        "cover_art": cover_art,
        "shazam_url": shazam_url,
        "apple_music_url": apple_music_url,
        "spotify_url": spotify_url,
        "youtube_url": youtube_url,
        "shazam_track_id": track.get("key"),
    }


def _has_allowed_extension(filename: Optional[str]) -> bool:
    if not filename:
        # Voice messages from Telegram often arrive without a helpful
        # filename/extension (e.g. raw .oga blobs) — don't reject those.
        return True
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS or suffix == ""


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "shazam-recognition", "status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> JSONResponse:
    _check_api_key(x_api_key)

    if not _has_allowed_extension(file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    request_id = uuid.uuid4().hex[:8]
    started_at = time.perf_counter()

    suffix = Path(file.filename).suffix if file.filename else ".bin"
    tmp_path: Optional[str] = None

    try:
        total_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
                    )
                tmp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(
            "recognize request_id=%s filename=%s size=%s",
            request_id, file.filename, total_bytes,
        )

        try:
            result = await asyncio.wait_for(
                _shazam.recognize(tmp_path),
                timeout=RECOGNIZE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("recognize request_id=%s timed out", request_id)
            raise HTTPException(status_code=504, detail="Recognition timed out")
        except Exception as exc:
            logger.exception("recognize request_id=%s failed: %s", request_id, exc)
            raise HTTPException(status_code=502, detail="Recognition service error")

        track_info = _extract_track_info(result or {})
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "recognize request_id=%s matched=%s duration_ms=%.0f",
            request_id, bool(track_info), duration_ms,
        )

        if not track_info:
            return JSONResponse(status_code=200, content={"matched": False, "track": None})

        return JSONResponse(status_code=200, content={"matched": True, "track": track_info})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
