"""Voice routes: STT (/agent/transcribe) + TTS (/agent/speak). Hermetic."""

from types import SimpleNamespace

from app.agent.routes import get_generator, get_settings
from app.settings import Settings


class _FakeTranscriptions:
    def __init__(self, text="what is rrf"):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


class _FailingTranscriptions:
    def __init__(self, message):
        self.message = message

    def create(self, **kwargs):
        raise RuntimeError(self.message)


class _FakeGenerator:
    def __init__(self, transcriptions=None):
        inner = transcriptions if transcriptions is not None else _FakeTranscriptions()
        self.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=inner))

    def available(self):
        return True


class _UnavailableGenerator:
    def available(self):
        return False


def _settings(**overrides) -> Settings:
    s = Settings(llm_api_key="")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _override(app, generator, settings):
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_settings] = lambda: settings


def test_transcribe_happy_path(app, client):
    gen = _FakeGenerator()
    settings = _settings()
    _override(app, gen, settings)

    resp = client.post(
        "/agent/transcribe",
        files={"audio": ("clip.webm", b"RIFF-fake-bytes", "audio/webm")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "what is rrf"
    assert body["model"] == settings.stt_model

    call = gen.client.audio.transcriptions.calls[0]
    assert call["model"] == settings.stt_model
    filename, data, mime = call["file"]
    assert filename.endswith(".webm")
    assert data == b"RIFF-fake-bytes"
    assert mime == "audio/webm"


def test_transcribe_rejects_non_audio(app, client):
    _override(app, _FakeGenerator(), _settings())
    resp = client.post(
        "/agent/transcribe",
        files={"audio": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 415
    assert "audio/*" in resp.json()["detail"]


def test_transcribe_rejects_empty_audio(app, client):
    _override(app, _FakeGenerator(), _settings())
    resp = client.post(
        "/agent/transcribe",
        files={"audio": ("clip.webm", b"", "audio/webm")},
    )
    assert resp.status_code == 400


def test_transcribe_unavailable_llm(app, client):
    _override(app, _UnavailableGenerator(), _settings())
    resp = client.post(
        "/agent/transcribe",
        files={"audio": ("clip.webm", b"data", "audio/webm")},
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_transcribe_upstream_failure_maps_502(app, client, monkeypatch):
    monkeypatch.setattr("app.generate.time.sleep", lambda _s: None)
    gen = _FakeGenerator(_FailingTranscriptions("simulated upstream failure"))
    _override(app, gen, _settings())

    resp = client.post(
        "/agent/transcribe",
        files={"audio": ("clip.webm", b"data", "audio/webm")},
    )
    assert resp.status_code == 502
    assert "transcription failed" in resp.json()["detail"]


def test_transcribe_rate_limit_maps_503(app, client, monkeypatch):
    monkeypatch.setattr("app.generate.time.sleep", lambda _s: None)
    gen = _FakeGenerator(_FailingTranscriptions("Error code 429 - rate limit reached"))
    _override(app, gen, _settings())

    resp = client.post(
        "/agent/transcribe",
        files={"audio": ("clip.webm", b"data", "audio/webm")},
    )
    assert resp.status_code == 503


def test_speak_happy_path(app, client, monkeypatch):
    seen = {}

    async def fake_synth(text, voice):
        seen["text"] = text
        seen["voice"] = voice
        return b"MP3-BYTES"

    monkeypatch.setattr("app.agent.routes._edge_synth", fake_synth)
    _override(app, _FakeGenerator(), _settings(tts_provider="edge", tts_voice="en-US-Test"))

    resp = client.post("/agent/speak", json={"text": "  RRF merges ranked lists. "})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert resp.content == b"MP3-BYTES"
    assert seen["text"] == "RRF merges ranked lists."
    assert seen["voice"] == "en-US-Test"


def test_speak_provider_disabled(app, client):
    _override(app, _FakeGenerator(), _settings(tts_provider="none"))
    resp = client.post("/agent/speak", json={"text": "hello"})
    assert resp.status_code == 503
    assert "TC_TTS_PROVIDER" in resp.json()["detail"]


def test_speak_blank_text(app, client):
    _override(app, _FakeGenerator(), _settings(tts_provider="edge"))
    resp = client.post("/agent/speak", json={"text": "   "})
    assert resp.status_code == 400


def test_health_exposes_voice_flags(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "stt_configured" in data
    assert data["tts_provider"] in ("edge", "none")
