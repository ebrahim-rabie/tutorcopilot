def test_home_serves_spa(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "TutorCopilot" in resp.text
    assert "/static/style.css" in resp.text
    assert "/static/app.js" in resp.text


def test_static_assets_served(client):
    js = client.get("/static/app.js")
    css = client.get("/static/style.css")
    assert js.status_code == 200
    assert css.status_code == 200
    assert "/agent/stream" in js.text
    assert "--accent" in css.text


def test_static_missing_404(client):
    resp = client.get("/static/does-not-exist.js")
    assert resp.status_code == 404


def test_api_routes_unaffected_by_static_mount(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/ask" in paths
    assert "/agent/ask" in paths
    assert "/" not in paths


def test_voice_ui_wired(client):
    html = client.get("/").text
    assert 'id="mic"' in html
    assert 'id="tts"' in html
    js = client.get("/static/app.js").text
    assert "SpeechRecognition" in js
    assert "speechSynthesis" in js
    assert "MediaRecorder" in js
    assert "/agent/transcribe" in js
    assert "/agent/speak" in js


def test_voice_routes_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/agent/transcribe" in paths
    assert "/agent/speak" in paths
