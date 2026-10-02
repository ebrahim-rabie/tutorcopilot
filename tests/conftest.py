import sys
from pathlib import Path

import pytest

from app.main import create_app
from app.settings import Settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def app():
    settings = Settings()
    settings.chunk_strategy = "recursive"
    return create_app(settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
