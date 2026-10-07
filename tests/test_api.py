from fastapi.testclient import (
    TestClient
)

from api.main import app


client = TestClient(
    app
)


def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert (
        "Visual Defect Detection"
        in response.text
    )