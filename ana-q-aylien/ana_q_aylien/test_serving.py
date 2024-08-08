import pytest
import json

from ana_q_aylien.serving import run_app


@pytest.fixture(scope="session", autouse=True)
def client():
    test_app = run_app()
    test_client = test_app.test_client()
    return test_client


def post_json(client, url, json_dict):
    """Send dictionary json_dict as json to the specified url"""
    response = client.post(
        url, data=json.dumps(json_dict), content_type="application/json"
    )
    return json.loads(response.data)


def test_client_works(client):
    response = client.get("/__ping")
    assert b"Success" in response.data


def test_predict_text(client):
    data = {
        "text": "@Pascal Ham is tasty"
    }
    response = post_json(client, "/predict", data)
    text = response["sentiment"]
    assert text == "Neutral"

