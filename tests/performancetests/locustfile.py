"""Locust load test for the deployed prediction API.

Run headless against the Cloud Run service (or a local container):

    invoke load-test
    invoke load-test --users 100 --spawn-rate 10 --run-time 5m
    locust -f tests/performancetests/locustfile.py  # interactive web UI on :8089

Each simulated user POSTs a base64-encoded 64x64 RGB image to /predict,
the same payload format the frontend sends. Everry request also
writes a PNG to the monitoring bucket, so clean up gs://eurosat_monitoring/
predictions/ after a test run to keep the drift reports meaningful.
"""

import base64
import io
import random

import numpy as np
from locust import HttpUser, between, task
from PIL import Image


def _random_image_b64(seed: int) -> str:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# A few distinct payloads so consecutive requests are not byte-identical.
PAYLOADS = [_random_image_b64(seed) for seed in range(10)]


class PredictUser(HttpUser):
    host = "https://eurosat-api-999981877996.europe-west3.run.app"
    wait_time = between(0.5, 2.0)

    @task(10)
    def predict(self) -> None:
        body = {"instances": [{"image_b64": random.choice(PAYLOADS)}]}
        with self.client.post("/predict", json=body, name="/predict", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            elif "predictions" not in response.json():
                response.failure("Response missing 'predictions' key")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
