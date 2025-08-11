import os
import mlflow
import requests
import pytest

# Optionally get the MLflow server URI from an environment variable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def test_mlflow_server_health():
    """Test that the MLflow tracking server is reachable."""
    try:
        response = requests.get(MLFLOW_TRACKING_URI)
        # For example, the server should respond with a 200 status code if healthy.
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
    except Exception as e:
        pytest.fail(f"Could not reach the MLflow tracking server at {MLFLOW_TRACKING_URI}: {e}")

def test_mlflow_run():
    """Test a simple MLflow run."""
    try:
        with mlflow.start_run():
            mlflow.log_param("test_param", 42)
    except Exception as e:
        pytest.fail(f"Failed to start or log in an MLflow run: {e}")