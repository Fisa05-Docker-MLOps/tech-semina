# MLOps Project Troubleshooting Log

This document summarizes the troubleshooting steps and resolutions encountered while setting up and debugging the MLOps project.

## 1. `mlflow-backend-store` (MySQL) Failed to Start

*   **Problem**: The `mlflow-backend-store` container (MySQL database) failed to start after `docker-compose up -d`.
*   **Diagnosis**: `docker logs mlflow-backend-store` showed `ERROR 1396 (HY000) at line 74: Operation CREATE USER failed for 'mlops_user'@'%'`.
*   **Root Cause**: The `mlops-db-ingestion/db-setup/init.sql` script contained `CREATE USER` and `GRANT` statements. Docker's MySQL entrypoint automatically handles user creation and privilege granting based on `MYSQL_USER`, `MYSQL_PASSWORD`, and `MYSQL_DATABASE` environment variables defined in `docker-compose.yml`. These explicit SQL commands in `init.sql` were redundant and caused a conflict, leading to the error.
*   **Resolution**: Removed the conflicting `CREATE USER` and `GRANT` statements from `mlops-db-ingestion/db-setup/init.sql`.

## 2. `inference-server` Waiting for Model Availability

*   **Problem**: The user wanted the `inference-server` to wait until a model was available in MLflow before fully starting. Initially, `depends_on` only ensured the `mlflow-server` was running, not that a model was registered.
*   **Initial Attempt**: Added `training-client` to `inference-server`'s `depends_on` list in `docker-compose.yml`. This ensured `inference-server` would start only after `training-client` completed its initial run.
*   **Refinement**: For more robust and dynamic waiting, a dedicated Python script (`start_server.py`) was created for the `inference-server`. This script polls the MLflow Model Registry for the specified model alias and only proceeds to start the Uvicorn server once the model is found.
*   **Resolution**:
    *   Created `mlops_inference_server/start_server.py` with model polling logic.
    *   Modified `inference-server`'s `command` in `docker-compose.yml` to `["python", "start_server.py"]`.

## 3. `inference-server` `boto3` Module Not Found

*   **Problem**: After implementing `start_server.py`, `inference-server` logs showed `An unexpected error occurred: No module named 'boto3'`.
*   **Diagnosis**: The error indicated a missing Python dependency.
*   **Root Cause**: The `boto3` library, essential for MLflow's interaction with S3-compatible artifact stores (like MinIO), was not listed in `mlops_inference_server/requirements.txt`.
*   **Resolution**: Added `boto3` to `mlops_inference_server/requirements.txt`.

## 4. `inference-server` `ValueError: path in endpoint is not allowed` (MinIO Client)

*   **Problem**: After `boto3` was installed, `inference-server` crashed with `ValueError: path in endpoint is not allowed` during `minio_client` initialization in `main.py`.
*   **Diagnosis**: Logs showed `INFO: Initializing Minio client with endpoint: http://mlflow-artifact-store:9000, secure: False`. The `minio` Python client library expects the `endpoint` argument to be `host:port` without any scheme (`http://` or `https://`).
*   **Root Cause**: The `MINIO_ENDPOINT` environment variable in `docker-compose.yml` for `inference-server` was incorrectly set to `http://mlflow-artifact-store:9000`.
*   **Resolution**: Removed the `http://` prefix from `MINIO_ENDPOINT` in `docker-compose.yml` for the `inference-server` service, changing it to `mlflow-artifact-store:9000`.

## 5. `main.py` `MlflowException` During Model Load (Alias vs. Source URI)

*   **Problem**: Even after `start_server.py` successfully loaded the model, `main.py`'s `_startup` function (which calls `ensure_model_ready`) still failed with `mlflow.exceptions.MlflowException: No versions of model with name 'BTC_LSTM_Production' and stage 'staging' found`.
*   **Diagnosis**: `start_server.py` successfully loaded the model using `mlflow.pyfunc.load_model(model_source_uri)` (where `model_source_uri` was `models:/...`), but `main.py` was trying to load it using `models:/<model_name>/<alias>` URI. The error message suggested `mlflow.pyfunc.load_model` had issues resolving aliases directly or expected a stage.
*   **Root Cause**: The `ensure_model_ready` function in `main.py` was attempting to load the model using a `models:/<model_name>/<alias>` URI, which was problematic for `mlflow.pyfunc.load_model` in this context.
*   **Resolution**: Modified `mlops_inference_server/main.py`'s `ensure_model_ready` function to retrieve the model's direct artifact `source` URI (e.g., `mv.source`) from `ml_client.get_model_version_by_alias()` and then use that `model_source_uri` with `mlflow.pyfunc.load_model()`.

## 6. `main.py` `ValueError: Unsupported artifact URI` During Scaler Load

*   **Problem**: After fixing the model loading in `main.py`, the `ensure_model_ready` function then failed with `ValueError: Unsupported artifact URI: models:/m-7983d5c85e9d48358dc1cdc38437e5a8` when trying to parse the `source` URI for scaler download.
*   **Diagnosis**: The `source` attribute of `ModelVersion` (e.g., `mv.source`) returns a `models:/` URI, but the custom `parse_s3_uri` function expects an `s3://` URI. Additionally, the scalers (`x_scaler.pkl`, `y_scaler.pkl`) were not being logged as MLflow artifacts alongside the model in the `training-client`.
*   **Root Cause**: The `main.py` was trying to parse an MLflow-specific `models:/` URI as a direct `s3://` URI for downloading scalers. The scalers were not properly managed as MLflow artifacts.
*   **Resolution (Pending)**:
    1.  **Modify `mlflow-client/module/mlflow_train.py`**: Log `x_scaler.pkl` and `y_scaler.pkl` as MLflow artifacts using `mlflow.log_artifact` within the MLflow run.
    2.  **Modify `mlops_inference_server/main.py`**: After loading the model, use `mlflow.artifacts.download_artifacts` (which understands `runs:/` URIs) to download the scalers from the model's associated run artifact URI. This will replace the custom `parse_s3_uri` and `download_prefix_from_minio` for scalers.
