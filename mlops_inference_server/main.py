import os
import io
import pathlib
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl
import httpx

import mlflow
from mlflow.tracking import MlflowClient
from minio import Minio
from minio.error import S3Error

# ---- 로그 ----
logger = logging.getLogger("uvicorn.error")

# ---- 환경변수 ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "BTC_LSTM_Production")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = bool(int(os.getenv("MINIO_SECURE", "0")))

LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/tmp/btc-lstm-model")
X_SCALER_FILENAME = os.getenv("X_SCALER_FILENAME", "x_scaler.pkl")
Y_SCALER_FILENAME = os.getenv("Y_SCALER_FILENAME", "y_scaler.pkl")

# ---- MLflow 설정 ----
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
ml_client = MlflowClient()

# ---- MinIO 클라이언트 ----
logger.info(f"Initializing Minio client with endpoint: {MINIO_ENDPOINT}, secure: {MINIO_SECURE}")
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

app = FastAPI(title="LSTM Inference via MLflow+MinIO", version="1.0.0")

# =========================
# 요청/응답 스키마
# =========================
from typing import List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl

# --- 요구 shape & 컬럼 인덱스 ---
SEQ_LEN_REQUIRED = int(os.getenv("SEQ_LEN_REQUIRED", "12"))
N_FEATURES_REQUIRED = int(os.getenv("N_FEATURES_REQUIRED", "18"))
BTC_CLOSE_INDEX = int(os.getenv("BTC_CLOSE_INDEX", "-1"))  # -1 => 마지막 컬럼

class PredictPayload(BaseModel):    
    # 단일 시퀀스 2D: [T, F] 또는 배치=1의 3D: [1, T, F]
    X: Union[List[List[float]], List[List[List[float]]]] = Field(..., description="[T,F] 또는 [1,T,F]")
    callback_url: Optional[HttpUrl] = None
    metadata: Optional[dict] = None

class PredictResponse(BaseModel):
    pred_btc_close_next: float
    actual_btc_close_last: float
    posted_to_callback: bool
    model_alias: str

def _to_batch3d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 2:
        # [T,F] -> [1,T,F]
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"입력은 [T,F] 또는 [1,T,F]여야 합니다. 현재 shape={arr.shape}")
    if arr.shape[0] != 1:
        raise ValueError(f"이 엔드포인트는 배치=1만 지원합니다. 현재 batch={arr.shape[0]}")
    if arr.shape[1] != SEQ_LEN_REQUIRED or arr.shape[2] != N_FEATURES_REQUIRED:
        raise ValueError(
            f"입력 shape는 [1,{SEQ_LEN_REQUIRED},{N_FEATURES_REQUIRED}] 이어야 합니다. 현재={arr.shape}"
        )
    return arr

def _resolve_close_index(n_features: int) -> int:
    if BTC_CLOSE_INDEX == -1:
        return n_features - 1
    if not (0 <= BTC_CLOSE_INDEX < n_features):
        raise ValueError(f"BTC_CLOSE_INDEX={BTC_CLOSE_INDEX}가 범위(0~{n_features-1})를 벗어났습니다.")
    return BTC_CLOSE_INDEX

# =========================
# 유틸
# =========================
def parse_s3_uri(uri: str):
    # s3://bucket/path/to/prefix
    if not uri.startswith("s3://"):
        raise ValueError(f"지원하지 않는 artifact URI: {uri}")
    without = uri[5:]
    bucket, _, prefix = without.partition("/")
    if not bucket or not prefix:
        raise ValueError(f"S3 URI 파싱 실패: {uri}")
    return bucket, prefix

def ensure_local_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def download_prefix_from_minio(bucket: str, prefix: str, local_dir: str):
    ensure_local_dir(local_dir)
    objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
    count = 0
    for obj in objects:
        rel = obj.object_name[len(prefix):].lstrip("/")  # prefix 제거한 상대경로
        local_path = pathlib.Path(local_dir) / rel
        ensure_local_dir(str(local_path.parent))
        minio_client.fget_object(bucket, obj.object_name, str(local_path))
        count += 1
    if count == 0:
        raise RuntimeError(f"MinIO에서 prefix에 해당하는 객체가 없습니다: s3://{bucket}/{prefix}")
    logger.info(f"MinIO에서 {count}개 객체 다운로드: s3://{bucket}/{prefix} -> {local_dir}")

def resolve_model_source_from_registry(model_name: str, alias: str) -> str:
    mv = ml_client.get_model_version_by_alias(name=model_name, alias=alias)
    # source: s3://<bucket>/<...>/artifacts/btc-lstm-model
    return mv.source

def try_load_pickle(path: pathlib.Path):
    if not path.exists():
        return None
    import joblib
    try:
        return joblib.load(path)
    except Exception as e:
        logger.warning(f"스케일러 로드 실패({path}): {e}")
        return None

def load_pytorch_model(model_dir: str):
    # 커스텀 클래스 로드를 위해 훈련 시 사용했던 패키지가 import 가능해야 함.
    # ex) `pip install -e .` 로 module.* 를 배포
    try:
        model = mlflow.pytorch.load_model(model_dir)
        model.eval()
        return model
    except Exception as e:
        logger.exception("PyTorch 모델 로드 실패")
        raise RuntimeError(f"모델 로드 실패: {e}") from e

def ensure_model_ready(alias: Optional[str] = None):
    """모델 디렉토리 동기화 + 모델, 스케일러 로드"""
    alias = alias or MODEL_ALIAS
    # 이미 로드되어 있고 같은 alias면 통과
    if getattr(app.state, "model", None) is not None and getattr(app.state, "model_alias", None) == alias:
        return

    logger.info(f"Loading model '{REGISTERED_MODEL_NAME}' with alias '{alias}'...")
    try:
        # Get model version info by alias
        mv = ml_client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=alias)
        model_version = mv.version
        model_source_uri = mv.source # This is the s3:// URI or models:/ URI from MLflow

        # Use mlflow.pyfunc.load_model with the direct source URI
        model = mlflow.pyfunc.load_model(model_source_uri)
        app.state.model = model
        app.state.model_alias = alias
        logger.info(f"Model '{REGISTERED_MODEL_NAME}' (version {model_version}) loaded successfully from source URI.")

        # Scalers are usually logged as artifacts alongside the model.
        # We need to download them from the artifact URI associated with the model version.
        # This requires getting the model version info first.
        mv = ml_client.get_model_version_by_alias(name=REGISTERED_MODEL_NAME, alias=alias)
        source = mv.source # This will be the s3:// URI
        bucket, prefix = parse_s3_uri(source)
        
        # Download scalers to LOCAL_MODEL_DIR
        download_prefix_from_minio(bucket, prefix, LOCAL_MODEL_DIR)

        x_scaler = try_load_pickle(pathlib.Path(LOCAL_MODEL_DIR) / X_SCALER_FILENAME)
        y_scaler = try_load_pickle(pathlib.Path(LOCAL_MODEL_DIR) / Y_SCALER_FILENAME)
        app.state.x_scaler = x_scaler
        app.state.y_scaler = y_scaler
        logger.info("Scalers loaded successfully.")

    except Exception as e:
        logger.exception(f"Failed to load model '{REGISTERED_MODEL_NAME}' with alias '{alias}'.")
        raise RuntimeError(f"모델 로드 실패: {e}") from e

# 콜백 POST
async def post_callback(url: str, payload: dict):
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.status_code

# =========================
# 라우트
# =========================
@app.on_event("startup")
def _startup():
    try:
        ensure_model_ready(MODEL_ALIAS)
    except Exception as e:
        # 시작 시 로드 실패해도, 이후 /reload 로 복구 가능하게만 기록
        logger.error(f"초기 모델 로드 실패: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_alias": getattr(app.state, "model_alias", None),
        "x_scaler": app.state.x_scaler is not None if hasattr(app.state, "x_scaler") else False,
        "y_scaler": app.state.y_scaler is not None if hasattr(app.state, "y_scaler") else False,
    }

@app.post("/reload")
def reload_model(alias: Optional[str] = None):
    try:
        ensure_model_ready(alias or MODEL_ALIAS)
        return {
            "status": "reloaded",
            "model_alias": app.state.model_alias,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패: {e}")

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictPayload, tasks: BackgroundTasks):
    # 모델 준비
    ensure_model_ready(MODEL_ALIAS)

    model = app.state.model
    x_scaler = getattr(app.state, "x_scaler", None)
    y_scaler = getattr(app.state, "y_scaler", None)

    try:
        x = _to_batch3d(req.X)                 # [1,T,F]
        B, T, F = x.shape
        close_idx = _resolve_close_index(F)

        # ---- 실제 마지막 날 종가 추출 (역스케일 시도) ----
        last_row = x[0, -1, :].copy()          # [F]
        actual_last_close = float(last_row[close_idx])
        if x_scaler is not None:
            try:
                inv = x_scaler.inverse_transform(last_row.reshape(1, -1))  # [1,F]
                actual_last_close = float(inv[0, close_idx])
            except Exception as e:
                logger.warning(f"x_scaler inverse_transform 실패(스킵): {e}")

        # ---- 예측(다음 날 종가) ----
        with torch.no_grad():
            xt = torch.from_numpy(x)           # CPU 추론
            yhat = model(xt)                   # [1,1] 가정
        yhat = yhat.squeeze().cpu().numpy()    # -> 스칼라 또는 shape (1,)

        # y 역스케일 시도
        if y_scaler is not None:
            try:
                yhat = y_scaler.inverse_transform(np.array(yhat, dtype=np.float32).reshape(-1, 1)).flatten()[0]
            except Exception as e:
                logger.warning(f"y_scaler inverse_transform 실패(스킵): {e}")
                yhat = float(np.array(yhat).flatten()[0])
        else:
            yhat = float(np.array(yhat).flatten()[0])

        pred_next_close = float(yhat)

    except Exception as e:
        logger.exception("예측 실패")
        raise HTTPException(status_code=400, detail=f"예측 실패: {e}")

    # ---- 콜백 전송(옵션) ----
    posted = False
    if req.callback_url:
        payload = {
            "pred_btc_close_next": pred_next_close,
            "actual_btc_close_last": actual_last_close,
            "metadata": req.metadata
        }
        async def _send():
            try:
                await post_callback(str(req.callback_url), payload)
                logger.info(f"콜백 전송 성공 -> {req.callback_url}")
            except Exception as e:
                logger.error(f"콜백 전송 실패 -> {req.callback_url}: {e}")
        tasks.add_task(_send)
        posted = True

    return PredictResponse(
        pred_btc_close_next=pred_next_close,
        actual_btc_close_last=actual_last_close,
        posted_to_callback=posted,
        model_alias=getattr(app.state, "model_alias", MODEL_ALIAS),
    )

