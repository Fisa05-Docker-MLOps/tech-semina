import os
import pathlib
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl
import httpx
import joblib
import mlflow
import time
from mlflow.exceptions import RestException
from utils import setup_logger
from db import fetch_all_btc_four_six, get_model_aliases


# ---- 로그 ----
logger = setup_logger(name="uvicorn.error")
# ---- 환경변수 ----
# MLflow가 MinIO에 접근하기 위해 필요한 환경변수들입니다.
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
# os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "BTC_LSTM_Production")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
# --- 요구 shape & 컬럼 인덱스 ---
# pyfunc 모델을 사용하면 이 값들이 모델 내부에 캡슐화될 수 있습니다.
SEQ_LEN_REQUIRED = int(os.getenv("SEQ_LEN_REQUIRED", "12"))
N_FEATURES_REQUIRED = int(os.getenv("N_FEATURES_REQUIRED", "18"))
# 예측에 사용할 피처들의 컬럼 이름을 리스트로 관리하는 것이 좋습니다.
# FEATURE_COLUMNS = [f'feature_{i}' for i in range(N_FEATURES_REQUIRED)]
FEATURE_COLUMNS = [
    'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume',
    'ndx_open', 'ndx_high', 'ndx_low', 'ndx_close',
    'vix_open', 'vix_high', 'vix_low', 'vix_close',
    'gold_open', 'gold_high', 'gold_low', 'gold_close', 'gold_volume'
]

# ---- MLflow 설정 ----
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
app = FastAPI(title="LSTM Inference with MLflow PyFunc", version="2.0.0")

# =========================
# 요청/응답 스키마
# =========================



class alias_response(BaseModel):
    aliases: List[str]


class predict_date_response(BaseModel):
    prediction: List[float]

# =========================
# 모델 및 상태 관리
# =========================
def ensure_model_ready(alias: Optional[str] = None, wait_timeout=300, retry_interval=5):
    """MLflow Registry에서 pyfunc 모델을 로드합니다."""
    alias = alias or MODEL_ALIAS
    # 이미 로드되어 있고 같은 alias면 통과
    if getattr(app.state, "model", None) is not None and getattr(app.state, "model_alias", None) == alias:
        return
    # MLflow가 Registry 조회, MinIO 다운로드, 모델 로드를 한 번에 처리합니다.
    start_time = time.time()
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    while True:
        logger.info(f"MLflow로부터 모델 로딩 시작: {model_uri}")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            app.state.model = model
            app.state.model_alias = alias
            # 로드된 모델의 버전 정보 저장
            client = mlflow.tracking.MlflowClient()
            version_details = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
            app.state.model_version = version_details.version
            logger.info(f"모델 로딩 완료: {REGISTERED_MODEL_NAME} version={version_details.version} (alias: {alias})")
            break
        except RestException as e:
            if "not found" in str(e) and (time.time() - start_time) < wait_timeout:
                logger.warning(f"모델 alias {alias} 아직 없음. {retry_interval}초 후 재시도...")
                time.sleep(retry_interval)
            else:
                logger.exception("MLflow 모델 로드 실패")
                raise RuntimeError(f"모델 로드 실패: {e}") from e


# =========================
# 콜백 유틸
# =========================
async def post_callback(url: str, payload: dict):
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.status_code


# =========================
# 라우트 (Routes)
# =========================
@app.on_event("startup")
def _startup():
    '''
    서버 시작시 모델을 요청하는 메서드
    '''
    try:
        ensure_model_ready(MODEL_ALIAS)
    except Exception as e:
        logger.error(f"초기 모델 로드 실패: {e}")


@app.get("/health")
def health():
    '''
    서버 상태 체크에 쓰이는 함수
    '''
    model = getattr(app.state, "model", None)
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_alias": getattr(app.state, "model_alias", None),
        "model_version": getattr(app.state, "model_version", None),
    }


@app.post("/reload")
def reload_model(alias: Optional[str] = None):
    '''
    새로운 모델을 설정하기 위해 쓰이는 함수
    '''
    try:
        # app.state를 초기화하여 새로운 모델을 로드하도록 강제
        app.state.model = None
        app.state.model_alias = None
        app.state.model_version = None
        ensure_model_ready(alias or MODEL_ALIAS)
        return {
            "status": "reloaded",
            "model_alias": app.state.model_alias,
            "model_version": app.state.model_version,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictPayload, tasks: BackgroundTasks):
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. /reload를 시도하세요.")
    model = app.state.model
    try:
        # 입력 데이터를 pandas DataFrame으로 변환
        x_df = pd.DataFrame(req.X, columns=FEATURE_COLUMNS)
        # pyfunc 모델은 전/후처리 로직을 내장하고 있음
        prediction = model.predict(x_df)
        pred_next_close = float(prediction[0])
    except Exception as e:
        logger.exception("예측 실패")
        raise HTTPException(status_code=400, detail=f"예측 실패: {e}")
    # ---- 콜백 전송(옵션) ----
    posted = False
    if req.callback_url:
        payload = {
            "pred_btc_close_next": pred_next_close,
            "metadata": req.metadata,
            "model_alias": app.state.model_alias,
            "model_version": app.state.model_version,
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
        posted_to_callback=posted,
        model_alias=app.state.model_alias,
        model_version=app.state.model_version,
    )


@app.get("/aliases")
def give_aliases():
    try:
        alias_list = get_model_aliases()
        logger.log(msg="aliases 키로 alias list 전송")
        return {"aliases": alias_list}
    except Exception as e:
        logger.error(msg=f"데이터베이스로부터 모델 alias를 불러오는데 실패했습니다 {e}")