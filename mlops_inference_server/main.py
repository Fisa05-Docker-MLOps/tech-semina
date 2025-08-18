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
from db import fetch_all_btc_four_six, get_model_aliases, fetch_all_features
from datetime import datetime


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
class PredictPayload(BaseModel):
    # 단일 시퀀스 2D: [T, F]
    X: List[List[float]] = Field(..., description="[T,F] 형식의 입력 데이터")
    callback_url: Optional[HttpUrl] = None
    metadata: Optional[dict] = None
class PredictResponse(BaseModel):
    pred_btc_close_next: float
    # pyfunc 모델은 입력의 원본 값을 알 수 없으므로, 필요하다면 클라이언트가 직접 계산해야 합니다.
    # actual_btc_close_last: float
    posted_to_callback: bool
    model_alias: str
    model_version: str



class alias_response(BaseModel):
    aliases: List[str]

class btc_info_response(BaseModel):
    datetime: List[datetime]
    btc_open: List[float]
    btc_high: List[float]
    btc_low: List[float]
    btc_close: List[float]
    btc_volume: List[float]


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
        alias = 'backtest_' + alias
        ensure_model_ready(alias or MODEL_ALIAS)
        return {
            "status": "reloaded",
            "model_alias": app.state.model_alias,
            "model_version": app.state.model_version,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패: {e}")

@app.post("/predict")
async def predict(req: PredictPayload):
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다. /reload를 시도하세요.")
    model = app.state.model

    try:
        # alias에서 시작일 추출
        start_date = app.state.model_alias.replace("backtest_", "")
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 00:00:00"

        # features 가져오기
        df = fetch_all_features(start_date)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"{start_date} 이후 데이터가 없습니다.")

        # 순차 예측
        predictions = []
        for i in range(len(df)):
            x_df = pd.DataFrame([df.loc[i, FEATURE_COLUMNS].tolist()], columns=FEATURE_COLUMNS)
            pred = float(model.predict(x_df)[0])
            predictions.append(pred)

        return {
            "start_date": start_date,
            "predictions": predictions
        }

    except Exception as e:
        logger.exception("예측 실패")
        raise HTTPException(status_code=400, detail=f"예측 실패: {e}")


@app.get("/aliases")
def give_aliases():
    '''
    모델의 전체 alias를 반환하는 api 추가
    '''
    try:
        alias_list = get_model_aliases()
        logger.info(msg="aliases 키로 alias list 전송")
        return {"aliases": alias_list}
    except Exception as e:
        logger.error(msg=f"데이터베이스로부터 모델 alias를 불러오는데 실패했습니다 {e}")


@app.get("/btc-info", response_model=btc_info_response)
def give_btc_info():
    '''
    2025.04 ~ 2025.06 기간의 btc 정보를 반환하는 api
    '''
    try:
        btc_df = fetch_all_btc_four_six()
        logger.info(msg="btc 정보 전송")

        return {
            "datetime": btc_df["datetime"].tolist(),
            "btc_open": btc_df["btc_open"].tolist(),
            "btc_high": btc_df["btc_high"].tolist(),
            "btc_low": btc_df["btc_low"].tolist(),
            "btc_close": btc_df["btc_close"].tolist(),
            "btc_volume": btc_df["btc_volume"].tolist(),
        }
    except Exception as e:
        logger.error(msg=f"데이터베이스로부터 btc 정보를 불러오는데 실패했습니다 {e}")
        raise HTTPException(status_code=500, detail=f"btc 정보 조회 실패: {e}")