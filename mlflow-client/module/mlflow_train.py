import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.data.pandas_dataset import from_pandas

from db import get_ohlcv
from module.utils import seed_everything, RMSELoss
from module.data import TimeSeriesDatasetXY
from module.model import LSTM, LstmWithScaler, train_epoch, predict


def train_model_with_month(base_date: str, alias: str):
    # -------------------------------------------------------------------
    # 1. MLflow 서버 및 실험 설정
    # -------------------------------------------------------------------
    # 중앙 MLflow 서버의 주소를 설정합니다. (환경 변수로 설정하는 것을 권장)
    # 예: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mysql+pymysql://... --default-artifact-root s3://...
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000") # 실제 환경에서는 MLflow 서버의 IP 주소 또는 도메인
    EXPERIMENT_NAME = "BTC Close Prediction with LSTM"
    REGISTERED_MODEL_NAME = "BTC_LSTM_Production"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 결과를 저장할 실험(Experiment)의 이름을 지정합니다. 없으면 새로 생성됩니다.
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -------------------------------------------------------------------
    # 2. 데이터 준비
    # -------------------------------------------------------------------

    data = get_ohlcv(base_datetime=base_date)
    TARGET = 'btc_close'
    SEED = 42

    data['datetime'] = pd.to_datetime(data['datetime'])

    x_train = data.drop(columns=['datetime']).copy()
    y_train = data[TARGET].copy()

    # 데이터 스케일링
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(x_train)
    y_scaler.fit(y_train.values.reshape(-1, 1))

    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).flatten()

    # torch Dataset & DataLoader 객체 생성
    SEQ_LEN = 12
    BATCH_SIZE = 64
    LR = 0.001
    EPOCH = 100

    seed_everything(SEED)
    train_ds = TimeSeriesDatasetXY(x_train_scaled, y_train_scaled, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    print("data ready")

    # -------------------------------------------------------------------
    # 3. MLflow 실험 실행 (Run)
    # -------------------------------------------------------------------
    run_name = f"LSTM_{base_date}"
    # 'with' 구문을 사용하면 run이 끝날 때 자동으로 종료됩니다.
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        exp_id = run.info.experiment_id
        print(f"MLflow Run ID: {run_id}")

        # --- 학습 데이터 기록 ---
        training_dataset = from_pandas(
            x_train,
            name="BTC Training Features"
        )
        mlflow.log_input(training_dataset, context="training")
        print("Logged Train Dataset")

        # --- 파라미터(Parameters) 기록 ---
        # 모델 학습에 사용된 하이퍼파라미터를 기록합니다.
        params = {
            "input_size": x_train_scaled.shape[1],
            "hidden_size": 64,
            "num_layers": 1,
            "seq_len": SEQ_LEN,
            "learning_rate": LR,
            "epochs": EPOCH,
            "batch_size": BATCH_SIZE
        }
        mlflow.log_params(params)
        print("Logged Parameters:", params)

        # --- 모델 학습 ---
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(SEED)
        model = LSTM(
            input_size=params['input_size'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            output_size=1
        ).to(DEVICE)
        criterion = RMSELoss()
        optimizer = Adam(model.parameters(), lr=LR)

        print("모델 학습 시작...")
        for epoch in range(EPOCH):
            train_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE
            )

        # --- 메트릭(Metrics) 기록 ---
        # 모델 성능 지표를 기록합니다.
        y_pred = predict(model, train_loader, DEVICE)
        mse_loss = MSELoss()(torch.tensor(y_pred.flatten()), torch.tensor(y_train_scaled[SEQ_LEN:])).item()
        mlflow.log_metric("mse_loss", mse_loss)
        print(f"Logged Metrics: {mse_loss:.4f}")

        # --- 태그(Tags) 기록 ---
        # 실험을 구분하기 위한 추가 정보를 기록합니다.
        mlflow.set_tag("model_type", "LSTM")
        mlflow.set_tag("developer", "LJH")

        # --- 아티팩트(Artifacts) 기록 ---
        # 1. 학습된 모델 자체를 아티팩트로 기록
        # 이 함수는 모델, 환경정보(conda.yaml), 모델 시그니처 등을 함께 저장하여
        # 나중에 추론 서버에서 쉽게 불러올 수 있게 해줍니다.
        # registered_model_name을 지정하면 모델 레지스트리에 자동으로 등록됩니다.
        
        # Scaler 아티팩트 등록
        os.makedirs("./scaler", exist_ok=True)
        scaler_x_path = "./scaler/x_scaler.pkl"
        scaler_y_path = "./scaler/y_scaler.pkl"
        joblib.dump(x_scaler, scaler_x_path)
        joblib.dump(y_scaler, scaler_y_path)

        sample_data = x_train.head(SEQ_LEN)

        # 스케일러 포함 예측 파이프라인 정의
        pipeline_model = LstmWithScaler(
            model=model.cpu(),
            x_scaler=x_scaler,
            y_scaler=y_scaler
        )

        # pyfunc 모델로 파이프라인 로깅
        model_info = mlflow.pyfunc.log_model(
            name="btc-lstm-model", # 아티팩트 저장소 내의 폴더 이름
            python_model=pipeline_model,
            # 의존성이 있는 사용자 정의 코드 경로 지정
            infer_code_paths=["./module/model.py"], 
            # 모델과 함께 저장할 추가 파일들
            artifacts={
                "x_scaler": scaler_x_path,
                "y_scaler": scaler_y_path
            },
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=sample_data
        )
        print("\nPyTorch LSTM 모델이 MLflow에 성공적으로 기록되었습니다.")

        new_version = model_info.registered_model_version
        client = MlflowClient()
        # 3. 생성된 버전에 날짜 정보를 담은 '별칭' 설정!
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=alias,
            version=new_version
        )
        print(f"✅ Version {new_version}에 '{alias}' 별칭 설정 완료")
        
        # 4. (참고) 기록된 모델 다시 불러오기
        # logged_model_uri = f"runs:/{run.info.run_id}/lstm-model"
        # loaded_model = mlflow.pytorch.load_model(logged_model_uri)
        # >>> 로컬 파일 경로로 불러오기
        # print("\n--- 모델 로드 테스트 시작 ---")
        # try:
        #     # 하드코딩된 실험 ID 대신, 방금 실행한 정보로 동적으로 경로 생성
        #     model_path = f"./mlruns/{exp_id}/{run_id}/artifacts/btc-lstm-model"
        #     if os.path.exists(model_path):
        #         loaded_model = mlflow.pyfunc.load_model(model_path)
        #         print("✅ 파일 경로로 모델 로드 성공:", loaded_model)
        #     else:
        #         # runs:/ 스키마로 로드 시도 (서버에 artifact-root가 올바르게 설정된 경우)
        #         logged_model_uri = f"runs:/{run_id}/btc-lstm-model"
        #         loaded_model = mlflow.pyfunc.load_model(logged_model_uri)
        #         print("✅ runs:/ URI로 모델 로드 성공:", loaded_model)

        # except Exception as e:
        #     print(f"❌ 모델 로드 실패: {e}")
        #     print("팁: mlflow server 실행 시 --default-artifact-root 옵션이 올바르게 설정되었는지 확인하세요.")

    print("\nMLflow Run Completed.")