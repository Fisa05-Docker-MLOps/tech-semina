import os
import time
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

from module.utils import seed_everything, RMSELoss
from module.data import TimeSeriesDatasetXY
from module.model import LSTM, train_epoch, predict

print("Started")

# -------------------------------------------------------------------
# 1. MLflow 서버 및 실험 설정
# -------------------------------------------------------------------
# 중앙 MLflow 서버의 주소를 설정합니다. (환경 변수로 설정하는 것을 권장)
# 예: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri mysql+pymysql://... --default-artifact-root s3://...
MLFLOW_TRACKING_URI = "http://mlflow:5000" # 실제 환경에서는 MLflow 서버의 IP 주소 또는 도메인
EXPERIMENT_NAME = "BTC Close Prediction with LSTM"
REGISTERED_MODEL_NAME = "BTC_LSTM_Production"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("URI setted")

# 결과를 저장할 실험(Experiment)의 이름을 지정합니다. 없으면 새로 생성됩니다.
mlflow.set_experiment(EXPERIMENT_NAME)

# -------------------------------------------------------------------
# 2. 데이터 준비
# -------------------------------------------------------------------
BASE_DATE = '2024-01-01'
DATA_PATH = './data/integreated_data.csv'
TARGET = 'btc_close'
SEED = 42

data = pd.read_csv(DATA_PATH)
data['datetime'] = pd.to_datetime(data['datetime'])

x_train = data.loc[data['datetime'] < '2024-02-01'].drop(columns=['datetime'])
y_train = data.loc[data['datetime'] < '2024-02-01', TARGET]

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
run_name = f"LSTM_HyperOpt_{time.strftime('%Y%m%d-%H%M%S')}"
# 'with' 구문을 사용하면 run이 끝날 때 자동으로 종료됩니다.
with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    exp_id = run.info.experiment_id
    print(f"MLflow Run ID: {run_id}")

    # --- 학습 데이터 기록 ---
    source_desc = f"Local file: {DATA_PATH}, date < 2024-02-01"
    training_dataset = from_pandas(
        x_train,
        source=source_desc,
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
    
    sample_data, _ = next(iter(train_loader))
    sample_data = sample_data.numpy().astype(np.float32)

    # 💡 PyTorch 모델 기록!
    mlflow.pytorch.log_model(
        pytorch_model=model.cpu(),       # 저장할 PyTorch 모델 객체
        name="btc-lstm-model",  # 아티팩트 저장소 내의 경로
        registered_model_name=REGISTERED_MODEL_NAME, # 모델 레지스트리에 등록할 이름 (선택사항)
        input_example=sample_data # 입력 예시 (모델의 입력 형태를 정의)
    )
    
    print("\nPyTorch LSTM 모델이 MLflow에 성공적으로 기록되었습니다.")

    # 4. (참고) 기록된 모델 다시 불러오기
    # logged_model_uri = f"runs:/{run.info.run_id}/lstm-model"
    # loaded_model = mlflow.pytorch.load_model(logged_model_uri)
    # >>> 로컬 파일 경로로 불러오기
    print("\n--- 모델 로드 테스트 시작 ---")
    try:
        # 하드코딩된 실험 ID 대신, 방금 실행한 정보로 동적으로 경로 생성
        model_path = f"./mlruns/{exp_id}/{run_id}/artifacts/btc-lstm-model"
        if os.path.exists(model_path):
            loaded_model = mlflow.pytorch.load_model(model_path)
            print("✅ 파일 경로로 모델 로드 성공:", loaded_model)
        else:
            # runs:/ 스키마로 로드 시도 (서버에 artifact-root가 올바르게 설정된 경우)
            logged_model_uri = f"runs:/{run_id}/btc-lstm-model"
            loaded_model = mlflow.pytorch.load_model(logged_model_uri)
            print("✅ runs:/ URI로 모델 로드 성공:", loaded_model)

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("팁: mlflow server 실행 시 --default-artifact-root 옵션이 올바르게 설정되었는지 확인하세요.")

print("\nMLflow Run Completed.")

# -------------------------------------------------------------------
# 4. 모델 레지스트리(Model Registry) 상호작용 (개선된 방식)
# -------------------------------------------------------------------
print("\nInteracting with Model Registry using Aliases...")
client = MlflowClient()

# --- 최신 버전 정보 가져오기 ---
# 가장 최근에 등록된 버전의 정보를 가져옵니다.
# search_model_versions는 더 상세한 검색을 제공합니다.
# "name='BTC_LSTM_Production'"은 검색 조건입니다.
latest_version_info = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")[-1]
latest_version = latest_version_info.version

print(f"Latest Model: {latest_version_info.name}, Version: {latest_version}, Current Aliases: {latest_version_info.aliases}")


# --- 모델 버전에 별칭(Alias) 설정하기 ---
# 'Staging' 단계로 보내는 대신 'staging'이라는 별칭을 붙입니다.
# 이 별칭은 해당 모델 이름 내에서 고유하며, 다른 버전에 있던 'staging' 별칭은 자동으로 이 버전으로 옮겨집니다.
alias_name = "staging"
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias=alias_name,
    version=latest_version
)
print(f"✅ Version {latest_version}에 '{alias_name}' 별칭을 성공적으로 설정했습니다.")


# --- (참고) 별칭으로 모델 불러오기 ---
# 나중에 추론 서버 등에서 이 별칭을 사용하여 모델을 불러올 수 있습니다.
try:
    model_by_alias = client.get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=alias_name
    )
    print(f"\n'{alias_name}' 별칭으로 모델을 찾았습니다: Version {model_by_alias.version}")
    # 모델 로드: mlflow.pytorch.load_model(model_by_alias.source)
except Exception as e:
    print(f"별칭으로 모델을 찾는 데 실패했습니다: {e}")