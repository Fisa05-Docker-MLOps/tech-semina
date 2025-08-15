
---
title: "MLflow 모델 로딩 및 Inference 서버 트러블슈팅 기록"
date: 2025-08-12
tags: [mlflow, pytorch, docker, inference, troubleshooting]
---

# MLflow 모델 로딩 및 Inference 서버 트러블슈팅

## 1. 문제 상황

- FastAPI 기반 Inference 서버에서 MLflow에 등록된 PyTorch 모델을 로드하려고 시도함
- 모델 로드 시 아래와 같은 오류 발생
  - `PyTorch 모델 로드 실패`
  - `Unable to locate credentials`
  - `Invalid endpoint: mlflow-artifact-store:9000`
  - `모델 alias 아직 없음` 반복 경고
- `models:/BTC_LSTM_Production@backtest_20240201` 형태로 모델 URI를 사용했음

---

## 2. 원인 및 분석

### 2.1. 모델 URI 형식 문제

- MLflow Registry의 `ModelVersion.source` 필드 값이 `models:/m-xxxxxx` 와 같은 내부 ID 형태로 반환되어 실제 경로로 변환 실패
- `mlflow.pyfunc.load_model()` 에서 모델 URI를 `models:/<model_name>/<version>` 또는 `models:/<model_name>@<alias>` 형식으로 정확히 지정해야 함
- `ModelVersion.version` 값을 이용해 정확한 모델 버전 경로를 지정하는 방식으로 해결 가능

### 2.2. AWS 인증 정보 문제

- MinIO를 S3 호환 저장소로 사용 중, MLflow가 MinIO에 접근 시 AWS 인증 정보(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)가 올바르게 설정되어 있지 않으면 `Unable to locate credentials` 오류 발생
- Docker Compose 환경변수에 해당 값을 명확히 세팅 필요

### 2.3. MLflow S3 Endpoint URL 문제

- 환경변수 `MLFLOW_S3_ENDPOINT_URL` 의 값이 올바르지 않아 `Invalid endpoint` 오류 발생
- 예) `http://mlflow-artifact-store:9000` 처럼 프로토콜 포함 및 정확한 도메인/포트 필요

### 2.4. 모델 alias가 아직 등록되지 않은 경우

- 서버 시작 시점에 alias에 해당하는 모델 버전이 아직 등록되지 않아 `모델 alias 아직 없음` 메시지 반복
- 이를 대비해 `ensure_model_ready` 함수에 재시도 로직을 추가하여 최대 300초까지 대기 후 재시도하도록 구현

---

## 3. 해결 방안

### 3.1. 모델 URI 명확화

```python
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"  # version은 ModelVersion.version 값
```

또는 alias 사용 시

```python
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
```

### 3.2. AWS 자격 증명 환경 변수 설정

```yaml
environment:
  AWS_ACCESS_KEY_ID: ${MINIO_ROOT_USER}
  AWS_SECRET_ACCESS_KEY: ${MINIO_ROOT_PASSWORD}
  MLFLOW_S3_ENDPOINT_URL: http://mlflow-artifact-store:9000
```

### 3.3. 재시도 로직 적용

- alias가 없으면 5초 간격으로 최대 300초까지 대기하며 재시도하도록 구현

---

## 4. 참고 코드 스니펫

```python
def ensure_model_ready(alias: Optional[str] = None, wait_timeout=300, retry_interval=5):
    alias = alias or MODEL_ALIAS
    start_time = time.time()
    while True:
        try:
            model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            app.state.model = model
            app.state.model_alias = alias
            # ...
            break
        except RestException as e:
            if "not found" in str(e) and (time.time() - start_time) < wait_timeout:
                time.sleep(retry_interval)
            else:
                raise
```

---

## 5. 결론

- MLflow 모델 로딩 시 정확한 모델 URI 지정과 S3 호환 스토리지 접근을 위한 AWS 자격 증명 세팅이 중요함
- 서버 시작 시 모델 준비가 안 되어 있을 경우를 대비해 재시도 로직을 넣어 안정성을 높임
- Docker Compose 환경변수 설정에 신경 써야 하며, 프로토콜과 endpoint 주소 형식이 올바른지 꼭 확인 필요

---
