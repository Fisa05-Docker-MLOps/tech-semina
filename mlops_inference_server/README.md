# LSTM Inference via MLflow + MinIO (FastAPI)

## 1) 설치 (Installation)
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 실행 (Running the Server)

### 추론 서버 실행 (Run Inference Server)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 콜백 서버 실행 (Run Callback Server - for testing callbacks)
```bash
uvicorn callback_server:app --reload --port 9001
```

## 3) API 엔드포인트 (API Endpoints)

FastAPI 서버는 다음 엔드포인트를 제공합니다:

### 1. `GET /health`
*   **설명:** 서버의 상태를 확인합니다. 현재 로드된 모델의 정보(로드 여부, alias, 버전)를 반환합니다.
*   **응답 예시:**
    ```json
    {
      "status": "ok",
      "model_loaded": true,
      "model_alias": "staging",
      "model_version": "1"
    }
    ```

### 2. `POST /reload`
*   **설명:** 새로운 모델 alias로 모델을 다시 로드합니다.
*   **요청 파라미터 (Query Parameter):**
    *   `alias` (string, optional): 로드할 모델의 alias. 지정하지 않으면 기본 alias가 사용됩니다.
*   **응답 예시:**
    ```json
    {
      "status": "reloaded",
      "model_alias": "backtest_20250327",
      "model_version": "2"
    }
    ```
*   **사용 예시 (curl):**
    ```bash
    curl -X POST "http://localhost:8000/reload?alias=backtest_20250327"
    ```

### 3. `POST /predict`
*   **설명:** 특정 alias에 해당하는 모델을 사용하여 비트코인 가격을 예측합니다. 예측에 필요한 데이터는 서버 내부에서 조회됩니다.
*   **요청 파라미터 (Query Parameter):**
    *   `alias` (string, required): 예측에 사용할 모델의 alias (예: `20250327`). 이 alias는 `start_date`로 사용됩니다.
*   **응답 예시:**
    ```json
    {
      "start_date": "2025-03-27 00:00:00",
      "predictions": [
        {"datetime": "2025-03-27T00:00:00", "prediction": 12345.67},
        // ...
      ]
    }
    ```
*   **사용 예시 (curl):**
    ```bash
    curl -X POST "http://localhost:8000/predict?alias=20250327"
    ```

### 4. `GET /predict-champion`
*   **설명:** 여러 모델 alias에 대해 순차적으로 예측을 수행하고, 마지막 alias는 전체 기간에 대해 예측합니다. 백테스팅 시나리오에 사용됩니다.
*   **응답 예시:**
    ```json
    {
      "start_date": "2025-03-27 00:00:00",
      "predictions": [
        {"datetime": "2025-03-27T00:00:00", "prediction": 12345.67},
        // ...
      ]
    }
    ```
*   **사용 예시 (curl):**
    ```bash
    curl "http://localhost:8000/predict-champion"
    ```

### 5. `GET /aliases`
*   **설명:** MLflow 모델 레지스트리에 등록된 모든 모델 alias 목록을 반환합니다.
*   **응답 예시:**
    ```json
    {
      "aliases": ["staging", "production", "backtest_20250327", "backtest_20250328"]
    }
    ```
*   **사용 예시 (curl):**
    ```bash
    curl "http://localhost:8000/aliases"
    ```

### 6. `GET /btc-info`
*   **설명:** 특정 기간(2025.04 ~ 2025.06)의 비트코인 시세 정보를 반환합니다.
*   **응답 예시:**
    ```json
    {
      "datetime": ["2025-04-01T00:00:00", ...],
      "btc_open": [60000.0, ...],
      "btc_high": [61000.0, ...],
      "btc_low": [59000.0, ...],
      "btc_close": [60500.0, ...],
      "btc_volume": [1000.0, ...]
    }
    ```
*   **사용 예시 (curl):**
    ```bash
    curl "http://localhost:8000/btc-info"
    ```

## 4) 콜백 기능 설명 (Callback Functionality Explanation)
이 프로젝트의 `mlops_inference_server`는 내부적으로 비동기 HTTP 요청을 통해 콜백을 수행하는 `post_callback` 유틸리티 함수를 포함하고 있습니다. 이는 예측 결과를 다른 서비스로 전달하는 등의 시나리오에 활용될 수 있습니다.

`callback_server.py`는 이러한 콜백 메커니즘을 테스트하기 위한 별도의 예시 서버입니다. `main.py`의 API 엔드포인트(`GET /predict` 등)는 직접적으로 `callback_url`을 요청 파라미터로 받지 않습니다. 콜백 기능은 `main.py` 내부 로직에서 필요에 따라 `post_callback` 함수를 호출하여 구현될 수 있습니다.

**콜백 서버 실행 예시:**
```bash
uvicorn callback_server:app --reload --port 9001
```
위 명령어로 콜백 서버를 실행한 후, `main.py` 내부에서 `post_callback` 함수를 호출하는 로직을 추가하여 예측 결과를 `http://localhost:9001/result`와 같은 콜백 URL로 전송하는 시나리오를 구성할 수 있습니다.
