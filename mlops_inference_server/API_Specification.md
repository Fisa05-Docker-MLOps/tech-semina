# MLOps Inference Server API Specification

이 문서는 `mlops_inference_server`의 API 엔드포인트, 요청/응답 스키마 및 기능을 설명합니다. 이 서버는 MLflow에 등록된 LSTM 모델을 사용하여 비트코인 가격을 예측합니다.

## 1. 기본 정보

*   **Base URL:** `http://<your-server-ip>:<port>` (예: `http://localhost:8000`)
*   **API Version:** `2.0.0`
*   **Title:** `LSTM Inference with MLflow PyFunc`

## 2. 엔드포인트

### 2.1. GET /health

서비스의 상태를 확인하고 현재 로드된 모델에 대한 정보를 제공합니다.

*   **Description:** 서버의 가용성 및 모델 로드 상태를 확인합니다.
*   **Method:** `GET`
*   **Path:** `/health`
*   **Responses:**
    *   **`200 OK`**:
        ```json
        {
            "status": "ok",
            "model_loaded": true,
            "model_alias": "staging",
            "model_version": "1"
        }
        ```
        *   `status`: 서비스 상태 (`"ok"`)
        *   `model_loaded`: 모델 로드 여부 (`true` 또는 `false`)
        *   `model_alias`: 현재 로드된 모델의 별칭 (예: `"staging"`, `"Production"`)
        *   `model_version`: 현재 로드된 모델의 버전 (예: `"1"`)

### 2.2. POST /reload

MLflow 모델을 다시 로드합니다. 특정 모델 별칭을 지정하여 다른 버전의 모델을 로드할 수 있습니다.

*   **Description:** 현재 서비스 중인 모델을 MLflow Registry에서 지정된 별칭의 최신 모델로 교체합니다.
*   **Method:** `POST`
*   **Path:** `/reload`
*   **Request Body (Optional):**
    ```json
    {
        "alias": "Production"
    }
    ```
    *   `alias` (string, optional): 로드할 모델의 별칭. 지정하지 않으면 서버의 기본 `MODEL_ALIAS` 환경 변수 값(예: `"staging"`)을 사용합니다.
*   **Responses:**
    *   **`200 OK`**:
        ```json
        {
            "status": "reloaded",
            "model_alias": "Production",
            "model_version": "2"
        }
        ```
        *   `status`: 모델 리로드 상태 (`"reloaded"`)
        *   `model_alias`: 새로 로드된 모델의 별칭
        *   `model_version`: 새로 로드된 모델의 버전
    *   **`500 Internal Server Error`**: 모델 리로드 실패 시.
        ```json
        {
            "detail": "모델 리로드 실패: <에러 메시지>"
        }
        ```

### 2.3. POST /predict

입력 데이터를 받아 비트코인 가격을 예측하고 결과를 반환합니다. 선택적으로 콜백 URL을 통해 예측 결과를 전송할 수 있습니다.

*   **Description:** 제공된 시계열 데이터를 기반으로 다음 비트코인 종가를 예측합니다.
*   **Method:** `POST`
*   **Path:** `/predict`
*   **Request Body:**
    *   **`PredictPayload` 스키마**:
        ```json
        {
            "X": [
                [0.1, 0.2, ..., 0.N],
                [0.1, 0.2, ..., 0.N],
                ... (SEQ_LEN_REQUIRED 만큼 반복)
            ],
            "callback_url": "http://your-callback-server.com/callback",
            "metadata": {
                "request_id": "abc-123",
                "user_id": "user-456"
            }
        }
        ```
        *   `X` (array of array of float, **required**): 예측을 위한 입력 데이터. `[T, F]` 형식의 2D 배열이어야 합니다.
            *   `T`: 시퀀스 길이 (`SEQ_LEN_REQUIRED` 환경 변수에 의해 정의되며, 기본값은 `12`).
            *   `F`: 피처(특징)의 수 (`N_FEATURES_REQUIRED` 환경 변수에 의해 정의되며, 기본값은 `18`).
            *   **예시:** `SEQ_LEN_REQUIRED=12`, `N_FEATURES_REQUIRED=18` 이면 `X`는 `12x18` 크기의 2D 배열이어야 합니다.
        *   `callback_url` (string, optional): 예측 결과가 전송될 콜백 URL. 유효한 HTTP/HTTPS URL이어야 합니다.
        *   `metadata` (object, optional): 콜백 페이로드에 포함될 추가 메타데이터.
*   **Responses:**
    *   **`200 OK`**:
        *   **`PredictResponse` 스키마**:
            ```json
            {
                "pred_btc_close_next": 45000.50,
                "posted_to_callback": true,
                "model_alias": "Production",
                "model_version": "2"
            }
            ```
            *   `pred_btc_close_next` (float): 예측된 다음 비트코인 종가.
            *   `posted_to_callback` (boolean): 예측 결과가 콜백 URL로 성공적으로 전송되었는지 여부.
            *   `model_alias` (string): 예측에 사용된 모델의 별칭.
            *   `model_version` (string): 예측에 사용된 모델의 버전.
    *   **`400 Bad Request`**: 입력 데이터 형식이 잘못되었거나 예측 처리 중 오류 발생 시.
        ```json
        {
            "detail": "예측 실패: <에러 메시지>"
        }
        ```
    *   **`503 Service Unavailable`**: 모델이 아직 로드되지 않은 경우.
        ```json
        {
            "detail": "모델이 로드되지 않았습니다. /reload를 시도하세요."
        }
        ```

## 3. 환경 변수

서버의 동작을 구성하기 위해 다음 환경 변수를 설정할 수 있습니다.

*   `MLFLOW_TRACKING_URI`: MLflow 트래킹 서버의 URI (기본값: `http://127.0.0.1:5000`)
*   `REGISTERED_MODEL_NAME`: MLflow에 등록된 모델의 이름 (기본값: `BTC_LSTM_Production`)
*   `MODEL_ALIAS`: 서버 시작 시 로드할 기본 모델 별칭 (기본값: `staging`)
*   `SEQ_LEN_REQUIRED`: 모델 예측에 필요한 입력 시퀀스 길이 (기본값: `12`)
*   `N_FEATURES_REQUIRED`: 모델 예측에 필요한 피처(특징)의 수 (기본값: `18`)

## 4. 콜백 메커니즘

`/predict` 엔드포인트는 `callback_url`을 통해 비동기 콜백을 지원합니다. 예측이 완료되면 서버는 지정된 `callback_url`로 `POST` 요청을 보냅니다.

*   **콜백 요청 페이로드 예시:**
    ```json
    {
        "pred_btc_close_next": 45000.50,
        "metadata": {
            "request_id": "abc-123",
            "user_id": "user-456"
        },
        "model_alias": "Production",
        "model_version": "2"
    }
    ```
    *   `pred_btc_close_next` (float): 예측된 비트코인 종가.
    *   `metadata` (object, optional): `PredictPayload`에서 전달된 메타데이터.
    *   `model_alias` (string): 예측에 사용된 모델의 별칭.
    *   `model_version` (string): 예측에 사용된 모델의 버전.

## 5. 모델 로드 및 관리

서버는 시작 시 `MODEL_ALIAS`에 지정된 모델을 MLflow Registry에서 자동으로 로드하려고 시도합니다. 모델이 즉시 사용 가능하지 않은 경우, `ensure_model_ready` 함수는 최대 300초 동안 5초 간격으로 재시도를 수행합니다. `/reload` 엔드포인트를 사용하여 수동으로 모델을 다시 로드하거나 다른 별칭의 모델로 전환할 수 있습니다.
