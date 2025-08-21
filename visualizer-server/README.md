# 📊 Visualizer Server (비트코인 예측 시각화 대시보드)

## 1) 한줄 소개 (One-line Introduction)
`visualizer-server`는 Streamlit 기반의 웹 대시보드로, 실제 비트코인 가격 데이터와 `mlops_inference_server`에서 제공하는 모델 예측 결과를 시각적으로 비교하고 분석할 수 있도록 돕습니다.

## 2) 주요 기능 (Key Features)
*   **실시간 비트코인 가격 시각화:** 실제 비트코인 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터를 캔들스틱 차트로 표시합니다.
*   **모델 예측 오버레이:** `mlops_inference_server`에서 가져온 다양한 모델의 예측 결과를 실제 가격 차트 위에 라인 그래프로 오버레이하여 시각적으로 비교할 수 있습니다.
*   **인터랙티브 모델 선택:** 사이드바를 통해 MLflow에 등록된 모델 alias 목록을 조회하고, 특정 모델을 선택하여 해당 모델의 예측을 생성하고 시각화할 수 있습니다.
*   **챔피언 모델 예측:** 전체 기간에 대한 챔피언 모델의 예측 결과를 한 번에 불러와 시각화하는 기능을 제공합니다.
*   **간편한 배포:** Streamlit 애플리케이션으로, 쉽게 실행하고 웹으로 접근할 수 있습니다.

## 3) 설치 (Installation)
1.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv .venv
    # Linux/macOS
    source .venv/bin/activate
    # Windows
    .venv\Scripts\activate
    ```
2.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

## 4) 환경 변수 (Environment Variables)
`visualizer-server`는 `mlops_inference_server`와 통신하기 위해 해당 서버의 주소를 환경 변수로 필요로 합니다.

*   **`INFERENCE_SERVER_URL`**: `mlops_inference_server`의 주소 (예: `http://localhost:8000`).
    *   `.env` 파일을 생성하여 설정하거나, 시스템 환경 변수로 설정할 수 있습니다.
    ```ini
    # .env 파일 예시
    INFERENCE_SERVER_URL=http://localhost:8000
    ```

## 5) 실행 방법 (Execution Method)
1.  **`mlops_inference_server` 실행:** `visualizer-server`가 정상적으로 작동하려면 `mlops_inference_server`가 먼저 실행 중이어야 합니다. `mlops_inference_server`의 `README.md`를 참조하여 서버를 시작하세요.
2.  **Streamlit 애플리케이션 실행:**
    ```bash
    streamlit run btc_candlestick_app.py
    ```
    명령어를 실행하면 웹 브라우저가 자동으로 열리며 대시보드에 접속됩니다. (기본 포트: `8501`)

## 6) 사용법 (Usage)
대시보드에 접속하면 다음 기능을 사용할 수 있습니다:
*   **"예측 기준 모델(Alias)을 선택하세요:" 드롭다운:** MLflow에 등록된 모델 alias 목록 중에서 예측을 보고 싶은 모델을 선택합니다.
*   **"선택한 모델로 예측 생성" 버튼:** 선택된 모델 alias를 사용하여 `mlops_inference_server`에 예측을 요청하고, 결과를 차트에 추가합니다.
*   **"Champion Model 예측" 버튼:** `mlops_inference_server`의 `/predict-champion` 엔드포인트를 호출하여 전체 기간에 대한 챔피언 모델의 예측 결과를 가져와 차트에 추가합니다.
*   **"예측 결과 모두 지우기" 버튼:** 현재 차트에 표시된 모든 모델 예측 라인을 제거합니다.
*   **캔들스틱 차트:** 실제 비트코인 가격의 추이를 보여줍니다.
*   **예측 라인:** 각 모델의 예측 결과가 다른 색상의 라인으로 캔들스틱 차트 위에 표시됩니다.
