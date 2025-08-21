# 📈 MLflow Client (Model Training & Backtesting)

## 1) 한줄 소개 (One-line Introduction)
`mlflow-client`는 MLflow를 활용하여 비트코인 가격 예측 LSTM 모델을 학습하고, 다양한 기간에 대한 백테스팅을 자동화하는 클라이언트 애플리케이션입니다.

## 2) 주요 기능 (Key Features)
*   **비트코인 가격 예측 모델 학습:** LSTM 기반의 시계열 예측 모델을 학습합니다.
*   **MLflow 통합:**
    *   **실험 추적:** 모델 학습 과정의 파라미터, 메트릭, 아티팩트(모델, 스케일러 등)를 MLflow에 기록합니다.
    *   **모델 버전 관리:** 학습된 모델을 MLflow Model Registry에 등록하고 버전을 관리합니다.
    *   **모델 Alias 설정:** 특정 날짜를 기준으로 모델에 alias를 부여하여 배포 및 관리를 용이하게 합니다.
*   **자동화된 백테스팅:** 지정된 기간과 간격에 따라 모델 학습을 반복 수행하여 백테스팅을 자동화합니다.
*   **데이터베이스 연동:** MySQL 데이터베이스로부터 학습 데이터를 조회합니다.
*   **Docker 지원:** 컨테이너화되어 일관된 개발 및 실행 환경을 제공합니다.

## 3) 설치 (Installation)

### 로컬 환경 (Local Environment)
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

### Docker 환경 (Docker Environment)
1.  **Docker 이미지 빌드:**
    ```bash
    ./scripts/build.sh
    ```
    또는
    ```bash
    docker build . -t mlflow-client:latest
    ```

## 4) 사용법 (Usage)

### 로컬 실행 (Local Execution)
1.  `.env` 파일에 데이터베이스 연결 정보를 설정합니다. (아래 "환경 변수" 섹션 참조)
2.  `back_test.py` 스크립트를 실행하여 모델 학습 및 백테스팅을 시작합니다.
    ```bash
    python back_test.py
    ```
    이 스크립트는 `back_test.py` 내부에 정의된 `start`, `end`, `interval`에 따라 모델 학습을 반복 수행합니다.

### Docker 실행 (Docker Execution)
1.  `.env` 파일에 데이터베이스 연결 정보를 설정합니다. (아래 "환경 변수" 섹션 참조)
2.  Docker 컨테이너를 실행합니다.
    ```bash
    ./scripts/run.sh
    ```
    또는
    ```bash
    docker run --gpus all -d \
      -v "$(pwd)":/app \
      -w /app \
      --env-file "./.env" \
      mlflow-client:latest \
      python back_test.py
    ```
    *   `--gpus all`: GPU를 사용하는 경우 필요합니다.
    *   `-v "$(pwd)":/app`: 현재 디렉토리를 컨테이너의 `/app` 디렉토리에 마운트합니다.
    *   `-w /app`: 컨테이너 내 작업 디렉토리를 `/app`으로 설정합니다.
    *   `--env-file "./.env"`: `.env` 파일의 환경 변수를 컨테이너 내로 주입합니다.
    *   `mlflow-client:latest`: 빌드한 Docker 이미지 이름입니다.
    *   `python back_test.py`: 컨테이너 실행 시 수행할 명령어입니다.

## 5) 환경 변수 (Environment Variables)
`db.py`에서 데이터베이스 연결을 위해 다음 환경 변수들이 필요합니다. 프로젝트 루트 디렉토리의 `.env.sample` 파일을 참조하여 `.env` 파일을 생성하고 값을 설정해야 합니다.

```
DB_HOST=your_db_host
DB_PORT=your_db_port
DB_NAME=your_db_name
DB_USER=your_db_user
DB_PASSWORD=your_db_password
MLFLOW_TRACKING_URI=http://mlflow-server:5000 # MLflow 서버 주소
```
*   `MLFLOW_TRACKING_URI`는 `mlflow_train.py`에서 MLflow 서버와 통신하기 위해 사용됩니다. Docker Compose 환경에서는 `http://mlflow-server:5000`과 같이 서비스 이름을 사용할 수 있습니다.

## 6) 프로젝트 구조 (Project Structure)
```
mlflow-client/
├── .dockerignore
├── .gitignore
├── back_test.py          # 메인 백테스팅 실행 스크립트
├── db.py                 # 데이터베이스 연결 및 데이터 조회 모듈
├── Dockerfile            # Docker 이미지 빌드 정의
├── requirements.txt      # Python 의존성 목록
├── module/               # 모델 학습 및 데이터 처리 관련 모듈
│   ├── data.py           # 데이터셋 정의 (TimeSeriesDataset)
│   ├── mlflow_train.py   # MLflow 연동 모델 학습 로직
│   ├── model.py          # LSTM 모델 정의 및 MLflow Pyfunc 래퍼
│   └── utils.py          # 유틸리티 함수 (시드 설정, RMSELoss, 날짜 생성 등)
└── scripts/              # Docker 관련 헬퍼 스크립트
    ├── build.sh          # Docker 이미지 빌드 스크립트
    └── run.sh            # Docker 컨테이너 실행 스크립트
```
