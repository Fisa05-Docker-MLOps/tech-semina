# 📥 MLOps DB Ingestion (데이터 수집 및 적재)

## 1) 한줄 소개 (One-line Introduction)
`mlops-db-ingestion`은 다양한 금융 시장 데이터(비트코인, 금, VIX, 나스닥 100)를 수집, 전처리 및 통합하여 MySQL 데이터베이스에 적재하는 ETL(Extract, Transform, Load) 파이프라인입니다. Docker Compose를 통해 컨테이너화된 서비스로 운영됩니다.

## 2) 주요 기능 (Key Features)
*   **다양한 금융 데이터 수집:** Binance (비트코인), Yahoo Finance (금, VIX, 나스닥 100) 등 여러 소스에서 시계열 데이터를 수집합니다.
*   **데이터 전처리 및 통합:** 수집된 데이터를 시간대 통일, 정렬, 병합(merge_asof), 결측치 처리 등을 통해 통합된 형태로 가공합니다.
*   **MySQL 데이터베이스 설정:** `init.sql` 스크립트를 통해 필요한 데이터베이스(`mlops_db`) 및 테이블(`btc_data`, `gold_data`, `vix_data`, `ndx100_data`, `integrated_data`)을 자동으로 생성합니다.
*   **자동화된 데이터 적재:** 전처리된 최종 데이터를 `integrated_data` 테이블에 주기적으로 적재합니다.
*   **Docker 기반 운영:** 모든 구성 요소가 Docker 컨테이너로 실행되어 환경 설정의 복잡성을 줄이고 일관된 운영 환경을 제공합니다.

## 3) 환경 변수 (Environment Variables)
데이터베이스 연결 및 인증을 위해 다음 환경 변수들이 필요합니다. 프로젝트 루트 디렉토리의 `.env.sample` 파일을 참조하여 `.env` 파일을 생성하고 값을 설정해야 합니다.

```ini
# .env 파일 예시
MYSQL_ROOT_PASSWORD="your_mysql_root_password" # MySQL root 계정 비밀번호
PASSWD="your_mlops_user_password" # mlops_user 계정 비밀번호 (일반적으로 MYSQL_ROOT_PASSWORD와 동일하게 설정)
```
*   `MYSQL_ROOT_PASSWORD`: `docker-compose.yml`에서 MySQL 컨테이너의 root 비밀번호를 설정하는 데 사용됩니다.
*   `PASSWD`: `data_ingestion.py`에서 `mlops_user`로 데이터베이스에 연결할 때 사용되는 비밀번호입니다. `db-setup/init.sql`에서 `mlops_user`를 생성할 때 설정된 비밀번호와 일치해야 합니다.

## 4) 실행 방법 (Execution Method)

### 1. 레포지토리 클론 (Clone Repository)
```bash
git clone <레포 URL> # 실제 레포 URL로 대체
cd <폴더명> # 해당 디렉토리로 이동
```

### 2. `.env` 파일 생성 및 설정
`.env.sample` 파일을 복사하여 `.env` 파일을 생성하고, 위 "환경 변수" 섹션에 따라 필요한 비밀번호를 설정합니다.
```bash
cp .env.sample .env
```

### 3. Docker Compose로 서비스 실행
모든 서비스를 빌드하고 백그라운드에서 실행합니다.
```bash
docker compose up -d --build
```

### 4. 컨테이너 상태 확인
```bash
docker compose ps
```
`mlops_db`와 `mlops-etl` 컨테이너가 모두 `running` 상태인지 확인합니다. `mlops-etl`은 데이터 적재를 완료하면 종료될 수 있습니다.

## 5) DB 접속 및 데이터 확인 (DB Access and Data Verification)

### DB 컨테이너 접속
```bash
docker exec -it mlops_db bash
```

### MySQL 접속
컨테이너 내부에서 MySQL 클라이언트로 접속합니다. 비밀번호는 `.env` 파일에 설정한 `MYSQL_ROOT_PASSWORD`를 사용합니다.
```bash
mysql -u root -p
```

### 데이터 확인
`mlops_db` 데이터베이스를 선택하고, 각 테이블에 데이터가 성공적으로 적재되었는지 확인합니다.
```sql
USE mlops_db;
SELECT COUNT(*) FROM btc_data;
SELECT COUNT(*) FROM gold_data;
SELECT COUNT(*) FROM ndx100_data;
SELECT COUNT(*) FROM vix_data;
SELECT COUNT(*) FROM integrated_data; -- 최종 통합 데이터
```
`integrated_data` 테이블에 데이터가 적재되면, `mlops-etl` 컨테이너의 역할이 완료된 것입니다.

## 6) 프로젝트 구조 (Project Structure)
```
mlops-db-ingestion/
├── .env.sample             # 환경 변수 샘플 파일
├── .gitignore
├── README.md               # 현재 문서
├── docker-compose.yml      # 전체 서비스 정의 및 오케스트레이션
├── data-ingestion/         # 데이터 수집 및 적재 컨테이너 (Python)
│   ├── Dockerfile          # data-ingestion 서비스 Docker 이미지 빌드 정의
│   ├── requirements.txt    # Python 의존성 목록
│   ├── data_ingestion.py   # 메인 데이터 수집 및 통합 로직
│   ├── loader_binance.py   # 바이낸스 비트코인 데이터 로더
│   ├── loader_gold.py      # 금 데이터 로더
│   ├── loader_ndx.py       # 나스닥 100 데이터 로더
│   └── loader_vix.py       # VIX 데이터 로더
└── db-setup/               # 데이터베이스 초기 설정
    └── init.sql            # MySQL DB 및 테이블 생성 스크립트
```
