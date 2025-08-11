## 파일 구조
```
├── data-ingestion/ # 데이터 수집 및 적재 컨테이너 (Python)
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── data_ingestion.py
│ ├── **.env (.env.sample 참고)**
│ └── ... (4개의 data load 파이썬 파일)
│
├── db-setup/
│ └── init.sql # 최초 DB/테이블 생성 스크립트
├── docker-compose.yml # 전체 서비스 정의
├── **.env (.env.sample 참고)**
└── README.md
```

## 실행 방법

### 1. 레포 클론
```bash
git clone <레포 URL>
cd <폴더명>
```

### 2. .env  파일 만들기
```bash
cp .env.sample .env
```
.env 내용을 환경에 맞게 수정(슬랙에 공유)



### 3. 실행

```bash
docker compose up -d --build
docker compose ps
```

### DB 접속 
```bash 
docker exec -it mlops_db bash
```

```bash
mysql -u root -p
```
-> 패스워드 입력(슬랙)

### 데이터 확인 
```bash 
USE mlops_db;
SELECT COUNT(*) FROM btc_data;
SELECT COUNT(*) FROM gold_data;
SELECT COUNT(*) FROM ndx100_data;
SELECT COUNT(*) FROM vix_data;
```
