# data-ingestion/Dockerfile

FROM python:3.11-slim

# 워킹 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# 기본 실행 명령
CMD ["python", "data_ingestion.py"]
