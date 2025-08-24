import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Docker 환경이 아닌 로컬에서 실행할 경우 .env 파일 로드
load_dotenv()

DB_HOST = os.getenv("MYSQL_HOST")
DB_PASSWD = os.getenv("MYSQL_USER_PASSWORD")

# 필수 환경 변수 확인
if not DB_HOST or not DB_PASSWD:
    print("오류: MYSQL_HOST 또는 MYSQL_USER_PASSWORD 환경 변수가 설정되지 않았습니다.", file=sys.stderr)
    sys.exit(1)

DB_URL = f"mysql+pymysql://mlops_user:{DB_PASSWD}@{DB_HOST}:3306/mlops_db?charset=utf8mb4"

try:
    # DB 연결 (연결 타임아웃 5초)
    engine = create_engine(DB_URL, connect_args={'connect_timeout': 5})
    with engine.connect() as conn:
        # 'integrated_data' 테이블 존재 여부 확인
        table_exists_query = """
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = 'mlops_db' AND table_name = 'integrated_data'
        """
        table_exists = pd.read_sql(table_exists_query, conn).iloc[0, 0]

        if table_exists:
            # 테이블에 데이터가 있는지 확인
            row_count_query = "SELECT COUNT(*) FROM integrated_data"
            row_count = pd.read_sql(row_count_query, conn).iloc[0, 0]
            
            if row_count > 0:
                print("Healthcheck 성공: 'integrated_data' 테이블에서 데이터가 확인되었습니다.")
                sys.exit(0)  # Healthy
            else:
                print("Healthcheck 실패: 'integrated_data' 테이블이 비어있습니다.")
                sys.exit(1)  # Unhealthy
        else:
            print("Healthcheck 실패: 'integrated_data' 테이블이 존재하지 않습니다.")
            sys.exit(1) # Unhealthy

except Exception as e:
    print(f"Healthcheck 실패: 오류 발생 - {e}", file=sys.stderr)
    sys.exit(1)  # Unhealthy
