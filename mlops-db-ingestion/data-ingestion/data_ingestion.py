# data_ingestion.py
import pandas as pd
import time
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from dotenv import load_dotenv
from datetime import datetime
import os

from loader_binance import fetch_btc_data
from loader_gold import fetch_gold_data
from loader_vix import fetch_vix_data
from loader_ndx import fetch_ndx_data


load_dotenv()

db_passwd = os.getenv("PASSWD")
DB_URL = f"mysql+pymysql://mlops_user:{db_passwd}@db/mlops_db?charset=utf8mb4"
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

def connect_to_db_with_sqlalchemy() -> Connection:
    max_retries = 20
    for i in range(max_retries):
        try:
            conn = engine.connect()  # or: with engine.begin() as conn: ...
            print("DB 연결 성공!")
            return conn
        except Exception as e:
            print(f"DB 연결 실패: {e}. DB가 시작될 때까지 대기... ({i+1}/{max_retries})")
            time.sleep(3)
    raise RuntimeError("DB 연결에 실패했습니다. DB 컨테이너 상태를 확인해주세요.")

def load_dataframe_to_db(conn: Connection, df: pd.DataFrame, table_name: str):
    print(f"테이블 '{table_name}'에 데이터 적재 시작...")
    df.to_sql(name=table_name, con=conn, if_exists='append', index=False)
    print(f"테이블 '{table_name}'에 데이터 적재 완료!")

if __name__ == '__main__':
    start = datetime(2024, 1, 1)
    end = datetime(2025, 7, 1)

    conn = connect_to_db_with_sqlalchemy()

    print("BTC 데이터 수집 중...")
    btc_df = fetch_btc_data(start, end)
    load_dataframe_to_db(conn, btc_df, "btc_data")

    print("GOLD 데이터 수집 중...")
    gold_df = fetch_gold_data("2024-01-01", "2025-07-01")
    load_dataframe_to_db(conn, gold_df, "gold_data")

    print("VIX 데이터 수집 중...")
    vix_df = fetch_vix_data("2024-01-01", "2025-07-01")
    load_dataframe_to_db(conn, vix_df, "vix_data")

    print("NDX100 데이터 수집 중...")
    ndx_df = fetch_ndx_data("2024-01-01", "2025-07-01")
    load_dataframe_to_db(conn, ndx_df, "ndx100_data")

    print("모든 작업 완료!")
