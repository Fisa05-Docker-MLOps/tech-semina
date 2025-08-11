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
DB_URL = f"mysql+pymysql://mlops_user:{db_passwd}@mlflow-backend-store/mlops_db?charset=utf8mb4"
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

def connect_to_db_with_sqlalchemy() -> Connection:
    max_retries = 20
    for i in range(max_retries):
        try:
            conn = engine.connect()
            print("DB 연결 성공!")
            return conn
        except Exception as e:
            print(f"DB 연결 실패: {e}. DB가 시작될 때까지 대기... ({i+1}/{max_retries})")
            time.sleep(3)
    raise RuntimeError("DB 연결에 실패했습니다. DB 컨테이너 상태를 확인해주세요.")

def load_dataframe_to_db(conn: Connection, df: pd.DataFrame, table_name: str):
    print(f"테이블 '{table_name}'에 데이터 적재 시작...")
    df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
    print(f"테이블 '{table_name}'에 데이터 적재 완료!")

if __name__ == '__main__':
    start = datetime(2024, 1, 1)
    end = datetime(2025, 7, 1)

    conn = connect_to_db_with_sqlalchemy()

    # 1. 개별 데이터 수집
    print("BTC 데이터 수집 중...")
    btc_df = fetch_btc_data(start, end)

    print("NDX100 데이터 수집 중...")
    ndx_df = fetch_ndx_data("2024-01-01", "2025-07-01")

    print("VIX 데이터 수집 중...")
    vix_df = fetch_vix_data("2024-01-01", "2025-07-01")

    print("GOLD 데이터 수집 중...")
    gold_df = fetch_gold_data("2024-01-01", "2025-07-01")

    # 2. 전처리 및 병합 과정 추가
    print("수집된 데이터 병합 및 전처리 시작...")
    
    # 시간대 통일 (UTC)
    btc_df['datetime'] = pd.to_datetime(btc_df['datetime'], utc=True)
    ndx_df['datetime'] = pd.to_datetime(ndx_df['datetime'], utc=True)
    vix_df['datetime'] = pd.to_datetime(vix_df['datetime'], utc=True)
    gold_df['datetime'] = pd.to_datetime(gold_df['datetime'], utc=True)
    
    # 정렬 (merge_asof 필수)
    btc_df = btc_df.sort_values('datetime')
    ndx_df = ndx_df.sort_values('datetime')
    vix_df = vix_df.sort_values('datetime')
    gold_df = gold_df.sort_values('datetime')

    # BTC와 NDX 병합
    df_merged = pd.merge_asof(
        btc_df,
        ndx_df,
        on='datetime',
        direction='backward',
        tolerance=pd.Timedelta('31min')
    )

    # 컬럼명 변경: _x -> btc_, _y -> ndx_
    df_merged.columns = [
        col.replace('_x', '').replace('_y', '') if col == 'datetime' else
        ('btc_' + col.replace('_x', '') if col.endswith('_x')
         else 'ndx_' + col.replace('_y', '') if col.endswith('_y')
         else col)
        for col in df_merged.columns
    ]

    # btc_volume 컬럼명 통일
    df_merged.rename(columns={"volume" : "btc_volume"}, inplace=True)
    
    # VIX 데이터 병합
    df_merged2 = pd.merge_asof(
        df_merged,
        vix_df,
        on='datetime',
        direction='backward'
    )
    df_merged2 = df_merged2.rename(columns={
        'open': 'vix_open',
        'high': 'vix_high',
        'low': 'vix_low',
        'close': 'vix_close'
    })

    # GOLD 데이터 병합
    df_merged3 = pd.merge_asof(
        df_merged2,
        gold_df,
        on='datetime',
        direction='backward'
    )
    df_merged3 = df_merged3.rename(columns={
        'open': 'gold_open',
        'high': 'gold_high',
        'low': 'gold_low',
        'close': 'gold_close',
        'volume': 'gold_volume'
    })

    # 결측치 처리 (ffill, bfill)
    final_df = df_merged3.ffill().bfill()
    print("데이터 병합 및 전처리 완료!")

    # 3. 통합된 데이터프레임을 'integrated_data' 테이블에 적재
    load_dataframe_to_db(conn, final_df, "integrated_data")


    print("모든 작업 완료!")