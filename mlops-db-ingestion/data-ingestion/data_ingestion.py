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

import logging

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

db_passwd = os.getenv("MYSQL_USER_PASSWORD")
DB_URL = f"mysql+pymysql://mlops_user:{db_passwd}@localhost:3306/mlops_db?charset=utf8mb4"
engine = create_engine(DB_URL, pool_pre_ping=True, future=True, connect_args={'autocommit': True})

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
    if df.empty:
        print("데이터프레임이 비어있어 DB에 저장하지 않습니다.")
        return
    try:
        df.to_sql(name=table_name, con=conn, if_exists='append', index=False)
        print(f"테이블 '{table_name}'에 데이터 적재 완료!")
    except Exception as e:
        print(f"DB 적재 중 오류 발생: {e}")

if __name__ == '__main__':
    conn = connect_to_db_with_sqlalchemy()

    while True:
        try:
            # 1. db에서 마지막 시간 가져오기
            try:
                latest_timestamp_str = pd.read_sql("SELECT MAX(datetime) FROM integrated_data", conn).iloc[0, 0]
                if latest_timestamp_str is None or pd.isna(latest_timestamp_str):
                    start = datetime(2024, 1, 1)
                else:
                    start = pd.to_datetime(latest_timestamp_str) + pd.Timedelta(hours=1)
            except Exception as e:
                print(f"Could not get latest timestamp: {e}")
                start = datetime(2024, 1, 1)

            # 2. 개별 데이터 수집
            print("BTC 데이터 수집 중...")
            btc_df = fetch_btc_data(start)

            print("NDX100 데이터 수집 중...")
            ndx_df = fetch_ndx_data(start.strftime("%Y-%m-%d"))

            print("VIX 데이터 수집 중...")
            vix_df = fetch_vix_data(start.strftime("%Y-%m-%d"))

            print("GOLD 데이터 수집 중...")
            gold_df = fetch_gold_data(start.strftime("%Y-%m-%d"))

            # 3. 전처리 및 병합 과정 추가
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

            # 4. 통합된 데이터프레임을 'integrated_data' 테이블에 적재
            load_dataframe_to_db(conn, final_df, "integrated_data")

            print("모든 작업 완료! 1시간 후 다시 시작합니다.")
            time.sleep(3600) # 1 hour
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60) # wait for 1 minute before retrying
