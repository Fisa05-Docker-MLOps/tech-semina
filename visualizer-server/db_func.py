import os
from dotenv import load_dotenv

import numpy as np  # numpy 라이브러리 불러오기
import pandas as pd  # pandas 라이브러리 불러오기
import pymysql # pymysql 라이브러리 불러오기


# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 이제 os.environ을 통해 .env 파일에 정의된 변수를 사용할 수 있습니다.
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = int(os.environ.get('DB_PORT'))
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')

def get_db_connection():
    """
    데이터베이스 연결을 생성하고 반환하는 함수입니다.
    
    Returns:
        pymysql.connections.Connection: 데이터베이스 연결 객체
    """
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
        raise ValueError("환경 변수가 올바르게 설정되지 않았습니다.")

    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            passwd=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8'
        )
    except Exception as e:
        raise ConnectionError(f"데이터베이스 연결에 실패했습니다: {e}")

    return connection


def fetch_all_btc_four_six() -> pd.DataFrame:
    """
    연결 된 DB에서 ohlcv 데이터를 가져오는 함수입니다.

    Args:
        base_datetime (str): 기준 날짜 및 시간 (예: '2023-10-01 00:00:00')
    Returns:
        pd.DataFrame: ohlcv 데이터가 포함된 DataFrame
    """

    with get_db_connection() as conn:
        query_get_ohlcv = f"""
            SELECT datetime, btc_open, btc_high, btc_low, btc_close, btc_volume
            FROM {DB_NAME}.integrated_data
            WHERE datetime BETWEEN '2025-04-01' AND '2025-06-30'
            ;
            """
        df = pd.read_sql_query(
            query_get_ohlcv,
            conn
        )

    return df.sort_values(by='datetime').reset_index(drop=True)

if __name__ == "__main__":
    df = fetch_all_btc("2024-02-01 00:00:00")
    print(df.head(25))
    print(df.count())