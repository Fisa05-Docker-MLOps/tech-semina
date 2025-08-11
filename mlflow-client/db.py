import os
from dotenv import load_dotenv

import numpy as np  # numpy 라이브러리 불러오기
import pandas as pd  # pandas 라이브러리 불러오기
import pymysql # pymysql 라이브러리 불러오기


# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 이제 os.environ을 통해 .env 파일에 정의된 변수를 사용할 수 있습니다.
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = int(os.environ.get('DB_PORT', '3306'))
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


def get_ohlcv(base_datetime: str) -> pd.DataFrame:
    """
    연결 된 DB에서 ohlcv 데이터를 가져오는 함수입니다.

    Args:
        base_datetime (str): 기준 날짜 및 시간 (예: '2023-10-01 00:00:00')
    Returns:
        pd.DataFrame: ohlcv 데이터가 포함된 DataFrame
    """

    df_all = []
    with get_db_connection() as conn:
        # 아직 table 명이 정해지지 않음 -> 수정&확인 필요
        for symbol in ['btc', 'gold', 'ndx100', 'vix']:
            query_get_ohlcv = """
                SELECT *
                FROM %s.%s
                WHERE datetime BETWEEN DATE_SUB(%s, INTERVAL 1 MONTH) AND %s;
                """
            df = pd.read_sql_query(
                query_get_ohlcv,
                conn,
                params=[DB_NAME, symbol, base_datetime, base_datetime]
            )
            df_all.append(df)
        
    if not df_all:
        return pd.DataFrame()

    return pd.DataFrame(df_all, ignore_index=True).sort_values(by='Datetime').reset_index(drop=True)
